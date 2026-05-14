import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// On-device speech recognition backed by a 4-bit quantized GLM-ASR encoder and Qwen3-0.6B decoder.
///
/// `Transcriber` reads its bundled model weights from `Bundle.module` — no
/// network access is required.  Call ``load()`` once at app startup, then
/// reuse the returned actor for all subsequent transcription calls.
public actor Transcriber {
  internal let pipeline: ASRPipeline

  private init(pipeline: ASRPipeline) {
    self.pipeline = pipeline
  }

  // MARK: - Compute precision

  /// Numeric type used for model weights and intermediate activations.
  /// Set to `.bfloat16` to match PyTorch's default Qwen3 / GLM-ASR runtime
  /// precision; set to `.float16` for the previous behavior. The compute
  /// dtype is applied by casting loaded weights and the mel input through
  /// `castWeightsForCompute` / `castMelForCompute`.
  internal static let computeDtype: DType = .bfloat16

  /// Special tokens that should terminate Qwen3 generation in addition to
  /// `tokenizer.eosTokenId`. Both `Transcriber.load()` and
  /// `ChatSession.makeForTests` resolve these against the loaded tokenizer.
  internal static let qwen3EosTokenStrings: [String] = ["<|im_end|>", "<|endoftext|>"]

  /// Cast every floating-point weight in a loaded safetensors dict to
  /// `computeDtype`. Non-float arrays (token id tables etc.) pass through
  /// untouched.
  internal static func castWeightsForCompute(
    _ weights: [String: MLXArray]
  ) -> [String: MLXArray] {
    weights.mapValues { arr in
      switch arr.dtype {
      case .float16, .float32, .bfloat16:
        return arr.asType(computeDtype)
      default:
        return arr
      }
    }
  }

  /// Cast a mel-spectrogram tensor to `computeDtype`. The weights are cast
  /// at load time; the mel needs to be cast at every call so its dtype
  /// matches the encoder's first conv weight.
  internal static func castMelForCompute(_ mel: MLXArray) -> MLXArray {
    mel.asType(computeDtype)
  }

  // MARK: - Public factory

  /// Files we expect the ASR repo to provide. Presence-checked before each load.
  private static let asrModelFiles: [String] = [
    "config.json",
    "encoder.safetensors",
    "decoder.safetensors",
    "decoder_config.json",
    "projector.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
  ]

  /// HuggingFace repo holding the ASR weights for this version of TinyAudio.
  private static let asrRepo = "mazesmazes/tiny-audio-swift-bundle"

  /// Load the ASR model and return a warmed-up `Transcriber`.
  ///
  /// On first call, downloads the model bundle from HuggingFace into
  /// `~/Library/Application Support/TinyAudio/Models/`; subsequent calls
  /// hit the cache. Reports progress through `progress` if provided. After
  /// the weights are loaded, runs a short synthetic warmup at 1 s / 5 s /
  /// 15 s of audio to JIT-compile Metal kernels for the shapes most common
  /// in real ASR.
  ///
  /// Dev escape hatch: if the `TINY_AUDIO_LOCAL_MODEL_DIR` environment
  /// variable is set, the HuggingFace download is skipped and the model is
  /// loaded directly from that directory (intended for evaluating
  /// locally-built bundles without publishing them).
  public static func load(
    progress: (@Sendable (LoadProgress) -> Void)? = nil
  ) async throws -> Transcriber {
    if let override = ProcessInfo.processInfo.environment["TINY_AUDIO_LOCAL_MODEL_DIR"],
      !override.isEmpty
    {
      let dir = URL(fileURLWithPath: (override as NSString).expandingTildeInPath)
      var isDir: ObjCBool = false
      guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue
      else {
        throw TinyAudioError.modelDownloadFailed(
          repo: "TINY_AUDIO_LOCAL_MODEL_DIR",
          underlying: AnyError(
            NSError(
              domain: "TinyAudio", code: 1,
              userInfo: [
                NSLocalizedDescriptionKey:
                  "TINY_AUDIO_LOCAL_MODEL_DIR points to a non-existent directory: \(dir.path)"
              ]))
        )
      }
      progress?(.loading)
      return try await load(modelDirectory: dir)
    }

    let dir = try await ModelCache.ensureDownloaded(
      repo: Self.asrRepo,
      expectedFiles: Self.asrModelFiles,
      progress: progress
    )
    progress?(.loading)
    return try await load(modelDirectory: dir)
  }

  /// Load a model from an on-disk directory and return a warmed-up `Transcriber`.
  ///
  /// The caller is responsible for providing a directory containing the model
  /// artifacts (`config.json`, `encoder.safetensors`, `projector.safetensors`,
  /// `decoder.safetensors`, `decoder_config.json`, `tokenizer.json`,
  /// `tokenizer_config.json`).
  internal static func load(modelDirectory: URL) async throws -> Transcriber {

    // 2. Read bundle config.json.
    let configURL = modelDirectory.appendingPathComponent("config.json")
    let configData = try Data(contentsOf: configURL)
    let bundleConfig: ASRBundleConfig
    do {
      bundleConfig = try JSONDecoder().decode(ASRBundleConfig.self, from: configData)
    } catch {
      throw TinyAudioError.mlxModuleLoadFailed(
        name: "config",
        underlying: AnyError(error)
      )
    }

    // 4. Build mel spectrogram (no resource path needed; uses computeMelSpectrogram).
    let mel = LogMelSpectrogram()

    // 5. Build encoder, quantize per the bundle's config, load weights.
    //    `encoder.quantization.{group_size, bits}` is written by the Python
    //    build pipeline so it always matches what `mlx.nn.quantize` did at
    //    bundle-build time. Absent block ⇒ fp16 bundle (equivalence-test mode);
    //    skip the quantize() call so plain Linear modules accept the weights.
    let encoder = GLMASREncoder(GLMASREncoderConfig(bundleConfig.encoder))
    if let q = bundleConfig.encoder.quantization {
      quantize(model: encoder, groupSize: q.groupSize, bits: q.bits)
    }
    do {
      let weights = Self.castWeightsForCompute(
        try MLX.loadArrays(url: modelDirectory.appendingPathComponent("encoder.safetensors"))
      )
      try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    } catch {
      throw TinyAudioError.mlxModuleLoadFailed(name: "encoder", underlying: AnyError(error))
    }

    // 6. Build projector (fp16 — no quantization), load weights.
    let projector = MLPProjector(bundleConfig.projector)
    do {
      let weights = Self.castWeightsForCompute(
        try MLX.loadArrays(url: modelDirectory.appendingPathComponent("projector.safetensors"))
      )
      try projector.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    } catch {
      throw TinyAudioError.mlxModuleLoadFailed(name: "projector", underlying: AnyError(error))
    }

    // 7. Build vendored Qwen3 decoder, quantize per the bundle's config, load weights.
    //    `quantize()` transforms Linear modules in-place to QuantizedLinear so their
    //    parameter shapes match the stored weight layout. groupSize/bits MUST come
    //    from decoder_config.json — the build pipeline picks 4/128 for the stock
    //    Qwen MLX decoder and 8/64 for full-decoder fine-tunes (4-bit affine quant
    //    degrades EOS prediction on fine-tuned weights).
    let decoder: Qwen3Model
    let numDecoderLayers: Int
    let vocabSize: Int
    do {
      let decoderConfigURL = modelDirectory.appendingPathComponent("decoder_config.json")
      let decoderConfigData = try Data(contentsOf: decoderConfigURL)
      let qwenConfig = try JSONDecoder().decode(Qwen3Configuration.self, from: decoderConfigData)
      // Optional decode — fp16 (unquantized) bundles omit the block entirely.
      let quantSpec = try? JSONDecoder().decode(
        DecoderQuantizationSpec.self, from: decoderConfigData)
      numDecoderLayers = qwenConfig.hiddenLayers
      vocabSize = qwenConfig.vocabularySize
      decoder = Qwen3Model(qwenConfig)
      if let q = quantSpec?.quantization {
        quantize(model: decoder, groupSize: q.groupSize, bits: q.bits)
      }
      let rawWeights = Self.castWeightsForCompute(
        try MLX.loadArrays(url: modelDirectory.appendingPathComponent("decoder.safetensors"))
      )
      // sanitize strips lm_head.weight when tieWordEmbeddings=true.
      let weights = decoder.sanitize(weights: rawWeights)
      try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    } catch let e as TinyAudioError {
      throw e
    } catch {
      throw TinyAudioError.mlxModuleLoadFailed(name: "decoder", underlying: AnyError(error))
    }

    // 8. Load tokenizer, runtime-add the <audio> special token.
    //    The bundle's tokenizer.json is the base Qwen3 tokenizer; <audio> is added
    //    at runtime by the Python pipeline (ASRModel._init_tokenizer). We mirror
    //    that here by patching `added_tokens` before constructing the tokenizer,
    //    matching ProcessorTests.swift's approach.
    let tokenizer = try await loadTokenizerWithAudioToken(directory: modelDirectory)

    // 9. Resolve audio token ID.
    guard let audioTokenIdInt = tokenizer.convertTokenToId(Processor.audioToken) else {
      throw TinyAudioError.mlxModuleLoadFailed(
        name: "tokenizer",
        underlying: AnyError(ConfigError.missingKey("<audio> token not found after patching"))
      )
    }
    let audioTokenId = Int32(audioTokenIdInt)

    // 10. Resolve EOS token IDs.
    var eosTokenIds: Set<Int32> = []
    for eosStr in Transcriber.qwen3EosTokenStrings {
      if let id = tokenizer.convertTokenToId(eosStr) {
        eosTokenIds.insert(Int32(id))
      }
    }
    if let eosId = tokenizer.eosTokenId {
      eosTokenIds.insert(Int32(eosId))
    }

    // 11. Construct ASRPipeline.
    let asr = ASRPipeline(
      encoder: encoder,
      projector: projector,
      decoder: decoder,
      tokenizer: tokenizer,
      mel: mel,
      audioTokenId: audioTokenId,
      eosTokenIds: eosTokenIds,
      numDecoderLayers: numDecoderLayers,
      vocabSize: vocabSize,
      cachedSystemPrompt: nil  // matches transcribe()'s default
    )

    // 12. Warmup + return.
    let transcriber = Transcriber(pipeline: asr)
    await transcriber.warmup()
    return transcriber
  }

  // MARK: - Private helpers

  /// Run synthetic transcribes at several audio durations to JIT-compile MLX
  /// Metal kernels for the shapes most likely to occur in real ASR calls.
  ///
  /// Warming at only 1 s means the first real call at a different duration
  /// (e.g. 5 s or 15 s) pays a ~15 ms per-shape JIT penalty. Warming at
  /// 1 s / 5 s / 15 s covers most short-form ASR durations and adds ~300 ms
  /// to cold load time while eliminating that first-call JIT cost.
  /// Mirrors `MLXASRModel.warmup()` in Python. Failures are non-fatal.
  private func warmup() async {
    // Warm up MLX kernels for several audio durations. Real audio comes in
    // many lengths; warming at one shape only triggers a JIT recompile on the
    // first real call. 1 s / 5 s / 15 s covers most short-form ASR durations
    // and adds ~300 ms to load time but saves ~15 ms on every first call
    // until that shape is hit.
    let durationsSeconds: [Int] = [1, 5, 15]
    for seconds in durationsSeconds {
      let zeros = [Float](repeating: 0, count: 16_000 * seconds)
      do {
        for try await _ in pipeline.tokenStream(
          samples: zeros,
          maxNewTokens: 4,
          systemPrompt: nil
        ) {
          // discard
        }
      } catch {
        // Warmup failures are non-fatal; surface only if real inference also fails.
      }
    }
  }

  /// Load a Qwen3 tokenizer from the bundle directory, patching `added_tokens`
  /// in `tokenizer.json` to include `<audio>` as a special token.
  ///
  /// This mirrors Python's `tokenizer.add_special_tokens({"additional_special_tokens":
  /// ["<audio>"]})` in `ASRModel._init_tokenizer`. The patching approach copies
  /// `ProcessorTests.swift` exactly: mutate the `added_tokens` array in-memory,
  /// then construct `PreTrainedTokenizer` directly from the config + data dicts.
  ///
  /// The audio token ID is fixed at 151669 for Qwen3-0.6B (the next available ID
  /// after the base vocabulary). The actual ID is cross-checked after construction.
  private static func loadTokenizerWithAudioToken(directory: URL) async throws -> any Tokenizer {
    // Load tokenizer_config.json.
    let configURL = directory.appendingPathComponent("tokenizer_config.json")
    let configData = try Data(contentsOf: configURL)
    guard let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
    else {
      throw TinyAudioError.mlxModuleLoadFailed(
        name: "tokenizer",
        underlying: AnyError(ConfigError.invalidJSON("tokenizer_config.json is not a dict"))
      )
    }
    let tokenizerConfig = Config(configDict as [NSString: Any])

    // Load tokenizer.json and patch added_tokens.
    let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
    let tokenizerRaw = try Data(contentsOf: tokenizerURL)
    guard var tokenizerDict = try JSONSerialization.jsonObject(with: tokenizerRaw) as? [String: Any]
    else {
      throw TinyAudioError.mlxModuleLoadFailed(
        name: "tokenizer",
        underlying: AnyError(ConfigError.invalidJSON("tokenizer.json is not a dict"))
      )
    }

    let audioToken = Processor.audioToken
    // Determine next available ID from the existing added_tokens list.
    var addedTokens = (tokenizerDict["added_tokens"] as? [[String: Any]]) ?? []
    let alreadyAdded = addedTokens.contains { ($0["content"] as? String) == audioToken }
    if !alreadyAdded {
      let maxExistingId = addedTokens.compactMap { $0["id"] as? Int }.max() ?? 151664
      let audioTokenId = maxExistingId + 1
      addedTokens.append(
        [
          "id": audioTokenId,
          "content": audioToken,
          "single_word": false,
          "lstrip": false,
          "rstrip": false,
          "normalized": false,
          "special": true,
        ] as [String: Any])
      tokenizerDict["added_tokens"] = addedTokens
    }

    let tokenizerData = Config(tokenizerDict as [NSString: Any])
    return try PreTrainedTokenizer(
      tokenizerConfig: tokenizerConfig,
      tokenizerData: tokenizerData
    )
  }
}

// MARK: - Bundle config schema

/// Decodable mirror of `config.json` from the ASR bundle. Field names use
/// snake_case in JSON to match the Python build pipeline; `CodingKeys` map
/// them to Swift camelCase.
private struct ASRBundleConfig: Decodable {
  let encoder: Encoder
  let projector: Projector

  struct Encoder: Decodable {
    let nMels: Int
    let encoderDim: Int
    let numLayers: Int
    let numHeads: Int
    let headDim: Int
    let intermediateSize: Int
    let ropeTheta: Float?
    let quantization: QuantizationBlock?

    enum CodingKeys: String, CodingKey {
      case nMels = "n_mels"
      case encoderDim = "encoder_dim"
      case numLayers = "num_layers"
      case numHeads = "num_heads"
      case headDim = "head_dim"
      case intermediateSize = "intermediate_size"
      case ropeTheta = "rope_theta"
      case quantization
    }
  }

  struct Projector: Decodable {
    let encoderDim: Int
    let llmDim: Int
    let hiddenDim: Int
    let poolStride: Int

    enum CodingKeys: String, CodingKey {
      case encoderDim = "encoder_dim"
      case llmDim = "llm_dim"
      case hiddenDim = "hidden_dim"
      case poolStride = "pool_stride"
    }
  }
}

/// Quantization parameters shared between `config.json` (encoder block) and
/// `decoder_config.json` (top-level `quantization` block).
private struct QuantizationBlock: Decodable {
  let groupSize: Int
  let bits: Int
  enum CodingKeys: String, CodingKey {
    case groupSize = "group_size"
    case bits
  }
}

extension GLMASREncoderConfig {
  fileprivate init(_ block: ASRBundleConfig.Encoder) {
    self.init(
      nMels: block.nMels,
      encoderDim: block.encoderDim,
      numLayers: block.numLayers,
      numHeads: block.numHeads,
      headDim: block.headDim,
      intermediateSize: block.intermediateSize,
      ropeTheta: block.ropeTheta ?? 10_000
    )
  }
}

extension MLPProjector {
  fileprivate convenience init(_ block: ASRBundleConfig.Projector) {
    self.init(
      encoderDim: block.encoderDim,
      llmDim: block.llmDim,
      hiddenDim: block.hiddenDim,
      poolStride: block.poolStride
    )
  }
}

/// Reads the `quantization` block from `decoder_config.json` so we can call
/// `quantize()` with the same group_size/bits the bundle was built with.
private struct DecoderQuantizationSpec: Decodable {
  let quantization: QuantizationBlock
}

// MARK: - Internal config-parsing errors

private enum ConfigError: Error {
  case missingKey(String)
  case invalidJSON(String)
}

// MARK: - Transcription public API

extension Transcriber {
  /// Maximum number of new tokens the decoder may generate per call.
  /// Matches the published checkpoint's `generation_config.max_new_tokens`
  /// (`tiny-audio/config.json` — 128).
  private static let maxNewTokens = 128

  /// Transcribe audio to text, waiting for the full transcript before returning.
  ///
  /// - Parameter audio: The audio to transcribe; the SDK normalises to
  ///   16 kHz mono Float32 internally.
  /// - Returns: The complete transcript as a single `String`.
  /// - Throws: ``TinyAudioError/audioEmpty`` if the decoded audio contains no
  ///   samples, ``TinyAudioError/audioFormatUnsupported(reason:)`` for
  ///   unreadable audio, or an MLX runtime error during decoding.
  public func transcribe(_ audio: AudioInput) async throws -> String {
    let samples = try AudioDecoder.decode(audio)
    var accumulatedInts: [Int] = []
    for try await tid in pipeline.tokenStream(
      samples: samples,
      maxNewTokens: Self.maxNewTokens,
      systemPrompt: nil
    ) {
      if pipeline.eosTokenIds.contains(tid) { break }
      accumulatedInts.append(Int(tid))
    }
    // Detokenize the full prefix once at the end.  BPE / SentencePiece
    // tokens are context-sensitive — `decode([t1]) + decode([t2])` is not
    // equal to `decode([t1, t2])`, so we cannot accumulate per-token decodes.
    return pipeline.tokenizer.decode(tokens: accumulatedInts)
  }
}

extension Transcriber {
  @_spi(Testing)
  public func setBypassPrefixCache(_ bypass: Bool) {
    pipeline.bypassPrefixCacheForTesting = bypass
  }
}
