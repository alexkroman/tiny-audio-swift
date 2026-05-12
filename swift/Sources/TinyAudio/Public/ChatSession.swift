import Foundation
import Hub
import MLXLLM
import MLXLMCommon

/// Generation parameters for ``ChatSession``.
public struct GenerationConfig: Sendable {
  public var maxTokens: Int
  public var temperature: Float

  public init(maxTokens: Int = 512, temperature: Float = 0.2) {
    self.maxTokens = maxTokens
    self.temperature = temperature
  }

  public static let `default` = GenerationConfig()
}

/// On-device chat session backed by `mlx-community/Qwen3.5-2B-OptiQ-4bit`.
///
/// `ChatSession.load(systemPrompt:)` downloads the model on first use,
/// caches it under `~/Library/Application Support/TinyAudio/Models/`,
/// and builds a one-time KV-cache primer for the system prompt. Per-turn
/// `respond(to:)` calls reload that primer cache and only prefill the
/// user message.
public actor ChatSession {
  public static let repoId = "mlx-community/Qwen3.5-2B-OptiQ-4bit"

  private static let expectedFiles: [String] = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "chat_template.jinja",
  ]

  private let container: ModelContainer
  private let generation: GenerationConfig
  private let systemPrompt: String
  private let primerCacheURL: URL?

  private init(
    container: ModelContainer,
    generation: GenerationConfig,
    systemPrompt: String,
    primerCacheURL: URL?
  ) {
    self.container = container
    self.generation = generation
    self.systemPrompt = systemPrompt
    self.primerCacheURL = primerCacheURL
  }

  public static func load(
    systemPrompt: String,
    generation: GenerationConfig = .default,
    progress: (@Sendable (LoadProgress) -> Void)? = nil
  ) async throws -> ChatSession {
    let dir = try await ModelCache.ensureDownloaded(
      repo: repoId,
      expectedFiles: expectedFiles,
      progress: progress
    )
    progress?(.loading)
    let configuration = ModelConfiguration(directory: dir)
    let container = try await LLMModelFactory.shared.loadContainer(
      configuration: configuration
    )
    let primer = try? await buildPrimerCache(
      container: container,
      systemPrompt: systemPrompt
    )
    return ChatSession(
      container: container,
      generation: generation,
      systemPrompt: systemPrompt,
      primerCacheURL: primer
    )
  }

  private static func buildPrimerCache(
    container: ModelContainer,
    systemPrompt: String
  ) async throws -> URL {
    let parameters = GenerateParameters(maxTokens: 4, temperature: 0.0)
    let primer = MLXLMCommon.ChatSession(
      container,
      instructions: systemPrompt,
      generateParameters: parameters
    )
    _ = try await primer.respond(to: "")
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("tinyaudio-chat-primer-\(UUID().uuidString).safetensors")
    try await primer.saveCache(to: url)
    return url
  }

  public nonisolated func respond(
    to userMessage: String
  ) -> AsyncThrowingStream<String, Error> {
    // `let` actor properties are immutable; reads don't need actor isolation.
    let container = self.container
    let cfg = self.generation
    let systemPrompt = self.systemPrompt
    let primerURL = self.primerCacheURL
    let parameters = GenerateParameters(
      maxTokens: cfg.maxTokens,
      temperature: cfg.temperature
    )

    return AsyncThrowingStream { continuation in
      Task {
        do {

          let inner: MLXLMCommon.ChatSession
          if let primerURL,
            let (cache, _) = try? MLXLMCommon.loadPromptCache(url: primerURL)
          {
            inner = MLXLMCommon.ChatSession(
              container, cache: cache, generateParameters: parameters)
          } else {
            inner = MLXLMCommon.ChatSession(
              container,
              instructions: systemPrompt,
              generateParameters: parameters
            )
          }

          for try await chunk in inner.streamResponse(to: userMessage) {
            if Task.isCancelled {
              continuation.finish()
              return
            }
            continuation.yield(chunk)
          }
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
    }
  }
}
