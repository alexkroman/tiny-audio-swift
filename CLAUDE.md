# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a dual-language repo:

- `swift/` — the shipping product: a SwiftPM package named **TinyAudio** for on-device speech-to-text on iOS / macOS / visionOS, plus two executable CLIs and a SwiftUI demo. This is where almost all work happens.
- `scripts/bundle/` — a Python package used **offline** to build, convert, and upload the MLX model bundle that the Swift SDK downloads at runtime from HuggingFace (`mazesmazes/tiny-audio-swift-bundle`). It does not run as part of the Swift build.

The Swift package lives at `swift/`, not the repo root — every `swift` command needs `--package-path swift` or a `cd swift` first.

## Commands

### Swift (primary)

```bash
cd swift && swift build
cd swift && swift test
cd swift && swift test --filter <SuiteOrTestName>     # e.g. --filter PrefixCache

# Demo apps (workspace exposes iOS + macOS schemes alongside SDK targets)
open swift/TinyAudio.xcworkspace
swift run --package-path swift/Examples/TinyAudioDemo TinyAudioDemo   # macOS

# CLI tools defined in Package.swift
swift run --package-path swift tiny-audio-swift-eval
swift run --package-path swift tiny-audio-vad-bench
```

Formatting (Apple `swift-format`, ships with Swift 5.8+; pre-commit enforces lint):

```bash
swift-format format --in-place --recursive swift/Sources swift/Tests swift/Examples
swift-format lint   --strict   --recursive swift/Sources swift/Tests
```

### Env-gated tests

Several test suites download large models or hit hardware and are skipped by default. Set the corresponding env var to opt in:

| Var                             | Effect                                                                                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TINY_AUDIO_E2E=1`              | Runs `E2ETokenID`, `PrefixCacheTests` — downloads ~460 MB on first run, validates token-ID parity with the Python reference.                     |
| `TINY_AUDIO_PROFILE=1`          | Runs `ProfilingTests` **and** enables per-phase timing inside `ASRPipeline` (adds forced `MLX.eval` syncs — never set in production benchmarks). |
| `TINY_AUDIO_MIC=1`              | Runs live-mic integration test — requires real mic + permission grant.                                                                           |
| `TINYAUDIO_RUN_NETWORK_TESTS=1` | Runs `ChatSessionIntegrationTests` — downloads ~1.4 GB Qwen3.5-2B-OptiQ-4bit.                                                                    |

### Python (bundle build pipeline, rarely needed)

```bash
poetry install
poetry run python -m scripts.bundle.cli build-bundle --projector mazesmazes/tiny-audio-embedded-2
poetry run python -m scripts.bundle.cli convert-decoder --checkpoint <repo-or-path>
poetry run python -m scripts.bundle.cli push-bundle --projector <repo>
```

`build-bundle` writes to `swift/Sources/TinyAudio/Resources/Model/` (gitignored — see `.gitignore`). The shipped SDK does **not** read from that directory; it downloads from HuggingFace at first `Transcriber.load()`. The local-build path is only used for development/eval.

### Pre-commit

`.pre-commit-config.yaml` runs `ruff` + `ruff-format` on Python and `swift-format lint --strict` + `swift test` on Swift. The Swift test hook is slow (multi-minute first build because of MLX-Swift) — move it to `pre-push` by editing `stages: [pre-push]` and running `pre-commit install --hook-type pre-push`. Install with `pre-commit install`.

## Architecture

### Two public actors, two model bundles

The SDK exposes two independent runtime entrypoints, each backed by its own HuggingFace repo:

1. **`Transcriber`** (ASR) → `mazesmazes/tiny-audio-swift-bundle`. Custom-built bundle (encoder + projector + decoder + tokenizer) assembled by `scripts/bundle/`.
2. **`ChatSession`** (general-purpose chat) → `mlx-community/Qwen3.5-2B-OptiQ-4bit`. Stock MLX community model loaded via `MLXLLM.LLMModelFactory`.

Both share `ModelCache.ensureDownloaded` (in `Internal/ModelCache.swift`) which writes to `~/Library/Application Support/TinyAudio/Models/models/<owner>/<repo>/` — the path layout HuggingFace's `HubApi.snapshot` produces natively, so files don't need to be copied/symlinked after download.

> Note: `swift/README.md` still describes the bundled-Git-LFS model — that's stale. The current implementation (and the `feat/runtime-model-download` branch) downloads on first launch.

### Transcriber data flow (`Model/ASRPipeline.swift`)

```
AudioInput → AudioDecoder (16 kHz mono Float32)
          → LogMelSpectrogram        (vDSP via MLXAudioCore)
          → GLMASREncoder            (4-bit quantized)
          → MLPProjector             (fp16)
          → embedding splice         (overwrites <audio> placeholder rows in Qwen3 input embeddings)
          → vendored Qwen3Model      (prefill then greedy decode)
          → token IDs → tokenizer.decode → text
```

Decoded once at end of stream: BPE/SentencePiece is context-sensitive, so you cannot accumulate per-token decodes — `transcribe()` collects token IDs then calls `tokenizer.decode(tokens:)` once.

Key non-obvious details:

- The audio token (`<audio>`) is **added at runtime** to the bundled `tokenizer.json` (`Transcriber.loadTokenizerWithAudioToken`). The bundle ships the base Qwen3 tokenizer; we patch `added_tokens` in-memory to match Python's `add_special_tokens(...)`.
- `Transcriber.load` reads `quantization.{group_size, bits}` from `encoder` config and `decoder_config.json`. Different bundles ship different quantization — fine-tuned decoders use 8-bit/64 (4-bit affine quant degrades EOS prediction); stock decoders use 4-bit/128; fp16 bundles omit the block entirely and skip the `quantize()` call.
- A persistent KV-cache "prefix cache" pre-prefills the constant chat-template prefix once at init (`ASRPipeline.buildPrefixCache`). Each `tokenStream` call reuses it. `bypassPrefixCacheForTesting` (`@_spi(Testing)`) forces the legacy full-prefill path for parity tests.
- Cold-load warmup runs synthetic transcribes at 1 s / 5 s / 15 s of zeros to JIT Metal kernels for the shapes most common in real audio. Skipping this adds ~15 ms per first-call shape.

### Vendored Qwen3 — DO NOT EDIT WITHOUT READING `UPSTREAM.md`

`Sources/TinyAudio/Vendored/Qwen3/Qwen3Model.swift` is a **patched copy** of `mlx-swift-lm`'s Qwen3 (commit pinned in `Vendored/Qwen3/UPSTREAM.md`). The patches:

1. Add an `inputEmbeddings:` parameter to `Qwen3Model.callAsFunction` and `Qwen3ModelInner.callAsFunction` — required so `ASRPipeline` can splice projected audio embeddings in **place of** token-embedding lookups. Upstream Qwen3 has no such hook.
2. Lower visibility to `internal` (module-private) and drop `LLMModel` / `KVCacheDimensionProvider` / `LoRAModel` conformances (unused by `ASRPipeline`).
3. Re-add the quantized KV-cache types from upstream `KVCache.swift` (`QuantizedKVCache`, `quantizedScaledDotProductAttention`, `maybeQuantizeKVCache`).

When rebasing against a newer mlx-swift-lm, follow the recipe in `Vendored/Qwen3/UPSTREAM.md` step-by-step. The `inputEmbeddings` patch is **load-bearing** — without it the ASR pipeline cannot inject audio embeddings.

`ChatSession` uses the **unvendored, stock** `MLXLLM.Qwen3` via `LLMModelFactory` — different code path entirely. The vendored Qwen3 is only used by `Transcriber`.

### Microphone path

`MicrophoneTranscriber` (live mic) layers Silero VAD on top of `Transcriber`:

- VAD model bundled as a CoreML resource at `Resources/silero_vad.mlpackage` (declared in `Package.swift` as `.copy(...)`).
- `VADStreamer` runs the VAD model and emits onset/offset events; complete utterances are routed through `Transcriber.transcribe`.
- The `events` `AsyncStream` is **non-throwing** — per-utterance errors emit `.error(_)` events; the session continues until `stop()` is called.
- Host apps must declare `NSMicrophoneUsageDescription` in `Info.plist`.

### MLX bootstrap

`MLXBootstrap.ensureMetallibAvailable()` exists because `swift run` from the package root does not always co-locate `mlx.metallib` with the binary — it locates the metallib in nearby SwiftPM artifacts and copies it next to the running executable. The demo and `tiny-audio-swift-eval` call this at startup. Library consumers (apps that link TinyAudio via SwiftPM/Xcode) don't need it — Xcode handles the copy.

## Conventions

- Swift 6.0, strict concurrency. Actors guard public API surface (`Transcriber`, `ChatSession`, `MicrophoneTranscriber`). The `SendableBox<T>: @unchecked Sendable` helper in `ASRPipeline.swift` is the documented pattern for crossing a Task boundary with immutable-after-init non-Sendable state.
- All MLX compute happens in `bfloat16` (see `Transcriber.computeDtype`) — both loaded weights and mel inputs are cast at runtime. Don't accidentally introduce fp32 paths.
- Public API surface is intentionally narrow (7 types — see README table). New public types should be added consciously.
- `@_spi(Testing)` is used for test-only hooks (e.g. `setBypassPrefixCache`). Don't mark something `public` just to access it from tests.
- Tests use Swift Testing (`@Test`, `@Suite`, `#expect`) rather than XCTest.
