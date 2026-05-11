import Foundation
import Testing

@testable import TinyAudio

@Suite("ChatSession API surface")
struct ChatSessionAPITests {
  @Test("GenerationConfig defaults")
  func generationDefaults() {
    let c = GenerationConfig.default
    #expect(c.maxTokens == 512)
    #expect(abs(c.temperature - 0.2) < 0.001)
  }

  @Test("GenerationConfig custom")
  func generationCustom() {
    let c = GenerationConfig(maxTokens: 64, temperature: 0.0)
    #expect(c.maxTokens == 64)
    #expect(c.temperature == 0.0)
  }

  @Test("ChatSession.repoId is the OptiQ 4-bit Qwen3.5-2B build")
  func repoId() {
    #expect(ChatSession.repoId == "mlx-community/Qwen3.5-2B-OptiQ-4bit")
  }
}

/// Network-gated integration test — only runs when TINYAUDIO_RUN_NETWORK_TESTS is set,
/// because it downloads ~1.4 GB on first invocation and requires Apple Silicon Metal.
@Suite(
  "ChatSession integration",
  .disabled(if: ProcessInfo.processInfo.environment["TINYAUDIO_RUN_NETWORK_TESTS"] == nil)
)
struct ChatSessionIntegrationTests {
  @Test("load + respond returns non-empty completion")
  func loadAndRespond() async throws {
    let session = try await ChatSession.load(
      systemPrompt: "You are a helpful assistant. Reply with exactly one short sentence.",
      generation: GenerationConfig(maxTokens: 32, temperature: 0.0)
    )
    var collected = ""
    for try await chunk in session.respond(to: "Say hi.") {
      collected += chunk
    }
    #expect(!collected.isEmpty)
  }
}
