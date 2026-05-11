import Foundation
import Testing

@testable import TinyAudio

@Suite("PushToTalkCapture")
struct PushToTalkCaptureTests {
  /// Verifies the public API shape compiles and the type is constructible
  /// without triggering a mic permission prompt. `warmUp()` is safe to call
  /// without permission — it only does `engine.prepare()`, no audio I/O.
  @Test func initAndWarmUpDoNotPromptForMicAccess() async {
    let capture = PushToTalkCapture()
    await capture.warmUp()
    // Confirm the public methods compile with the expected signatures.
    let _: () async throws -> Void = capture.start
    let _: () async throws -> [Float] = capture.stop
  }

  /// Live mic round-trip. Gated behind TINY_AUDIO_MIC=1; speak for ~3 s.
  ///
  /// Run with:
  ///   TINY_AUDIO_MIC=1 swift test --package-path swift --filter PushToTalkCapture
  @Test func liveMicCaptureReturnsSamples() async throws {
    guard ProcessInfo.processInfo.environment["TINY_AUDIO_MIC"] == "1" else {
      print("Skipping liveMicCaptureReturnsSamples: set TINY_AUDIO_MIC=1 to run.")
      return
    }

    let capture = PushToTalkCapture()
    await capture.warmUp()
    try await capture.start()

    print("\n=== Speak now (3 s) ===\n")
    try await Task.sleep(nanoseconds: 3 * 1_000_000_000)

    let samples = try await capture.stop()
    // Expect ~48 000 samples at 16 kHz for 3 s of capture (±20% for engine warmup).
    #expect(samples.count > 30_000, "expected at least 30 k samples, got \(samples.count)")
    #expect(samples.count < 60_000, "expected at most 60 k samples, got \(samples.count)")
    let peak = samples.map { abs($0) }.max() ?? 0
    print("Captured \(samples.count) samples, peak amplitude \(peak)")
  }
}
