import Foundation
import Testing

@testable import TinyAudio

@Suite("resampleToMono16k")
struct ResamplingTests {
  @Test func passthroughAt16k() throws {
    let samples: [Float] = [0.1, 0.2, 0.3, 0.4, -0.4, -0.3, -0.2, -0.1]
    let result = try resampleToMono16k(samples, sampleRate: 16_000)
    #expect(result == samples)
  }

  @Test func emptyReturnsEmpty() throws {
    let result = try resampleToMono16k([], sampleRate: 48_000)
    #expect(result.isEmpty)
  }

  @Test func downsamples48kTo16k() throws {
    // 100 ms of silence at 48 kHz = 4800 samples; expect ~1600 at 16 kHz
    // (polyphase filter tail tolerance ±200).
    let samples = [Float](repeating: 0, count: 4_800)
    let result = try resampleToMono16k(samples, sampleRate: 48_000)
    #expect(
      result.count > 1_400 && result.count < 1_700,
      "expected ~1600 samples, got \(result.count)")
  }

  @Test func upsamples8kTo16k() throws {
    // 100 ms at 8 kHz = 800 samples; expect ~1600 at 16 kHz.
    let samples = [Float](repeating: 0.5, count: 800)
    let result = try resampleToMono16k(samples, sampleRate: 8_000)
    #expect(
      result.count > 1_400 && result.count < 1_700,
      "expected ~1600 samples, got \(result.count)")
  }

  /// Verifies the resampler is actually doing DSP, not just stretching the
  /// array. A constant-amplitude signal must have its DC level preserved in
  /// the middle of the output (the polyphase filter has transient artifacts
  /// at the boundaries that we skip past).
  @Test func preservesDCLevel() throws {
    let samples = [Float](repeating: 0.5, count: 9_600)  // 200 ms at 48 kHz
    let result = try resampleToMono16k(samples, sampleRate: 48_000)
    #expect(result.count > 200, "result too short to check middle")
    let mid = result.count / 2
    let middle = Array(result[(mid - 100)..<(mid + 100)])
    let mean = middle.reduce(0, +) / Float(middle.count)
    #expect(abs(mean - 0.5) < 0.01, "DC level should be preserved, got \(mean)")
  }

  @Test func throwsOnNonPositiveSampleRate() {
    #expect(throws: TinyAudioError.self) {
      try resampleToMono16k([0.1, 0.2, 0.3], sampleRate: 0)
    }
    #expect(throws: TinyAudioError.self) {
      try resampleToMono16k([0.1, 0.2, 0.3], sampleRate: -1)
    }
    #expect(throws: TinyAudioError.self) {
      try resampleToMono16k([0.1, 0.2, 0.3], sampleRate: .nan)
    }
  }
}
