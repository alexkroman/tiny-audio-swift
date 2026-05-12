@preconcurrency import AVFoundation
import Foundation

/// Resample a mono Float32 buffer to 16 kHz.
///
/// `samples` is interpreted as a single channel of Float32 PCM at
/// `sampleRate` Hz. Output is the same data resampled to 16 kHz.
///
/// - Returns: an empty array if `samples` is empty (this function does NOT
///   throw ``TinyAudioError/audioEmpty`` for empty input — that's intentional
///   for push-to-talk callers that no-op silently on zero-length captures);
///   the input unchanged if `sampleRate` is exactly 16 000;
///   otherwise a resampled copy.
/// - Throws: ``TinyAudioError/audioFormatUnsupported(reason:)`` if
///   `AVAudioConverter` cannot be constructed for the requested rate or the
///   conversion itself errors out.
///
/// This is a one-shot resampler — feed the complete buffer once. It is not
/// safe to call repeatedly on streaming chunks; the resampler's internal
/// filter state is not preserved across calls and tail samples will be dropped.
public func resampleToMono16k(
  _ samples: [Float],
  sampleRate: Double
) throws -> [Float] {
  if samples.isEmpty { return [] }
  if sampleRate == 16_000 { return samples }
  guard sampleRate.isFinite, sampleRate > 0 else {
    throw TinyAudioError.invalidArgument(
      reason: "sampleRate must be positive and finite, got \(sampleRate)")
  }

  guard
    let inFormat = AVAudioFormat(
      commonFormat: .pcmFormatFloat32,
      sampleRate: sampleRate,
      channels: 1,
      interleaved: false),
    let inBuffer = AVAudioPCMBuffer(
      pcmFormat: inFormat,
      frameCapacity: AVAudioFrameCount(samples.count))
  else {
    throw TinyAudioError.audioFormatUnsupported(
      reason: "could not build input format/buffer at \(sampleRate) Hz")
  }
  // floatChannelData is non-nil for pcmFormatFloat32 buffers; baseAddress is
  // non-nil because we already returned for empty input above.
  inBuffer.frameLength = AVAudioFrameCount(samples.count)
  samples.withUnsafeBufferPointer { src in
    inBuffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
  }
  return try AudioResampler.toMono16k(inBuffer)
}
