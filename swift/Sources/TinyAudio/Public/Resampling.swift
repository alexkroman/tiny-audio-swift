// swift/Sources/TinyAudio/Public/Resampling.swift
@preconcurrency import AVFoundation
import Foundation

/// Resample a mono Float32 buffer to 16 kHz.
///
/// `samples` is interpreted as a single channel of Float32 PCM at
/// `sampleRate` Hz. Output is the same data resampled to 16 kHz.
///
/// - Returns: an empty array if `samples` is empty; the input unchanged if
///   `sampleRate` is already 16 000; otherwise a resampled copy.
/// - Throws: ``TinyAudioError/audioFormatUnsupported(reason:)`` if
///   `AVAudioConverter` cannot be constructed for the requested rate or the
///   conversion itself errors out.
///
/// This is a one-shot resampler — feed the complete buffer once. It is not
/// safe to call repeatedly on streaming chunks; the polyphase filter state is
/// not preserved across calls and tail samples will be dropped.
public func resampleToMono16k(
  _ samples: [Float],
  sampleRate: Double
) throws -> [Float] {
  if samples.isEmpty { return [] }
  if abs(sampleRate - 16_000) < 1 { return samples }

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
  inBuffer.frameLength = AVAudioFrameCount(samples.count)
  samples.withUnsafeBufferPointer { src in
    inBuffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
  }
  return try AudioResampler.toMono16k(inBuffer)
}
