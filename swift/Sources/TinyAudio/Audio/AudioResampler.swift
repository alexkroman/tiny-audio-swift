@preconcurrency import AVFoundation
import Foundation

/// Convert an `AVAudioPCMBuffer` to 16 kHz mono Float32 `[Float]`.
///
/// This is a thin wrapper around `AVAudioConverter`. It downmixes multi-channel
/// audio by averaging channels, then resamples to 16 kHz. Returns the raw
/// Float32 samples.
struct AudioResampler {
  static let targetSampleRate: Double = 16_000

  static func toMono16k(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
    let inFormat = buffer.format
    let frameCount = AVAudioFrameCount(buffer.frameLength)
    guard frameCount > 0 else { return [] }

    let outFormat = AVAudioFormat(
      commonFormat: .pcmFormatFloat32,
      sampleRate: targetSampleRate,
      channels: 1,
      interleaved: false
    )!

    if inFormat == outFormat {
      return Self.copyMono(buffer)
    }

    guard let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
      throw TinyAudioError.audioFormatUnsupported(
        reason: "no AVAudioConverter for \(inFormat) -> \(outFormat)")
    }

    let ratio = outFormat.sampleRate / inFormat.sampleRate
    let outCapacity = AVAudioFrameCount(Double(frameCount) * ratio + 1024)
    guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity) else {
      throw TinyAudioError.audioFormatUnsupported(reason: "could not allocate output buffer")
    }

    try convertOnce(
      converter: converter,
      input: buffer,
      output: outBuffer,
      reuseConverter: false
    )
    return Self.copyMono(outBuffer)
  }

  /// Drive `converter` through a single conversion of `input` into `output`.
  ///
  /// The converter's input callback delivers `input` exactly once and then
  /// signals completion. Pass `reuseConverter: false` for one-shot
  /// conversions where the converter is discarded afterwards; pass `true`
  /// when the same converter will be reused across many calls — sending
  /// `.endOfStream` would finalize it and every subsequent `convert()` would
  /// return zero samples.
  ///
  /// The `InputState` class boxes the one-shot delivery flag because the
  /// input block is `@Sendable` under Swift 6 even though it's called
  /// synchronously on the calling thread; the closure captures by reference.
  static func convertOnce(
    converter: AVAudioConverter,
    input: AVAudioPCMBuffer,
    output: AVAudioPCMBuffer,
    reuseConverter: Bool
  ) throws {
    final class InputState: @unchecked Sendable { var delivered = false }
    let state = InputState()
    var convError: NSError?
    let status = converter.convert(to: output, error: &convError) { _, outStatus in
      if state.delivered {
        outStatus.pointee = reuseConverter ? .noDataNow : .endOfStream
        return nil
      }
      state.delivered = true
      outStatus.pointee = .haveData
      return input
    }
    if status == .error || convError != nil {
      throw TinyAudioError.audioFormatUnsupported(
        reason: "conversion failed: \(convError?.localizedDescription ?? "unknown")"
      )
    }
  }

  private static func copyMono(_ buffer: AVAudioPCMBuffer) -> [Float] {
    let frameCount = Int(buffer.frameLength)
    guard frameCount > 0,
      let channels = buffer.floatChannelData
    else { return [] }
    let channelCount = Int(buffer.format.channelCount)
    if channelCount == 1 {
      return Array(UnsafeBufferPointer(start: channels[0], count: frameCount))
    }
    // Average across channels.
    var out = [Float](repeating: 0, count: frameCount)
    for c in 0..<channelCount {
      let ch = channels[c]
      for i in 0..<frameCount {
        out[i] += ch[i]
      }
    }
    let inv = 1.0 / Float(channelCount)
    for i in 0..<frameCount { out[i] *= inv }
    return out
  }
}
