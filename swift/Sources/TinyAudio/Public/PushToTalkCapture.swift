@preconcurrency import AVFoundation
import Foundation

/// Push-to-talk microphone capture.
///
/// Sibling to ``MicrophoneTranscriber`` for callers that want the *raw audio*
/// from a hold-to-talk gesture, not a streaming VAD-endpointed transcript.
/// Capture begins on ``start()`` and ends on ``stop()``, which returns the
/// captured audio resampled to 16 kHz mono Float32.
///
/// ## Lifecycle
///
/// ```swift
/// let capture = PushToTalkCapture()
/// await capture.warmUp()              // optional: pre-warm at launch
/// try await capture.start()           // begin capture (mic indicator on)
/// // ... user holds key, speaks ...
/// let samples = try await capture.stop()  // 16 kHz mono Float32
/// ```
///
/// `warmUp()` pre-prepares `AVAudioEngine` and caches the input format so the
/// first `start()` doesn't pay first-time hardware-route discovery latency.
/// It performs no audio I/O and does not trigger the system mic indicator.
///
/// Host apps must declare `NSMicrophoneUsageDescription` in `Info.plist`.
public actor PushToTalkCapture {
  private let engine = AVAudioEngine()
  private var rawSamples: [Float] = []
  private var inputSampleRate: Double = 0
  private var cachedInputFormat: AVAudioFormat?
  private var prepared = false

  public init() {}

  /// Pre-allocate `AVAudioEngine` resources and cache the input format so the
  /// first `start()` is fast. Safe to call multiple times; idempotent.
  public func warmUp() {
    if prepared { return }
    let input = engine.inputNode
    let inputFormat = input.outputFormat(forBus: 0)
    self.cachedInputFormat = inputFormat
    self.inputSampleRate = inputFormat.sampleRate
    engine.prepare()
    prepared = true
  }

  /// Install the input tap and start the engine.
  ///
  /// Throws if `AVAudioEngine.start()` fails (mic permission denied, no input
  /// device, hardware busy, etc.).
  public func start() async throws {
    rawSamples.removeAll(keepingCapacity: true)
    let input = engine.inputNode
    let inputFormat = cachedInputFormat ?? input.outputFormat(forBus: 0)
    self.cachedInputFormat = inputFormat
    self.inputSampleRate = inputFormat.sampleRate

    // AVAudioInputNode taps must use the node's native format on macOS — asking
    // for a different format throws "Failed to create tap due to format mismatch."
    // Capture in native format and do one-shot SRC at stop() time. Per-buffer SRC
    // with endOfStream after each call drops most output frames (the polyphase
    // filter never accumulates state).
    input.installTap(onBus: 0, bufferSize: 4_096, format: inputFormat) { [weak self] buffer, _ in
      guard let chan = buffer.floatChannelData?[0] else { return }
      let chunk = Array(UnsafeBufferPointer(start: chan, count: Int(buffer.frameLength)))
      Task { [weak self] in await self?.append(chunk) }
    }

    if !prepared {
      engine.prepare()
      prepared = true
    }
    do {
      try engine.start()
    } catch {
      input.removeTap(onBus: 0)
      throw error
    }
  }

  /// Stop the engine and return 16 kHz mono Float32 samples.
  ///
  /// Throws ``TinyAudioError/audioFormatUnsupported(reason:)`` if the
  /// captured audio cannot be resampled.
  public func stop() async throws -> [Float] {
    engine.inputNode.removeTap(onBus: 0)
    engine.stop()
    let raw = rawSamples
    rawSamples.removeAll(keepingCapacity: false)
    return try resampleToMono16k(raw, sampleRate: inputSampleRate)
  }

  private func append(_ chunk: [Float]) {
    rawSamples.append(contentsOf: chunk)
  }
}
