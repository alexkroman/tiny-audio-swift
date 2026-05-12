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
///
/// ## Known limitations
///
/// `stop()` may drop the last ~one-buffer-worth (~85 ms at 48 kHz, 4 096-frame
/// buffer) of audio: the tap callback spawns a detached `Task` per buffer to
/// hop onto the actor, and `Task`s in flight when `stop()` runs may complete
/// after the buffer snapshot. For push-to-talk dictation this is usually
/// acceptable; if you need bit-exact tail capture, pad your gesture by ~100 ms
/// after the user releases.
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
    // Idempotent: calling start() on an already-running engine would crash
    // with an NSException from installTap on a bus that already has a tap.
    // Matches MicrophoneTranscriber.start()'s contract.
    if engine.isRunning { return }
    rawSamples.removeAll(keepingCapacity: true)
    let input = engine.inputNode
    // Always re-query: the cached format from warmUp() can go stale if the
    // user switched input devices between warmUp() and start(). The cache
    // exists only to pay engine.prepare() cost upfront, not as a source of
    // truth for the live format.
    let inputFormat = input.outputFormat(forBus: 0)
    self.inputSampleRate = inputFormat.sampleRate

    // AVAudioInputNode taps must use the node's native format on macOS — asking
    // for a different format throws "Failed to create tap due to format mismatch."
    // Capture in native format and do one-shot SRC at stop() time. Per-buffer SRC
    // with endOfStream after each call drops most output frames (the polyphase
    // filter never accumulates state).
    input.installTap(onBus: 0, bufferSize: 4_096, format: inputFormat) { [weak self] buffer, _ in
      // Take channel 0 as mono. Input may be stereo on built-in/USB mics;
      // we discard channels 1+ rather than averaging because the additional
      // channels rarely carry independent voice content for push-to-talk.
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
    // No-op if never started; avoids removeTap on a tap-less bus (logs a
    // warning) and resampling of an empty buffer with an uninitialized rate.
    guard engine.isRunning else { return [] }
    engine.inputNode.removeTap(onBus: 0)
    engine.stop()
    let raw = rawSamples
    rawSamples.removeAll(keepingCapacity: false)
    return try resampleToMono16k(raw, sampleRate: inputSampleRate)
  }

  private func append(_ chunk: [Float]) {
    rawSamples.append(contentsOf: chunk)
  }

  // Actor deinit can synchronously access let-bound stored properties.
  // engine is non-isolated (it's a let), so engine.stop() is safe here.
  // Without this, dropping the actor while running leaves the mic indicator
  // on until ARC eventually releases the engine.
  deinit {
    if engine.isRunning {
      engine.inputNode.removeTap(onBus: 0)
      engine.stop()
    }
  }
}
