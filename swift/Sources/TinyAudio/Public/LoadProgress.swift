import Foundation

/// Reports progress while loading a TinyAudio model. Same shape for both
/// `Transcriber.load(progress:)` and `ChatSession.load(systemPrompt:progress:)`.
public enum LoadProgress: Sendable {
  /// Verifying which files are already cached.
  case checking
  /// Downloading missing files. `fractionCompleted` is 0.0...1.0
  /// across all files for one repo.
  case downloading(fractionCompleted: Double)
  /// Reading weights from cache into MLX memory.
  case loading
}
