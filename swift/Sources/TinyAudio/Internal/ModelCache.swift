import Foundation
import Hub

/// Internal helpers for resolving per-repo cache directories and verifying
/// which files are already on disk. Network logic lives in `ensureDownloaded`.
enum ModelCache {
  /// Repo slugs may contain `/`; replace with `_` so they're path-safe.
  static func directory(for repo: String) throws -> URL {
    let appSupport = try FileManager.default.url(
      for: .applicationSupportDirectory,
      in: .userDomainMask,
      appropriateFor: nil,
      create: true
    )
    let slug = repo.replacingOccurrences(of: "/", with: "_")
    let dir =
      appSupport
      .appendingPathComponent("TinyAudio", isDirectory: true)
      .appendingPathComponent("Models", isDirectory: true)
      .appendingPathComponent(slug, isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  static func hasAllFiles(_ files: [String], in dir: URL) -> Bool {
    for name in files {
      if !FileManager.default.fileExists(atPath: dir.appendingPathComponent(name).path) {
        return false
      }
    }
    return true
  }
}
