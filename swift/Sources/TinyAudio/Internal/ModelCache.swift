import Foundation
import Hub

/// Internal helpers for resolving per-repo cache directories and verifying
/// which files are already on disk. Network logic lives in `ensureDownloaded`.
///
/// Path layout matches `HubApi`'s natural snapshot layout so files don't
/// need to be copied or symlinked after download:
///
///     <App Support>/TinyAudio/Models/models/<owner>/<repo>/<files>
enum ModelCache {
  /// Root directory passed to `HubApi(downloadBase:)`. Created on demand.
  static func cacheRoot() throws -> URL {
    let appSupport = try FileManager.default.url(
      for: .applicationSupportDirectory,
      in: .userDomainMask,
      appropriateFor: nil,
      create: true
    )
    let dir =
      appSupport
      .appendingPathComponent("TinyAudio", isDirectory: true)
      .appendingPathComponent("Models", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  /// On-disk snapshot directory for a repo. This is where files live after
  /// `HubApi.snapshot(from:)` returns, and where `Transcriber` / `ChatSession`
  /// read weights from.
  static func snapshotURL(for repo: String) throws -> URL {
    try cacheRoot()
      .appendingPathComponent("models", isDirectory: true)
      .appendingPathComponent(repo, isDirectory: true)
  }

  static func hasAllFiles(_ files: [String], in dir: URL) -> Bool {
    for name in files {
      if !FileManager.default.fileExists(atPath: dir.appendingPathComponent(name).path) {
        return false
      }
    }
    return true
  }

  /// Ensure `expectedFiles` exist in the snapshot directory for `repo`.
  /// If any are missing, fetch them from the HF repo via `HubApi.snapshot(...)`
  /// (HubApi resumes partial downloads automatically across runs).
  /// Returns the on-disk snapshot directory.
  static func ensureDownloaded(
    repo: String,
    expectedFiles: [String],
    progress: (@Sendable (LoadProgress) -> Void)?
  ) async throws -> URL {
    let target = try snapshotURL(for: repo)
    progress?(.checking)
    if hasAllFiles(expectedFiles, in: target) { return target }

    let hub = HubApi(downloadBase: try cacheRoot())
    do {
      let result = try await hub.snapshot(
        from: Hub.Repo(id: repo, type: .models),
        matching: expectedFiles,
        progressHandler: { p in
          progress?(.downloading(fractionCompleted: p.fractionCompleted))
        }
      )
      guard hasAllFiles(expectedFiles, in: result) else {
        throw TinyAudioError.modelDownloadFailed(
          repo: repo,
          underlying: AnyError(
            NSError(
              domain: "TinyAudio.ModelCache", code: -1,
              userInfo: [
                NSLocalizedDescriptionKey:
                  "snapshot completed but expected files missing at \(result.path)"
              ])
          )
        )
      }
      return result
    } catch let e as TinyAudioError {
      throw e
    } catch {
      throw TinyAudioError.modelDownloadFailed(repo: repo, underlying: AnyError(error))
    }
  }
}
