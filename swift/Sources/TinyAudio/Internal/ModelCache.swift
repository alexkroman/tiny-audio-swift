import Foundation
import Hub

/// Seam over HubApi so tests can stub network calls.
internal protocol ModelHub: Sendable {
  func upstreamCommit(repo: String) async throws -> String?

  func snapshot(
    repo: String,
    expectedFiles: [String],
    progress: @escaping @Sendable (Double) -> Void
  ) async throws -> URL
}

internal struct LiveModelHub: ModelHub {
  func upstreamCommit(repo: String) async throws -> String? {
    // Single HEAD via the URL-form helper avoids the listing+HEAD round trip
    // that `getFileMetadata(from:matching:)` would do.
    guard
      let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/config.json")
    else { return nil }
    return try await Hub.getFileMetadata(fileURL: url).commitHash
  }

  func snapshot(
    repo: String,
    expectedFiles: [String],
    progress: @escaping @Sendable (Double) -> Void
  ) async throws -> URL {
    let hub = HubApi(downloadBase: try ModelCache.cacheRoot())
    return try await hub.snapshot(
      from: Hub.Repo(id: repo, type: .models),
      matching: expectedFiles,
      progressHandler: { p in progress(p.fractionCompleted) }
    )
  }
}

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

  static let commitSidecarFilename = ".commit_hash"

  static func readLocalCommit(in dir: URL) -> String? {
    let url = dir.appendingPathComponent(commitSidecarFilename)
    guard let data = try? Data(contentsOf: url),
      let raw = String(data: data, encoding: .utf8)
    else { return nil }
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? nil : trimmed
  }

  static func writeLocalCommit(_ sha: String, in dir: URL) throws {
    let url = dir.appendingPathComponent(commitSidecarFilename)
    try sha.write(to: url, atomically: true, encoding: .utf8)
  }

  static func ensureDownloaded(
    repo: String,
    expectedFiles: [String],
    progress: (@Sendable (LoadProgress) -> Void)?,
    hub: ModelHub = LiveModelHub()
  ) async throws -> URL {
    let target = try snapshotURL(for: repo)
    progress?(.checking)
    let localFilesPresent = hasAllFiles(expectedFiles, in: target)
    let upstream = try? await hub.upstreamCommit(repo: repo)

    if localFilesPresent {
      let local = readLocalCommit(in: target)
      if local == nil {
        if let upstream { try? writeLocalCommit(upstream, in: target) }
        return target
      }
      if upstream == nil {
        return target
      }
      if local == upstream {
        return target
      }
    }

    do {
      let result = try await hub.snapshot(
        repo: repo,
        expectedFiles: expectedFiles,
        progress: { fraction in
          progress?(.downloading(fractionCompleted: fraction))
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
      if let upstream {
        try? writeLocalCommit(upstream, in: result)
      }
      return result
    } catch let e as TinyAudioError {
      throw e
    } catch {
      throw TinyAudioError.modelDownloadFailed(repo: repo, underlying: AnyError(error))
    }
  }
}
