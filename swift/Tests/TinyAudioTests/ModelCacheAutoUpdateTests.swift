import Foundation
import Testing

@testable import TinyAudio

@Suite("ModelCache auto-update helpers")
struct ModelCacheAutoUpdateHelperTests {
  private func makeTempDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("modelcache-au-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  @Test("readLocalCommit returns nil when sidecar is missing")
  func readLocalCommitMissing() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    #expect(ModelCache.readLocalCommit(in: dir) == nil)
  }

  @Test("readLocalCommit returns trimmed contents when sidecar exists")
  func readLocalCommitPresent() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let sidecar = dir.appendingPathComponent(".commit_hash")
    try "abc123\n".write(to: sidecar, atomically: true, encoding: .utf8)
    #expect(ModelCache.readLocalCommit(in: dir) == "abc123")
  }

  @Test("readLocalCommit returns nil for an empty file")
  func readLocalCommitEmpty() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let sidecar = dir.appendingPathComponent(".commit_hash")
    try "".write(to: sidecar, atomically: true, encoding: .utf8)
    #expect(ModelCache.readLocalCommit(in: dir) == nil)
  }

  @Test("writeLocalCommit creates the sidecar")
  func writeLocalCommitCreates() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try ModelCache.writeLocalCommit("deadbeef", in: dir)
    let contents = try String(
      contentsOf: dir.appendingPathComponent(".commit_hash"),
      encoding: .utf8)
    #expect(contents == "deadbeef")
  }

  @Test("writeLocalCommit overwrites an existing sidecar")
  func writeLocalCommitOverwrites() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    try ModelCache.writeLocalCommit("old", in: dir)
    try ModelCache.writeLocalCommit("new", in: dir)
    #expect(ModelCache.readLocalCommit(in: dir) == "new")
  }
}

private final class StubModelHub: ModelHub, @unchecked Sendable {
  var upstreamCommitToReturn: String?
  var upstreamCommitError: (any Error)?
  var snapshotTargetDir: URL!
  var snapshotFiles: [String] = []
  var snapshotError: (any Error)?

  private(set) var snapshotCallCount = 0

  func upstreamCommit(repo: String) async throws -> String? {
    if let e = upstreamCommitError { throw e }
    return upstreamCommitToReturn
  }

  func snapshot(
    repo: String,
    expectedFiles: [String],
    progress: @Sendable @escaping (Double) -> Void
  ) async throws -> URL {
    snapshotCallCount += 1
    if let e = snapshotError { throw e }
    try FileManager.default.createDirectory(
      at: snapshotTargetDir, withIntermediateDirectories: true)
    for name in snapshotFiles {
      FileManager.default.createFile(
        atPath: snapshotTargetDir.appendingPathComponent(name).path,
        contents: Data("stub".utf8))
    }
    return snapshotTargetDir
  }
}

@Suite("ModelCache.ensureDownloaded auto-update flow")
struct ModelCacheEnsureDownloadedTests {
  private func uniqueRepo() -> String { "test-owner/test-repo-\(UUID().uuidString)" }

  private func cachedSnapshotDir(for repo: String) throws -> URL {
    try ModelCache.snapshotURL(for: repo)
  }

  private func seedCache(repo: String, files: [String], sidecar: String?) throws -> URL {
    let dir = try cachedSnapshotDir(for: repo)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    for name in files {
      FileManager.default.createFile(
        atPath: dir.appendingPathComponent(name).path,
        contents: Data("cached".utf8))
    }
    if let sidecar {
      try ModelCache.writeLocalCommit(sidecar, in: dir)
    }
    return dir
  }

  private func cleanUp(repo: String) {
    if let dir = try? cachedSnapshotDir(for: repo) {
      try? FileManager.default.removeItem(at: dir)
    }
  }

  @Test("first install: no local files, snapshot is called, sidecar written")
  func firstInstall() async throws {
    let repo = uniqueRepo()
    defer { cleanUp(repo: repo) }
    let target = try cachedSnapshotDir(for: repo)
    let hub = StubModelHub()
    hub.upstreamCommitToReturn = "sha-A"
    hub.snapshotTargetDir = target
    hub.snapshotFiles = ["config.json"]

    let result = try await ModelCache.ensureDownloaded(
      repo: repo,
      expectedFiles: ["config.json"],
      progress: nil,
      hub: hub
    )

    #expect(result == target)
    #expect(hub.snapshotCallCount == 1)
    #expect(ModelCache.readLocalCommit(in: result) == "sha-A")
  }

  @Test("cache hit, same SHA: snapshot not called, sidecar unchanged")
  func cacheHitSameSHA() async throws {
    let repo = uniqueRepo()
    defer { cleanUp(repo: repo) }
    let target = try seedCache(repo: repo, files: ["config.json"], sidecar: "sha-A")
    let hub = StubModelHub()
    hub.upstreamCommitToReturn = "sha-A"

    let result = try await ModelCache.ensureDownloaded(
      repo: repo,
      expectedFiles: ["config.json"],
      progress: nil,
      hub: hub
    )

    #expect(result == target)
    #expect(hub.snapshotCallCount == 0)
    #expect(ModelCache.readLocalCommit(in: target) == "sha-A")
  }

  @Test("cache hit, different SHA: snapshot called, sidecar overwritten")
  func cacheHitDifferentSHA() async throws {
    let repo = uniqueRepo()
    defer { cleanUp(repo: repo) }
    let target = try seedCache(repo: repo, files: ["config.json"], sidecar: "sha-A")
    let hub = StubModelHub()
    hub.upstreamCommitToReturn = "sha-B"
    hub.snapshotTargetDir = target
    hub.snapshotFiles = ["config.json"]

    let result = try await ModelCache.ensureDownloaded(
      repo: repo,
      expectedFiles: ["config.json"],
      progress: nil,
      hub: hub
    )

    #expect(result == target)
    #expect(hub.snapshotCallCount == 1)
    #expect(ModelCache.readLocalCommit(in: target) == "sha-B")
  }

  @Test("backfill: sidecar missing, probe succeeds — sidecar written, no re-download")
  func backfill() async throws {
    let repo = uniqueRepo()
    defer { cleanUp(repo: repo) }
    let target = try seedCache(repo: repo, files: ["config.json"], sidecar: nil)
    let hub = StubModelHub()
    hub.upstreamCommitToReturn = "sha-X"

    let result = try await ModelCache.ensureDownloaded(
      repo: repo,
      expectedFiles: ["config.json"],
      progress: nil,
      hub: hub
    )

    #expect(result == target)
    #expect(hub.snapshotCallCount == 0)
    #expect(ModelCache.readLocalCommit(in: target) == "sha-X")
  }

  @Test("probe fails with cache + sidecar: silent fall through")
  func probeFailsWithCacheAndSidecar() async throws {
    let repo = uniqueRepo()
    defer { cleanUp(repo: repo) }
    let target = try seedCache(repo: repo, files: ["config.json"], sidecar: "sha-A")
    let hub = StubModelHub()
    hub.upstreamCommitError = NSError(domain: "test", code: 1)

    let result = try await ModelCache.ensureDownloaded(
      repo: repo,
      expectedFiles: ["config.json"],
      progress: nil,
      hub: hub
    )

    #expect(result == target)
    #expect(hub.snapshotCallCount == 0)
    #expect(ModelCache.readLocalCommit(in: target) == "sha-A")
  }

  @Test("probe fails with cache, no sidecar: silent fall through, sidecar remains missing")
  func probeFailsWithCacheNoSidecar() async throws {
    let repo = uniqueRepo()
    defer { cleanUp(repo: repo) }
    let target = try seedCache(repo: repo, files: ["config.json"], sidecar: nil)
    let hub = StubModelHub()
    hub.upstreamCommitError = NSError(domain: "test", code: 1)

    let result = try await ModelCache.ensureDownloaded(
      repo: repo,
      expectedFiles: ["config.json"],
      progress: nil,
      hub: hub
    )

    #expect(result == target)
    #expect(hub.snapshotCallCount == 0)
    #expect(ModelCache.readLocalCommit(in: target) == nil)
  }

  @Test("probe fails on first install: snapshot still attempted; sidecar skipped when probe nil")
  func probeFailsFirstInstall() async throws {
    let repo = uniqueRepo()
    defer { cleanUp(repo: repo) }
    let target = try cachedSnapshotDir(for: repo)
    let hub = StubModelHub()
    hub.upstreamCommitError = NSError(domain: "test", code: 1)
    hub.snapshotTargetDir = target
    hub.snapshotFiles = ["config.json"]

    let result = try await ModelCache.ensureDownloaded(
      repo: repo,
      expectedFiles: ["config.json"],
      progress: nil,
      hub: hub
    )

    #expect(result == target)
    #expect(hub.snapshotCallCount == 1)
    #expect(ModelCache.readLocalCommit(in: target) == nil)
  }

  @Test("snapshot failure surfaces as TinyAudioError")
  func snapshotErrorPropagates() async throws {
    let repo = uniqueRepo()
    defer { cleanUp(repo: repo) }
    let hub = StubModelHub()
    hub.upstreamCommitToReturn = "sha-A"
    hub.snapshotError = NSError(domain: "test", code: 42)
    hub.snapshotTargetDir = try cachedSnapshotDir(for: repo)

    do {
      _ = try await ModelCache.ensureDownloaded(
        repo: repo,
        expectedFiles: ["config.json"],
        progress: nil,
        hub: hub
      )
      Issue.record("expected ensureDownloaded to throw")
    } catch let error as TinyAudioError {
      if case .modelDownloadFailed(let r, _) = error {
        #expect(r == repo)
      } else {
        Issue.record("wrong TinyAudioError case: \(error)")
      }
    }
  }
}
