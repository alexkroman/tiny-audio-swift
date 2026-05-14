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
