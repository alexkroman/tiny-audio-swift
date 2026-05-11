import Foundation
import Testing

@testable import TinyAudio

@Suite("ModelCache (no network)")
struct ModelCacheTests {
  @Test("snapshotURL is under Application Support / TinyAudio / Models / models / <repo>")
  func snapshotURL() throws {
    let url = try ModelCache.snapshotURL(for: "owner/repo-name")
    let path = url.path
    #expect(path.hasSuffix("/Library/Application Support/TinyAudio/Models/models/owner/repo-name"))
  }

  @Test("cacheRoot is under Application Support / TinyAudio / Models")
  func cacheRoot() throws {
    let url = try ModelCache.cacheRoot()
    let path = url.path
    #expect(path.hasSuffix("/Library/Application Support/TinyAudio/Models"))
  }

  @Test("hasAllFiles returns false when dir empty")
  func hasAllFilesEmpty() throws {
    let tmp = FileManager.default.temporaryDirectory
      .appendingPathComponent("modelcache-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }
    #expect(ModelCache.hasAllFiles(["a.bin", "b.bin"], in: tmp) == false)
  }

  @Test("hasAllFiles returns true when every file exists")
  func hasAllFilesPresent() throws {
    let tmp = FileManager.default.temporaryDirectory
      .appendingPathComponent("modelcache-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }
    for name in ["a.bin", "b.bin"] {
      FileManager.default.createFile(
        atPath: tmp.appendingPathComponent(name).path,
        contents: Data())
    }
    #expect(ModelCache.hasAllFiles(["a.bin", "b.bin"], in: tmp) == true)
  }
}
