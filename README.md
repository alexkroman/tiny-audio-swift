# tiny-audio-swift

On-device speech-to-text for iOS, macOS, and visionOS, powered by MLX.

- **`swift/`** — the TinyAudio SwiftPM package (SDK, CLIs, and a SwiftUI demo).
- **`scripts/bundle/`** — offline Python pipeline for building the MLX model bundle published to HuggingFace.

## Quick start

```bash
cd swift
swift build
swift test
```

Open `swift/TinyAudio.xcworkspace` for the iOS and macOS demo schemes.

See `swift/README.md` for SDK usage and `CLAUDE.md` for the architecture overview.
