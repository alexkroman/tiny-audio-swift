#!/bin/bash
# Project health check: build + test the SPM package and the macOS demo app.
# Pipes xcodebuild through xcbeautify when available (brew install xcbeautify).
# Runs swift-format lint (ships with Swift 5.8+ via xcrun).
# Runs periphery when available (brew install periphery).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWIFT_DIR="$REPO_ROOT/swift"
DEMO_DIR="$SWIFT_DIR/Examples/TinyAudioDemo"
WORKSPACE="$SWIFT_DIR/TinyAudio.xcworkspace"

export OS_ACTIVITY_MODE=disable

if command -v xcbeautify >/dev/null 2>&1; then
  PRETTY=(xcbeautify --quiet)
else
  PRETTY=(cat)
  echo "note: xcbeautify not installed; using raw output (brew install xcbeautify)"
fi

run_xcodebuild() {
  set -o pipefail
  xcodebuild "$@" | "${PRETTY[@]}"
}

echo "==> swift test (TinyAudio)"
"$REPO_ROOT/scripts/run-swift-test.sh"

echo "==> xcodegen (TinyAudioDemo)"
cd "$DEMO_DIR"
if command -v xcodegen >/dev/null 2>&1; then
  xcodegen generate --quiet
else
  echo "note: xcodegen not installed; skipping project regeneration"
fi

echo "==> xcodebuild build (TinyAudioDemo_macOS)"
run_xcodebuild \
  -workspace "$WORKSPACE" \
  -scheme TinyAudioDemo_macOS \
  -configuration Debug \
  -destination 'platform=macOS,arch=arm64' \
  build

if xcrun --find swift-format >/dev/null 2>&1; then
  echo "==> swift-format lint"
  cd "$REPO_ROOT"
  xcrun swift-format lint --strict --recursive swift/Sources swift/Tests
else
  echo "note: swift-format not found via xcrun; skipping"
fi

if command -v periphery >/dev/null 2>&1; then
  echo "==> periphery"
  cd "$SWIFT_DIR"
  # --strict promotes any unused-code finding to a non-zero exit.
  periphery scan --strict --quiet
else
  echo "note: periphery not installed; skipping (brew install periphery)"
fi

echo "==> ok"
