#!/bin/bash
# Run `swift test` from swift/ and tolerate the known MLX metallib teardown
# crash that fires after all tests pass.
#
# `swift test` builds the test bundle without an `mlx.metallib`, so MLX's C++
# layer prints "Failed to load the default metallib" during process teardown
# and the process exits non-zero — even when every test passed. We treat that
# specific case as success; anything else propagates.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/swift"

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

swift test 2>&1 | tee "$LOG"
EXIT=${PIPESTATUS[0]}

if [ "$EXIT" -eq 0 ]; then
  exit 0
fi

if grep -q "Failed to load the default metallib" "$LOG" \
   && ! grep -qE "(✘|Test run had| failed after )" "$LOG"; then
  echo "note: ignoring known MLX metallib teardown crash (all tests passed)"
  exit 0
fi

exit "$EXIT"
