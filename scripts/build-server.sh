#!/usr/bin/env bash
# Build the whisper.cpp inference server (examples/server).
# Requires: cmake, C++17 toolchain. Run from repo root.
#
# Usage: ./scripts/build-server.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHISPER_CPP="${WHISPER_CPP:-$REPO_ROOT/whisper.cpp}"

if [[ ! -d "$WHISPER_CPP" ]]; then
  echo "whisper.cpp not found at $WHISPER_CPP"
  echo "Initialize the submodule: git submodule update --init"
  exit 1
fi

cd "$WHISPER_CPP"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --config Release

echo "Server binary: $WHISPER_CPP/build/bin/whisper-server"
if [[ "$(uname -s)" == "Darwin" ]]; then
  # macOS may put Release binary in build/bin/Release/
  if [[ -x build/bin/whisper-server ]]; then
    echo "Ready. Run: ./scripts/run-server.sh"
  elif [[ -x build/bin/Release/whisper-server ]]; then
    echo "Ready (Release). Run: ./scripts/run-server.sh"
  fi
else
  echo "Ready. Run: ./scripts/run-server.sh"
fi
