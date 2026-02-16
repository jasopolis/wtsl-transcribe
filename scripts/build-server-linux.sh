#!/usr/bin/env bash
# Build a Linux x64 whisper-server binary for Vercel (or other Linux deploy).
# Runs the build inside Docker so you can run this on macOS/Windows and commit
# the resulting bin/whisper-server to Git LFS. Requires Docker.
#
# Usage: ./scripts/build-server-linux.sh
#   Then: git add bin/whisper-server && git commit -m "Update prebuilt whisper-server (Linux)"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHISPER_CPP="$REPO_ROOT/whisper.cpp"

if [[ ! -d "$WHISPER_CPP" ]]; then
  echo "Initializing whisper.cpp submodule..."
  (cd "$REPO_ROOT" && git submodule update --init)
fi
if [[ ! -d "$WHISPER_CPP" ]]; then
  echo "whisper.cpp not found at $WHISPER_CPP"
  exit 1
fi

BIN_DIR="$REPO_ROOT/bin"
mkdir -p "$BIN_DIR"

echo "Building Linux x64 whisper-server in Docker (linux/amd64 for Vercel)..."
docker run --rm --platform linux/amd64 \
  -v "$REPO_ROOT:/repo:rw" \
  -w /repo/whisper.cpp \
  ubuntu:22.04 \
  bash -c '
    apt-get update -qq && apt-get install -y -qq cmake build-essential > /dev/null
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENBLAS=OFF
    cmake --build build -j
    for c in build/bin/whisper-server build/bin/Release/whisper-server; do
      if [[ -x "$c" ]]; then
        cp "$c" /repo/bin/whisper-server
        chmod +x /repo/bin/whisper-server
        echo "Built: /repo/bin/whisper-server"
        exit 0
      fi
    done
    echo "error: build produced no whisper-server"
    exit 1
  '

echo "Done. Commit the binary with Git LFS:"
echo "  git lfs install   # if not already"
echo "  git add bin/whisper-server"
echo "  git commit -m \"Update prebuilt whisper-server (Linux)\""
echo "  git push"
