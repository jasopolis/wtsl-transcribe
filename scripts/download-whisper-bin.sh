#!/usr/bin/env bash
# Download CPU-only prebuilt whisper-server from whisper.cpp releases (Linux x64).
# Used at Vercel build time to stay under 250MB (binary ~4MB).
# See: https://github.com/ggml-org/whisper.cpp/releases
#
# Usage: ./scripts/download-whisper-bin.sh [version]

set -euo pipefail
VERSION="${1:-v1.8.3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="${REPO_ROOT}/bin"
URL="https://github.com/ggml-org/whisper.cpp/releases/download/${VERSION}/whisper-bin-x64.zip"
ZIP="${BIN_DIR}/whisper-bin-x64.zip"

mkdir -p "$BIN_DIR"
echo "Downloading whisper CPU-only prebuilt ${VERSION}..."
curl -sL -o "$ZIP" "$URL"
echo "Extracting..."
unzip -o -q "$ZIP" -d "$BIN_DIR"
rm -f "$ZIP"

# Find whisper-server (may be in root or a subdir of the zip)
# Note: official whisper-bin-x64.zip may be Windows; on Linux Vercel build we may need to build from source.
SERVER=$(find "$BIN_DIR" -maxdepth 3 -type f -name "whisper-server" ! -name "*.exe" 2>/dev/null | head -1)
if [[ -z "$SERVER" ]]; then
  SERVER=$(find "$BIN_DIR" -maxdepth 3 -type f -executable -name "whisper-server" 2>/dev/null | head -1)
fi
if [[ -z "$SERVER" ]]; then
  echo "No Linux whisper-server in zip (official zip may be Windows). Building from whisper.cpp submodule..."
  WHISPER_CPP="${REPO_ROOT}/whisper.cpp"
  if [[ ! -d "$WHISPER_CPP" ]]; then
    echo "error: whisper.cpp submodule not found. Run: git submodule update --init"
    exit 1
  fi
  (cd "$WHISPER_CPP" && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENBLAS=OFF && cmake --build build -j --target whisper-server)
  for candidate in "$WHISPER_CPP/build/bin/whisper-server" "$WHISPER_CPP/build/bin/Release/whisper-server"; do
    if [[ -x "$candidate" ]]; then
      cp "$candidate" "$BIN_DIR/whisper-server"
      chmod +x "$BIN_DIR/whisper-server"
      echo "Ready (from source): $BIN_DIR/whisper-server"
      ls -la "$BIN_DIR/whisper-server"
      exit 0
    fi
  done
  echo "error: build produced no whisper-server"
  exit 1
fi

# Normalize: place at bin/whisper-server so api can use BIN_DIR/whisper-server
if [[ "$(dirname "$SERVER")" != "$BIN_DIR" ]]; then
  mv "$SERVER" "$BIN_DIR/whisper-server"
  # Remove any other extracted files to keep deployment small
  find "$BIN_DIR" -mindepth 1 -maxdepth 1 ! -name "whisper-server" -exec rm -rf {} +
fi
chmod +x "$BIN_DIR/whisper-server"
echo "Ready: $BIN_DIR/whisper-server"
ls -la "$BIN_DIR/whisper-server"
