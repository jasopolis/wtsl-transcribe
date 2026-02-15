#!/usr/bin/env bash
# Build the whisper.cpp inference server (examples/server).
# Requires: cmake, C++17 toolchain. Run from repo root.
#
# Usage: ./scripts/build-server.sh [--for-deploy]
#   --for-deploy  Use prebuilt bin/whisper-server if present (e.g. from Git LFS);
#                 otherwise init submodule, build, copy to bin/, then remove source.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHISPER_CPP="${WHISPER_CPP:-$REPO_ROOT/whisper.cpp}"
FOR_DEPLOY=false
[[ "${1:-}" == "--for-deploy" ]] && FOR_DEPLOY=true

# Deploy (e.g. Vercel): use prebuilt binary from Git LFS only; no cmake on deploy.
if $FOR_DEPLOY; then
  BIN_DIR="$REPO_ROOT/bin"
  PREBUILT="$BIN_DIR/whisper-server"
  if [[ -x "$PREBUILT" ]]; then
    echo "Using prebuilt bin/whisper-server (from Git LFS or previous build)."
  else
    echo "error: bin/whisper-server not found. Deploy requires a prebuilt binary in Git LFS."
    echo "  Local: run ./scripts/build-server-linux.sh then git add bin/whisper-server && git commit && git push"
    exit 1
  fi
  # Vercel expects an output directory (default "public"); create it so the build succeeds.
  mkdir -p "$REPO_ROOT/public"
  echo "Created public/ for Vercel output directory."
  exit 0
fi

if [[ ! -d "$WHISPER_CPP" ]]; then
  if $FOR_DEPLOY; then
    echo "Initializing whisper.cpp submodule..."
    (cd "$REPO_ROOT" && git submodule update --init)
  fi
  if [[ ! -d "$WHISPER_CPP" ]]; then
    echo "whisper.cpp not found at $WHISPER_CPP"
    echo "Initialize the submodule: git submodule update --init"
    exit 1
  fi
fi

cd "$WHISPER_CPP"
cmake -B build -DCMAKE_BUILD_TYPE=Release ${FOR_DEPLOY:+-DGGML_OPENBLAS=OFF}
cmake --build build -j --config Release

if $FOR_DEPLOY; then
  BIN_DIR="$REPO_ROOT/bin"
  mkdir -p "$BIN_DIR"
  for candidate in build/bin/whisper-server build/bin/Release/whisper-server; do
    if [[ -x "$candidate" ]]; then
      cp "$candidate" "$BIN_DIR/whisper-server"
      chmod +x "$BIN_DIR/whisper-server"
      echo "Built: $BIN_DIR/whisper-server"
      break
    fi
  done
  if [[ ! -x "$BIN_DIR/whisper-server" ]]; then
    echo "error: build produced no whisper-server"
    exit 1
  fi
  echo "Removing whisper.cpp source to reduce bundle size..."
  cd "$REPO_ROOT"
  rm -rf "$WHISPER_CPP"
  echo "Done."
  exit 0
fi

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
