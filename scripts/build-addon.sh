#!/usr/bin/env bash
# Build the whisper.cpp Node.js addon (examples/addon.node) and copy the
# resulting binary + shared libraries into bin/ for deployment.
#
# Requires: cmake, C++17 toolchain, libstdc++-13-dev (or equivalent).
#
# Usage:
#   ./scripts/build-addon.sh            # build from submodule
#   ./scripts/build-addon.sh --for-deploy  # use prebuilt if present; fail otherwise

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHISPER_CPP="${WHISPER_CPP:-$REPO_ROOT/whisper.cpp}"
FOR_DEPLOY=false
[[ "${1:-}" == "--for-deploy" ]] && FOR_DEPLOY=true

BIN_DIR="$REPO_ROOT/bin"
LIB_DIR="$BIN_DIR/lib"
ADDON_FILE="$BIN_DIR/whisper-addon.node"

# Deploy mode: use prebuilt addon from Git LFS only; skip cmake.
if $FOR_DEPLOY; then
  if [[ -f "$ADDON_FILE" ]] && [[ -d "$LIB_DIR" ]]; then
    echo "Using prebuilt whisper-addon.node (from Git LFS or previous build)."
    exit 0
  fi
  echo "error: bin/whisper-addon.node not found. Deploy requires a prebuilt addon."
  echo "  Build locally:  ./scripts/build-addon.sh"
  echo "  Then commit:    git add bin/ && git commit && git push"
  exit 1
fi

# Ensure submodule is present
if [[ ! -d "$WHISPER_CPP/examples/addon.node" ]]; then
  echo "Initializing whisper.cpp submodule..."
  (cd "$REPO_ROOT" && git submodule update --init)
fi
if [[ ! -d "$WHISPER_CPP/examples/addon.node" ]]; then
  echo "whisper.cpp not found at $WHISPER_CPP"
  echo "Initialize the submodule: git submodule update --init"
  exit 1
fi

# Install addon build deps (node-addon-api needed for napi.h)
echo "Installing addon.node build dependencies..."
(cd "$WHISPER_CPP/examples/addon.node" && npm install --ignore-scripts)

# Build with cmake-js (uses gcc to avoid clang/libstdc++ link issues)
echo "Building whisper.cpp addon.node..."
(
  cd "$WHISPER_CPP"
  rm -rf build
  CC="${CC:-gcc}" CXX="${CXX:-g++}" \
    npx cmake-js compile -T addon.node -B Release
)

# Copy addon + shared libs to bin/
mkdir -p "$LIB_DIR"
cp "$WHISPER_CPP/build/Release/addon.node.node" "$ADDON_FILE"

for lib in "$WHISPER_CPP/build/Release/"*.so*; do
  [[ -f "$lib" || -L "$lib" ]] && cp -P "$lib" "$LIB_DIR/"
done

echo "Built: $ADDON_FILE"
echo "Libs:  $LIB_DIR/"
ls -lh "$ADDON_FILE" "$LIB_DIR/"
echo ""
echo "Done. To commit for deploy:"
echo "  git add bin/"
echo "  git commit -m 'Update prebuilt whisper addon (Linux x64)'"
echo "  git push"
