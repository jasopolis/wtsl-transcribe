#!/usr/bin/env bash
# Run the Vercel dev server locally.
# Build first: ./scripts/build-addon.sh
# Requires: models/ggml-ipa-whisper-small-q5_0.bin (see scripts/README.md)
#
# Usage: ./scripts/run-server.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${MODEL:-$REPO_ROOT/models/ggml-ipa-whisper-small-q5_0.bin}"
ADDON="$REPO_ROOT/bin/whisper-addon.node"
LIB_DIR="$REPO_ROOT/bin/lib"

if [[ ! -f "$ADDON" ]]; then
  echo "whisper-addon.node not found. Build first: ./scripts/build-addon.sh"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL"
  echo "Convert and quantize the IPA model: ./scripts/run-convert-ipa-to-ggml.sh --quantize"
  exit 1
fi

export LD_LIBRARY_PATH="${LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "Starting Vercel dev server (LD_LIBRARY_PATH=$LD_LIBRARY_PATH)..."
exec npx vercel dev --listen 0.0.0.0:8080
