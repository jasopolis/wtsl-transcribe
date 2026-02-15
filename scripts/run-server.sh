#!/usr/bin/env bash
# Run the whisper.cpp inference server with the IPA model.
# Build first: ./scripts/build-server.sh
# Requires: models/ggml-ipa-whisper-small-q5_0.bin (see scripts/README.md)
#
# Usage: ./scripts/run-server.sh [extra options for whisper-server]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHISPER_CPP="${WHISPER_CPP:-$REPO_ROOT/whisper.cpp}"
MODEL="${MODEL:-$REPO_ROOT/models/ggml-ipa-whisper-small-q5_0.bin}"

SERVER_BIN=""
for candidate in \
  "$WHISPER_CPP/build/bin/whisper-server" \
  "$WHISPER_CPP/build/bin/Release/whisper-server"; do
  if [[ -x "$candidate" ]]; then
    SERVER_BIN="$candidate"
    break
  fi
done

if [[ -z "$SERVER_BIN" ]]; then
  echo "whisper-server not found. Build first: ./scripts/build-server.sh"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL"
  echo "Convert and quantize the IPA model: ./scripts/run-convert-ipa-to-ggml.sh --quantize"
  exit 1
fi

exec "$SERVER_BIN" \
  -m "$MODEL" \
  --host "0.0.0.0" \
  --port 8080 \
  -l en \
  --convert \
  "$@"
