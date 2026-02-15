#!/usr/bin/env bash
# Convert neurlang/ipa-whisper-small (HuggingFace) to ggml, then optionally quantize.
#
# Prerequisites:
#   - Python with: pip install torch transformers numpy huggingface_hub
#   - For quantization: build whisper.cpp (cmake -B build && cmake --build build -j --config Release)
#
# Usage:
#   ./scripts/run-convert-ipa-to-ggml.sh [--quantize]
#
# Output:
#   - models/ggml-ipa-whisper-small.bin (F16, ~466 MB)
#   - models/ggml-ipa-whisper-small-q5_0.bin (Q5_0, ~182 MB) if --quantize

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKDIR="${WORKDIR:-$REPO_ROOT/.convert-workdir}"
MODEL_HF="${MODEL_HF:-neurlang/ipa-whisper-small}"
MEL_URL="https://github.com/openai/whisper/raw/main/whisper/assets/mel_filters.npz"
CONVERT_PY="$SCRIPT_DIR/convert-h5-to-ggml.py"
OUT_MODEL_F16="$REPO_ROOT/models/ggml-ipa-whisper-small.bin"
OUT_MODEL_Q5="$REPO_ROOT/models/ggml-ipa-whisper-small-q5_0.bin"
QUANTIZE=false

for arg in "$@"; do
  case "$arg" in
    --quantize) QUANTIZE=true ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

mkdir -p "$WORKDIR"
mkdir -p "$REPO_ROOT/models"

# Directory layout for convert-h5-to-ggml: dir_whisper must contain whisper/assets/mel_filters.npz
WHISPER_ASSETS="$WORKDIR/whisper/whisper/assets"
mkdir -p "$WHISPER_ASSETS"
if [[ ! -f "$WHISPER_ASSETS/mel_filters.npz" ]]; then
  echo "Downloading mel_filters.npz from openai/whisper ..."
  curl -sSL -o "$WHISPER_ASSETS/mel_filters.npz" "$MEL_URL"
fi

# Download HuggingFace model
HF_DIR="$WORKDIR/ipa-whisper-small"
if [[ ! -d "$HF_DIR" ]] || [[ ! -f "$HF_DIR/config.json" ]]; then
  echo "Downloading $MODEL_HF from HuggingFace ..."
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_HF', local_dir='$HF_DIR', local_dir_use_symlinks=False)
"
fi

# Run conversion
OUT_GGML_DIR="$WORKDIR/ggml-out"
mkdir -p "$OUT_GGML_DIR"
echo "Running convert-h5-to-ggml.py ..."
python3 "$CONVERT_PY" "$HF_DIR" "$WORKDIR/whisper" "$OUT_GGML_DIR"

mv "$OUT_GGML_DIR/ggml-model.bin" "$OUT_MODEL_F16"
echo "Wrote $OUT_MODEL_F16"

if [[ "$QUANTIZE" == true ]]; then
  WHISPER_CPP_BUILD="${WHISPER_CPP_BUILD:-}"
  if [[ -z "$WHISPER_CPP_BUILD" ]]; then
    echo "To quantize, clone whisper.cpp, build it, and set WHISPER_CPP_BUILD to the build directory:"
    echo ""
    echo "  git clone https://github.com/ggml-org/whisper.cpp.git"
    echo "  cd whisper.cpp"
    echo "  cmake -B build"
    echo "  cmake --build build -j --config Release"
    echo ""
    echo "  export WHISPER_CPP_BUILD=\"\$(pwd)/build\""
    echo "  $0 --quantize"
    exit 1
  fi
  # Try common locations for the quantize binary (whisper-quantize; cmake bin, bin/Release, or build root)
  QUANT_BIN=""
  for candidate in \
    "$WHISPER_CPP_BUILD/bin/whisper-quantize" \
    "$WHISPER_CPP_BUILD/bin/Release/whisper-quantize" \
    "$WHISPER_CPP_BUILD/whisper-quantize"; do
    if [[ -x "$candidate" ]]; then
      QUANT_BIN="$candidate"
      break
    fi
  done
  if [[ -z "$QUANT_BIN" ]]; then
    echo "Quantize binary not found. WHISPER_CPP_BUILD=$WHISPER_CPP_BUILD"
    echo ""
    echo "Build whisper.cpp first:"
    echo "  git clone https://github.com/ggml-org/whisper.cpp.git && cd whisper.cpp"
    echo "  cmake -B build && cmake --build build -j --config Release"
    echo ""
    echo "Then set the build directory (use the real path; must contain bin/whisper-quantize):"
    echo "  export WHISPER_CPP_BUILD=\"\$(pwd)/build\"   # from inside whisper.cpp"
    echo "  $0 --quantize"
    exit 1
  fi
  echo "Quantizing to Q5_0 ..."
  "$QUANT_BIN" "$OUT_MODEL_F16" "$OUT_MODEL_Q5" q5_0
  echo "Wrote $OUT_MODEL_Q5"
fi

echo "Done."
