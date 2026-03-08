# Model conversion: IPA Whisper (HuggingFace) → ggml

This repo uses [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with a ggml model. The [neurlang/ipa-whisper-small](https://huggingface.co/neurlang/ipa-whisper-small) model is HuggingFace (safetensors), so it must be converted once and (optionally) quantized.

## One-time setup

- **Python**: `torch`, `transformers`, `numpy`, `huggingface_hub` (e.g. `pip install torch transformers numpy huggingface_hub`).
- **Quantization**: use the **whisper.cpp** submodule. Build it once; the same build provides `whisper-quantize` and the inference server.

## Convert and quantize

From the repo root:

```bash
# Initialize and build whisper.cpp (submodule)
git submodule update --init
./scripts/build-server.sh

# Convert only → writes models/ggml-ipa-whisper-small.bin (~466 MB)
./scripts/run-convert-ipa-to-ggml.sh

# Quantize using the submodule build
export WHISPER_CPP_BUILD="$(pwd)/whisper.cpp/build"
./scripts/run-convert-ipa-to-ggml.sh --quantize
```

The script looks for `whisper-quantize` in `$WHISPER_CPP_BUILD/bin/whisper-quantize`, `bin/Release/whisper-quantize`, or the build dir root.

The script downloads mel filters from [openai/whisper](https://github.com/openai/whisper) and the IPA model from HuggingFace into `.convert-workdir/`. Output goes to `models/`.

Use the **Q5_0** quantized model (`ggml-ipa-whisper-small-q5_0.bin`) with the inference server. If you need a smaller file, use a smaller quantization (e.g. Q4_0) or convert `ipa-whisper-base` and quantize that.
