# Transcribe

Transcribe speech to IPA using [neurlang/ipa-whisper-small](https://huggingface.co/neurlang/ipa-whisper-small) via the [whisper.cpp](https://github.com/ggml-org/whisper.cpp) inference server.

## Usage

**POST** audio to `/inference` (multipart form field `file`):

```bash
curl -X POST http://127.0.0.1:8080/inference \
  -F "file=@input.wav" \
  -F "temperature=0.0" \
  -F "response_format=json"
```

Use 16 kHz mono WAV for best results. Convert with:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le input.wav
```

Response is JSON; IPA text is in the segments (e.g. `/ˌɪntərˈnæʃənəl/`).

---

## Setup

### 1. Model (one-time)

Convert and quantize the HuggingFace IPA model to ggml. See **scripts/README.md** for details.

```bash
./scripts/run-convert-ipa-to-ggml.sh              # → ggml (~466 MB)
export WHISPER_CPP_BUILD="$(pwd)/whisper.cpp/build"
./scripts/run-convert-ipa-to-ggml.sh --quantize   # → models/ggml-ipa-whisper-small-q5_0.bin (~182 MB)
```

Repo uses **Git LFS** for `models/*.bin` and `bin/whisper-server`. One-time: `git lfs install`. Then `git add models/ggml-ipa-whisper-small-q5_0.bin` and commit.

### 2. Build and run server

```bash
git submodule update --init
./scripts/build-server.sh
./scripts/run-server.sh
```

Server: **http://0.0.0.0:8080**. Install **ffmpeg** for non-WAV uploads (optional).

---

## Deploy (Vercel)

Vercel’s builder has no cmake, so the **whisper-server binary is prebuilt and committed via Git LFS** (same approach as the model).

- **Prebuild binary (one-time or when updating whisper.cpp):**  
  From repo root, run `./scripts/build-server-linux.sh` (requires Docker). That produces a Linux x64 binary in `bin/whisper-server`. Then:
  ```bash
  git lfs install   # if not already
  git add bin/whisper-server
  git commit -m "Update prebuilt whisper-server (Linux)"
  git push
  ```
- **Model**: Tracked with Git LFS; Vercel runs `git lfs pull` in the install step.
- **Build**: `npm run build` uses the prebuilt `bin/whisper-server` from the repo (no cmake on Vercel). `api/inference.ts` proxies to it. Send WAV (16 kHz mono); no ffmpeg in bundle.
- **Deploy**: `npm i -g vercel` then `vercel`. Env: `MODEL_PATH`, `BIN_DIR` (optional).
- **Limits**: 250MB bundle, 1GB memory (in `vercel.json`). Use smaller model if needed.
