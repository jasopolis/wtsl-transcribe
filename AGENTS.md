# AGENTS.md

## Cursor Cloud specific instructions

### Overview

**wtsl-transcribe** is a speech-to-IPA transcription service. It wraps a whisper.cpp inference server with a fine-tuned IPA model and exposes a `POST /inference` endpoint.

### Architecture

- **`bin/whisper-server`** — prebuilt native binary (Git LFS). Currently ARM aarch64; an x86_64 build is pending.
- **`models/ggml-ipa-whisper-small-q5_0.bin`** — quantized GGML model (~168 MB, Git LFS).
- **`api/inference.ts`** — Vercel serverless function that spawns the binary and proxies requests.
- **`whisper.cpp/`** — git submodule (only needed if building the binary from source).

### Running the server (local dev)

The server needs two things: the native binary and the model file. Both are tracked via Git LFS.

```bash
# If using the prebuilt binary (after x86_64 version is committed):
./scripts/run-server.sh
# Listens on http://0.0.0.0:8080

# If building from source instead:
git submodule update --init
./scripts/build-server.sh
./scripts/run-server.sh
```

Building from source requires `cmake`, a C++17 toolchain, and `libstdc++-13-dev`.

### Testing inference

```bash
curl -X POST http://127.0.0.1:8080/inference \
  -F "file=@input.wav" \
  -F "temperature=0.0" \
  -F "response_format=json"
```

Use 16 kHz mono WAV for best results. Install `ffmpeg` for automatic format conversion via `--convert`.

### Key caveats

- **Git LFS is required.** Without `git lfs pull`, `models/*.bin` and `bin/whisper-server` are tiny pointer files, not usable binaries.
- **Binary architecture:** The committed `bin/whisper-server` is ARM aarch64. Cloud VMs are x86_64, so the prebuilt binary cannot run directly. Either build from source (`./scripts/build-server.sh`) or wait for an x86_64 binary to be committed.
- **Node.js version:** `package.json` requires `^24.13.0`. Use `nvm install 24` to get the right version.
- **No tsconfig.json** in the repo. TypeScript is used only for the Vercel function; type-check with: `npx -p typescript tsc --noEmit --esModuleInterop --module nodenext --moduleResolution nodenext --target esnext api/inference.ts`
- **No linter or test framework** is configured in this repo.
