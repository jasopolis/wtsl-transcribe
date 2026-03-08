# AGENTS.md

## Cursor Cloud specific instructions

### Overview

**wtsl-transcribe** is a speech-to-IPA transcription service. It uses whisper.cpp's
native Node.js addon with a fine-tuned IPA model and exposes a `POST /inference`
Vercel serverless endpoint.  Inference runs in-process — no child-process server
or TCP proxy is needed.

### Architecture

- **`bin/whisper-addon.node`** — prebuilt N-API addon compiled from the
  `whisper.cpp/examples/addon.node` reference implementation (Git LFS, Linux x64).
- **`bin/lib/`** — shared libraries (`libwhisper.so`, `libggml*.so`) required
  by the addon (Git LFS).
- **`models/ggml-ipa-whisper-small-q5_0.bin`** — quantized GGML model (~168 MB, Git LFS).
- **`api/inference.ts`** — Vercel serverless function that loads the addon and
  runs whisper.cpp inference in-process.
- **`whisper.cpp/`** — git submodule (only needed when building the addon from source).

### Building the addon

The addon is pre-built for Linux x64 and committed via Git LFS.  To rebuild:

```bash
# Requires: cmake, g++, libstdc++-13-dev, node-addon-api (installed automatically)
git submodule update --init
./scripts/build-addon.sh
# Output: bin/whisper-addon.node + bin/lib/*.so
```

### Running locally (dev)

```bash
# Pull LFS files (model + addon binary):
git lfs pull

# Quick test with Node directly:
LD_LIBRARY_PATH=bin/lib node -e "
  const { whisper } = require('./bin/whisper-addon.node');
  const { promisify } = require('util');
  promisify(whisper)({
    model: 'models/ggml-ipa-whisper-small-q5_0.bin',
    fname_inp: 'input.wav',
    language: 'en',
    use_gpu: false,
    no_prints: true,
  }).then(r => console.log(JSON.stringify(r, null, 2)));
"
```

### Testing inference (via Vercel dev or deployed)

```bash
curl -X POST http://127.0.0.1:3000/api/inference \
  -F "file=@input.wav" \
  -F "temperature=0.0" \
  -F "response_format=json"
```

Use 16 kHz mono WAV for best results.

### Key caveats

- **Git LFS is required.** Without `git lfs pull`, model and addon files are
  tiny pointer files.
- **Binary architecture:** The committed addon is built for Linux x64.  To
  rebuild for a different platform, run `./scripts/build-addon.sh`.
- **Node.js version:** `>=20.0.0`.  The addon uses N-API (ABI-stable across
  Node versions).
- **No tsconfig.json** in the repo. TypeScript is used only for the Vercel
  function; type-check with:
  `npx -p typescript tsc --noEmit --esModuleInterop --module nodenext --moduleResolution nodenext --target esnext api/inference.ts`
- **No linter or test framework** is configured in this repo.
