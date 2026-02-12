# Transcribe

Transcribe speech to IPA using [neurlang/ipa-whisper-small](https://huggingface.co/neurlang/ipa-whisper-small).

## Local development

```bash
uv sync
uv run fastapi dev
```

POST audio to `/api/transcriptions/` (multipart file). Max duration 10 s; max file size 4.5 MB.
