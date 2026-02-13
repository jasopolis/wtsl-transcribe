import asyncio
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, HTTPException, Request
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MAX_FILE_SIZE = int(4.5 * 1024 * 1024) # 4.5 MB (Vercel functions limit)
MAX_AUDIO_LENGTH = 10

MODEL_ID = "neurlang/ipa-whisper-small"
# Absolute path so cache is reused rnpm install -g vercelegardless of process CWD (e.g. uvicorn from project root)
CACHE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "cache"))
# Formats supported by soundfile (libsndfile). Not supported: AAC, WebM.
ALLOWED_MIME_TYPES = [
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/x-flac",
    "audio/ogg",
    "audio/x-aiff",
    "audio/aiff",
]

TARGET_SR = 16000
# Prepend silence so Whisper doesn't drop the beginning of the speech (known issue)
PREPEND_SILENCE_SEC = 1.5

app = FastAPI()

processor = None
model = None
_load_lock = asyncio.Lock()

@app.middleware("http")
async def validate_upload_size(request: Request, call_next):
    if request.url.path == "/api/transcriptions/" and request.method == "POST":
        content_length = request.headers.get("content-length")
        if not content_length:
            raise HTTPException(status_code=411, detail="Content-Length header required.")
        file_size = int(content_length)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Limit is {MAX_FILE_SIZE / 1024 / 1024} MB.",
            )
    response = await call_next(request)
    return response


def _load_model():
    global processor, model
    processor = WhisperProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, torch_dtype=torch.float16, local_files_only=True
    )


async def _get_model():
    global processor, model
    async with _load_lock:
        if model is None:
            await asyncio.to_thread(_load_model)
    return processor, model


def _load_audio_16k(path: str) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at 16 kHz (Whisper input)."""
    data, sampling_rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sampling_rate != TARGET_SR:
        # resample_poly for better quality when downsampling
        num_samples = int(round(len(data) * TARGET_SR / sampling_rate))
        data = resample_poly(data, TARGET_SR, sampling_rate).astype(np.float32)
        if len(data) > num_samples:
            data = data[:num_samples]
        elif len(data) < num_samples:
            data = np.pad(data, (0, num_samples - len(data)), mode="constant")
        sampling_rate = TARGET_SR
    return data, sampling_rate


def _transcribe(path: str) -> str:
    audio_array, sampling_rate = _load_audio_16k(path)
    duration_sec = len(audio_array) / sampling_rate
    if duration_sec > MAX_AUDIO_LENGTH:
        raise ValueError(
            f"Audio too long: {duration_sec:.2f}s. Maximum is {MAX_AUDIO_LENGTH}s."
        )
    # Prepend silence so model doesn't miss the first ~0.5s (Whisper quirk)
    prepend_samples = int(PREPEND_SILENCE_SEC * sampling_rate)
    audio_array = np.concatenate([np.zeros(prepend_samples, dtype=audio_array.dtype), audio_array])
    processed = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_attention_mask=True,
        return_tensors="pt",
    )
    # Match model dtype (float16)
    input_features = processed.input_features.to(dtype=model.dtype, device=model.device)
    attention_mask = processed.attention_mask.to(device=model.device)
    predicted_ids = model.generate(
        input_features,
        attention_mask=attention_mask,
    )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return f"/{''.join(transcription)}/"


@app.post("/api/transcriptions/")
async def transcribe_recording(file: UploadFile):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.close(fd)
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)
        await _get_model()
        try:
            result = await asyncio.to_thread(_transcribe, path)
        except ValueError as e:
            raise HTTPException(status_code=413, detail=str(e))
        return {"ipa_transcription": result}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
