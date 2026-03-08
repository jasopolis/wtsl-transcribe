# Cold Start Optimization Evaluation

**Goal:** Achieve &lt;10s inference on a 10-second audio clip for web-app use.

**Current state (preview deployment):** ~67s total for a 30-second clip (cold start + inference).

---

## 1. Time Breakdown

The ~67s total likely splits as:

| Phase | Estimated time | Notes |
|-------|----------------|-------|
| **Cold start** | ~15–25s | Model load (168 MB), addon + libs, symlink setup |
| **Inference** | ~42–52s | whisper small on 2 GB / 1 vCPU for 30s audio |

For a **10-second clip**, inference alone is roughly 1/3 of that (~14–17s) on the same hardware. So even without cold start, **inference alone may exceed 10s** on Vercel’s default 2 GB / 1 vCPU.

---

## 2. Vercel Features to Reduce Cold Start

### 2.1 Fluid Compute (Primary)

Fluid compute keeps at least one warm instance and avoids most cold starts:

- **Zero cold starts** for ~99.37% of requests
- **Scale to one:** one instance stays warm instead of scaling to zero
- **Cost:** provisioned memory while idle (~$0.0106/GB-hour)

**Enable in `vercel.json`:**

```json
{
  "fluid": true
}
```

Or via Project → Settings → Functions → Fluid Compute.

**Recommendation:** Enable Fluid compute first. This removes the ~15–25s cold start for almost all requests.

---

### 2.2 Bytecode Caching

- Default for Node.js 20+ on Vercel
- Reduces cold start by ~27–60% when cold starts do occur
- No config needed; already active for this project

---

### 2.3 Memory / CPU (Pro / Enterprise)

- **Default:** 2 GB / 1 vCPU
- **Performance:** 4 GB / 2 vCPUs (Pro/Enterprise only, via dashboard)

Higher memory/CPU speeds up model load and inference. Configure in Project → Settings → Functions → Advanced.

---

### 2.4 Cron Keep-Warm (Fallback)

If Fluid compute is not available (e.g. Hobby plan):

- Add a cron job that pings `/api/inference` every 5–10 minutes
- Requires a lightweight endpoint (e.g. GET returning 405) to avoid running full inference

---

### 2.5 Edge Functions

- **Not viable:** 128 MB memory limit; model alone is ~168 MB
- Edge is for lightweight, globally distributed logic, not heavy ML

---

## 3. Model Optimization for &lt;10s Inference

To reach &lt;10s inference on a 10-second clip, inference must be faster.

### 3.1 Switch to ipa-whisper-base

- **neurlang/ipa-whisper-base** exists on HuggingFace (74M vs 244M params)
- Base is ~2–3× faster than small
- Quantized base: ~50–60 MB vs ~168 MB (small)
- Smaller model → faster load and inference

**Steps:**

1. Extend `scripts/run-convert-ipa-to-ggml.sh` to support `MODEL_HF=neurlang/ipa-whisper-base`
2. Convert and quantize to `ggml-ipa-whisper-base-q5_0.bin`
3. Set `MODEL_PATH` or default path to the base model

**Trade-off:** Base may be slightly less accurate than small for IPA; worth benchmarking.

---

### 3.2 Quantization

- Current: Q5_0 (~168 MB)
- Q4_0: smaller and faster, with some accuracy loss
- Base + Q5_0 is a good balance of speed and quality

---

## 4. Summary of Recommendations

| Priority | Action | Impact |
|----------|--------|--------|
| **1** | Enable Fluid compute (`"fluid": true` in `vercel.json`) | Removes ~15–25s cold start for most requests |
| **2** | Use 4 GB / 2 vCPUs (Pro/Enterprise, dashboard) | Faster model load and inference |
| **3** | Switch to ipa-whisper-base | ~2–3× faster inference, smaller model |
| **4** | Cron keep-warm (if no Fluid) | Reduces cold starts when idle |

---

## 5. Realistic Expectations

| Scenario | Cold start | Inference (10s clip) | Total |
|----------|------------|----------------------|-------|
| Current (small, cold) | ~20s | ~15s | ~35s |
| Fluid + small (warm) | ~0s | ~15s | ~15s |
| Fluid + base (warm) | ~0s | ~5–8s | **~5–8s** |
| Fluid + base + 4 GB | ~0s | ~4–6s | **~4–6s** |

**Conclusion:** &lt;10s total for a 10-second clip is achievable with:

1. Fluid compute (warm instance)
2. ipa-whisper-base model
3. Optional: 4 GB / 2 vCPUs for extra speed

---

## 6. Quick Wins (No Code Changes)

1. **Enable Fluid compute** in the Vercel dashboard or `vercel.json`
2. **Set 4 GB / 2 vCPUs** in Project → Settings → Functions (Pro/Enterprise)

These alone should cut cold starts and improve inference time.

---

## 7. Next Steps for Model Change

To adopt ipa-whisper-base:

1. Add `--base` (or similar) to `run-convert-ipa-to-ggml.sh` to use `neurlang/ipa-whisper-base`
2. Output `ggml-ipa-whisper-base-q5_0.bin`
3. Add to Git LFS and update `MODEL_PATH` / default path in `api/inference.ts`
4. Benchmark accuracy vs small on your IPA use case
