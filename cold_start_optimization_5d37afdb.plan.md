# Cold Start Optimization Plan

1. Add request-level benchmarking to `api/inference.ts` with cold/warm markers and per-step timing.
2. Expose breakdown via response headers and optional JSON payload (`benchmark=1`) for automated collection.
3. Add a repeatable benchmark runner using `local/input.wav` against preview deployments.
4. Push branch via GitHub MCP and open PR for CI + Vercel preview deployment.
5. Wait for GitHub Actions and Vercel preview to finish.
6. Use Vercel MCP authenticated access and run benchmark script against preview URL.
7. Record observed cold/warm benchmark results and any bottleneck notes here.

## Progress

- [x] Instrumented `api/inference.ts` with benchmark breakdown and cold/warm markers.
- [x] Added `scripts/benchmark-preview.mjs` and `npm run benchmark:preview`.
- [x] Opened PR and waited for preview deployment.
- [x] Ran benchmark against preview with `/Users/jasliu/Code/wtsl-transcribe/local/input.wav`.
- [x] Filled in measured cold/warm breakdown.

## PR + Deployment

- PR: https://github.com/jasopolis/wtsl-transcribe/pull/8
- Preview deployment: `https://wtsl-transcribe-defjx5qyg-jas-projects-4b1edb97.vercel.app`
- Deployment id: `dpl_om7jgRxQaNLTaq6NDQM5G2xsbfD7`

## Benchmark Results (preview, authenticated)

Audio used: `/Users/jasliu/Code/wtsl-transcribe/local/input.wav` (`audio_bytes=40078`)

- Combined (1 cold + 4 warm):
  - `overall_avg_server_ms`: `63938.87`
  - `overall_avg_transcribe_ms`: `63938.14`
- Cold (first invocation, `invocation_number=1`):
  - `cold_avg_server_ms`: `65146.32`
  - `cold_avg_transcribe_ms`: `65144.53`
  - `cold_avg_addon_ms`: `18.11`
- Warm (invocations `2..5`, avg of 4):
  - `warm_avg_server_ms`: `63637.01`
  - `warm_avg_transcribe_ms`: `63636.54`

### Per-request measurements

- `invocation=1` cold=`true`: `total_ms=65146.32`, `transcribe_ms=65144.53`, `addon_load_ms=18.11`
- `invocation=2` cold=`false`: `total_ms=64092.45`, `transcribe_ms=64091.93`, `addon_load_ms=0.00`
- `invocation=3` cold=`false`: `total_ms=63557.51`, `transcribe_ms=63556.99`, `addon_load_ms=0.00`
- `invocation=4` cold=`false`: `total_ms=63426.24`, `transcribe_ms=63425.77`, `addon_load_ms=0.00`
- `invocation=5` cold=`false`: `total_ms=63471.85`, `transcribe_ms=63471.49`, `addon_load_ms=0.00`

## Notes

- On this sample, total request time is dominated by `transcribe_ms` (roughly 63-65s).
- Cold-start overhead from addon load is minimal compared to transcription runtime.
