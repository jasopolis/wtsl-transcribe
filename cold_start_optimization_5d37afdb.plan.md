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
- [ ] Opened PR and waited for preview deployment.
- [ ] Ran benchmark against preview with `local/input.wav`.
- [ ] Filled in measured cold/warm breakdown.
