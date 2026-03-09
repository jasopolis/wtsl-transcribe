#!/usr/bin/env node

import { readFile } from "fs/promises";

const DEFAULT_RUNS = 4;
const DEFAULT_AUDIO_PATH = "local/input.wav";

function parseArgs(argv) {
  const out = {
    url: process.env.BENCHMARK_URL || "",
    audioPath: process.env.BENCHMARK_AUDIO || DEFAULT_AUDIO_PATH,
    runs: Number(process.env.BENCHMARK_RUNS || DEFAULT_RUNS),
    language: process.env.BENCHMARK_LANGUAGE || "en",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if ((arg === "--url" || arg === "-u") && argv[i + 1]) out.url = argv[++i];
    else if ((arg === "--audio" || arg === "-a") && argv[i + 1]) out.audioPath = argv[++i];
    else if ((arg === "--runs" || arg === "-n") && argv[i + 1]) out.runs = Number(argv[++i]);
    else if ((arg === "--language" || arg === "-l") && argv[i + 1]) out.language = argv[++i];
  }

  if (!out.url) throw new Error("Missing benchmark URL. Set --url or BENCHMARK_URL.");
  if (!Number.isFinite(out.runs) || out.runs < 1) throw new Error("Runs must be a positive integer.");
  return out;
}

function toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function fmtMs(value) {
  return `${value.toFixed(2)}ms`;
}

async function run() {
  const cfg = parseArgs(process.argv.slice(2));
  const audioBuffer = await readFile(cfg.audioPath);
  const endpoint = cfg.url.replace(/\/$/, "");
  const rows = [];

  for (let i = 0; i < cfg.runs; i += 1) {
    const form = new FormData();
    form.append("file", new Blob([audioBuffer], { type: "audio/wav" }), "input.wav");
    form.append("response_format", "json");
    form.append("benchmark", "1");
    form.append("language", cfg.language);

    const reqStart = performance.now();
    const response = await fetch(endpoint, { method: "POST", body: form });
    const reqEnd = performance.now();
    const payload = await response.json().catch(() => ({}));

    const row = {
      run: i + 1,
      status: response.status,
      total_http_ms: reqEnd - reqStart,
      cold_start: (response.headers.get("x-cold-start") || "").toLowerCase() === "true",
      invocation_number: toNumber(response.headers.get("x-invocation-number")),
      addon_load_ms: toNumber(response.headers.get("x-benchmark-addon-load-ms")),
      body_collect_ms: toNumber(response.headers.get("x-benchmark-body-collect-ms")),
      multipart_parse_ms: toNumber(response.headers.get("x-benchmark-multipart-parse-ms")),
      temp_write_ms: toNumber(response.headers.get("x-benchmark-temp-write-ms")),
      transcribe_ms: toNumber(response.headers.get("x-benchmark-transcribe-ms")),
      cleanup_ms: toNumber(response.headers.get("x-benchmark-cleanup-ms")),
      total_server_ms: toNumber(response.headers.get("x-benchmark-total-ms")),
      text_length: typeof payload?.text === "string" ? payload.text.length : 0,
      error: payload?.error || "",
    };
    rows.push(row);
    console.log(
      `[run ${row.run}] status=${row.status} cold=${row.cold_start} server=${fmtMs(row.total_server_ms)} http=${fmtMs(row.total_http_ms)} transcribe=${fmtMs(row.transcribe_ms)}`
    );
  }

  const okRows = rows.filter((r) => r.status >= 200 && r.status < 300);
  if (okRows.length === 0) {
    console.log("\nNo successful responses.");
    console.log(JSON.stringify(rows, null, 2));
    process.exitCode = 1;
    return;
  }

  const avg = (key) => okRows.reduce((sum, r) => sum + r[key], 0) / okRows.length;
  const coldRows = okRows.filter((r) => r.cold_start);
  const warmRows = okRows.filter((r) => !r.cold_start);

  const summary = {
    runs: rows.length,
    successful_runs: okRows.length,
    cold_runs: coldRows.length,
    warm_runs: warmRows.length,
    avg_http_ms: avg("total_http_ms"),
    avg_server_ms: avg("total_server_ms"),
    avg_transcribe_ms: avg("transcribe_ms"),
    avg_addon_load_ms: avg("addon_load_ms"),
    avg_body_collect_ms: avg("body_collect_ms"),
    avg_multipart_parse_ms: avg("multipart_parse_ms"),
    avg_temp_write_ms: avg("temp_write_ms"),
    avg_cleanup_ms: avg("cleanup_ms"),
  };

  console.log("\nSummary:");
  console.log(JSON.stringify(summary, null, 2));
  console.log("\nRaw rows:");
  console.log(JSON.stringify(rows, null, 2));
}

run().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exitCode = 1;
});
