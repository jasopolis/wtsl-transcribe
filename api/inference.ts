/**
 * Vercel serverless function: transcribes audio via whisper.cpp Node addon.
 *
 * Loads the native addon (bin/whisper-addon.node) and runs inference in-process
 * — no child-process server, no TCP proxy.  The GGML model is loaded once per
 * cold-start and reused across warm invocations.
 *
 * Bundle budget:  addon (~800 KB) + libs (~2.6 MB) + model (~168 MB) ≈ 172 MB
 * well within Vercel's 250 MB compressed limit.
 */

import {
  existsSync, writeFileSync, unlinkSync, mkdirSync,
  statSync, readdirSync, lstatSync, symlinkSync
} from "fs";
import { join } from "path";
import { promisify } from "util";
import { createRequire } from "module";
import type { VercelRequest, VercelResponse } from "@vercel/node";

const BIN_DIR = process.env.BIN_DIR || join(process.cwd(), "bin");
const LIB_DIR = process.env.LIB_DIR || join(BIN_DIR, "lib");
const MODEL_PATH =
  process.env.MODEL_PATH ||
  join(process.cwd(), "models", "ggml-ipa-whisper-small-q5_0.bin");
const TMP_DIR = "/tmp/whisper-inference";
const TMP_LIB_DIR = "/tmp/whisper-libs";

/**
 * Vercel deployments don't preserve filesystem symlinks. The dynamic linker
 * expects soname aliases (e.g. libwhisper.so.1 -> libwhisper.so.1.8.3).
 * Recreate those aliases in a writable /tmp directory.
 */
const SONAME_MAP: Record<string, string> = {
  "libwhisper.so.1":   "libwhisper.so.1.8.3",
  "libwhisper.so":     "libwhisper.so.1.8.3",
  "libggml.so.0":      "libggml.so.0.9.6",
  "libggml.so":        "libggml.so.0.9.6",
  "libggml-base.so.0": "libggml-base.so.0.9.6",
  "libggml-base.so":   "libggml-base.so.0.9.6",
  "libggml-cpu.so.0":  "libggml-cpu.so.0.9.6",
  "libggml-cpu.so":    "libggml-cpu.so.0.9.6",
};

function ensureLibDir(): string {
  if (existsSync(TMP_LIB_DIR) && readdirSync(TMP_LIB_DIR).length > 0) {
    return TMP_LIB_DIR;
  }
  mkdirSync(TMP_LIB_DIR, { recursive: true });
  const entries = readdirSync(LIB_DIR);
  for (const entry of entries) {
    const fullPath = join(LIB_DIR, entry);
    const stat = lstatSync(fullPath);
    if (stat.isFile() && stat.size > 1000) {
      const dest = join(TMP_LIB_DIR, entry);
      if (!existsSync(dest)) symlinkSync(fullPath, dest);
    }
  }
  for (const [alias, target] of Object.entries(SONAME_MAP)) {
    const dest = join(TMP_LIB_DIR, alias);
    const realFile = join(LIB_DIR, target);
    if (!existsSync(dest) && existsSync(realFile)) {
      symlinkSync(realFile, dest);
    }
  }
  return TMP_LIB_DIR;
}

function loadAddon(): (params: Record<string, unknown>, cb: (err: Error | null, result?: unknown) => void) => void {
  if (!existsSync(LIB_DIR)) {
    throw new Error(`Shared libraries not found at ${LIB_DIR}. Run: npm run build`);
  }
  const libDir = ensureLibDir();
  const sep = ":";
  const current = process.env.LD_LIBRARY_PATH || "";
  const dirs = current ? current.split(sep) : [];
  if (!dirs.includes(libDir)) dirs.unshift(libDir);
  if (!dirs.includes(LIB_DIR)) dirs.unshift(LIB_DIR);
  process.env.LD_LIBRARY_PATH = dirs.join(sep);

  const addonPath = join(BIN_DIR, "whisper-addon.node");
  if (!existsSync(addonPath)) {
    throw new Error(`whisper-addon.node not found at ${addonPath}. Run: npm run build`);
  }
  const require_ = createRequire(__filename);
  const { whisper } = require_(addonPath);
  return whisper;
}

let whisperFn: ReturnType<typeof loadAddon> | null = null;
let lastAddonLoadMs = 0;
let invocationCount = 0;

function nowMs(): number {
  return Number(process.hrtime.bigint()) / 1e6;
}

function getWhisper() {
  if (!whisperFn) {
    const start = nowMs();
    whisperFn = loadAddon();
    lastAddonLoadMs = nowMs() - start;
  }
  return whisperFn;
}

interface WhisperResult {
  transcription: [string, string, string][];
  language?: string;
}

async function transcribe(audioPath: string, opts: { language?: string } = {}) {
  const whisper = getWhisper();
  const whisperAsync = promisify(whisper);
  if (!existsSync(MODEL_PATH)) throw new Error(`Model not found at ${MODEL_PATH}`);

  const result = (await whisperAsync({
    model: MODEL_PATH,
    fname_inp: audioPath,
    language: opts.language || "en",
    use_gpu: false,
    no_prints: true,
    no_timestamps: false,
    comma_in_time: false,
  })) as WhisperResult;

  const segments = (result.transcription || []).map(
    ([start, end, text]: [string, string, string]) => ({ start, end, text: text.trim() })
  );
  return {
    segments,
    text: segments.map((s: { text: string }) => s.text).join(" "),
    ...(result.language ? { language: result.language } : {}),
  };
}

function collectBody(req: VercelRequest): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function parseMultipartBuffer(buf: Buffer, boundary: string): { file?: Buffer; fields: Record<string, string> } {
  const sep = Buffer.from(`--${boundary}`);
  const fields: Record<string, string> = {};
  let fileData: Buffer | undefined;

  let pos = 0;
  while (pos < buf.length) {
    const start = buf.indexOf(sep, pos);
    if (start === -1) break;
    const partStart = start + sep.length;
    if (buf[partStart] === 0x2d && buf[partStart + 1] === 0x2d) break;
    const headerEnd = buf.indexOf(Buffer.from("\r\n\r\n"), partStart);
    if (headerEnd === -1) break;
    const headers = buf.subarray(partStart + 2, headerEnd).toString("utf8");
    const bodyStart = headerEnd + 4;
    const nextSep = buf.indexOf(sep, bodyStart);
    const bodyEnd = nextSep === -1 ? buf.length : nextSep - 2;
    const body = buf.subarray(bodyStart, bodyEnd);

    const nameMatch = headers.match(/name="([^"]+)"/);
    const filenameMatch = headers.match(/filename="([^"]+)"/);
    const name = nameMatch ? nameMatch[1] : "";

    if (filenameMatch) {
      fileData = body;
    } else {
      fields[name] = body.toString("utf8");
    }
    pos = nextSep === -1 ? buf.length : nextSep;
  }
  return { file: fileData, fields };
}

type BenchmarkBreakdown = {
  cold_start: boolean;
  invocation_number: number;
  addon_loaded_this_request: boolean;
  addon_load_ms: number;
  body_collect_ms: number;
  multipart_parse_ms: number;
  temp_write_ms: number;
  transcribe_ms: number;
  cleanup_ms: number;
  total_ms: number;
  audio_bytes: number;
};

function setBenchmarkHeaders(res: VercelResponse, benchmark: BenchmarkBreakdown): void {
  res.setHeader("X-Cold-Start", String(benchmark.cold_start));
  res.setHeader("X-Invocation-Number", String(benchmark.invocation_number));
  res.setHeader("X-Addon-Loaded-This-Request", String(benchmark.addon_loaded_this_request));
  res.setHeader("X-Benchmark-Addon-Load-Ms", benchmark.addon_load_ms.toFixed(2));
  res.setHeader("X-Benchmark-Body-Collect-Ms", benchmark.body_collect_ms.toFixed(2));
  res.setHeader("X-Benchmark-Multipart-Parse-Ms", benchmark.multipart_parse_ms.toFixed(2));
  res.setHeader("X-Benchmark-Temp-Write-Ms", benchmark.temp_write_ms.toFixed(2));
  res.setHeader("X-Benchmark-Transcribe-Ms", benchmark.transcribe_ms.toFixed(2));
  res.setHeader("X-Benchmark-Cleanup-Ms", benchmark.cleanup_ms.toFixed(2));
  res.setHeader("X-Benchmark-Total-Ms", benchmark.total_ms.toFixed(2));
  res.setHeader("X-Benchmark-Audio-Bytes", String(benchmark.audio_bytes));
}

export default async function handler(req: VercelRequest, res: VercelResponse): Promise<void> {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed. Use POST with multipart form (file=audio)." });
    return;
  }

  let tmpPath: string | undefined;
  let cleanupMs = 0;
  let bodyCollectMs = 0;
  let multipartParseMs = 0;
  let tempWriteMs = 0;
  let transcribeMs = 0;
  let audioBytes = 0;
  const requestStart = nowMs();
  const isColdStart = invocationCount === 0;
  invocationCount += 1;
  try {
    const bodyCollectStart = nowMs();
    let rawBody: Buffer;
    if (Buffer.isBuffer(req.body)) {
      rawBody = req.body;
    } else if (typeof req.body === "string") {
      rawBody = Buffer.from(req.body, "binary");
    } else {
      const raw = (req as unknown as { rawBody?: Buffer }).rawBody;
      rawBody = Buffer.isBuffer(raw) ? raw : await collectBody(req);
    }
    bodyCollectMs = nowMs() - bodyCollectStart;

    const multipartParseStart = nowMs();
    const contentType = req.headers["content-type"] || "";
    const boundaryMatch = contentType.match(/boundary=(.+)/);
    if (!boundaryMatch) {
      res.status(400).json({ error: "Missing multipart boundary in Content-Type" });
      return;
    }
    const { file, fields } = parseMultipartBuffer(rawBody, boundaryMatch[1]);
    multipartParseMs = nowMs() - multipartParseStart;
    if (!file || file.length === 0) {
      res.status(400).json({ error: 'Missing "file" field. Send audio as: -F "file=@audio.wav"' });
      return;
    }
    audioBytes = file.length;

    const tempWriteStart = nowMs();
    if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });
    tmpPath = join(TMP_DIR, `${Date.now()}-${Math.random().toString(36).slice(2)}.wav`);
    writeFileSync(tmpPath, file);
    tempWriteMs = nowMs() - tempWriteStart;

    const language = fields.language || "en";
    const responseFormat = fields.response_format || "json";
    const includeBenchmark =
      fields.benchmark === "1" ||
      fields.benchmark === "true" ||
      fields.include_benchmark === "1" ||
      fields.include_benchmark === "true";
    const addonLoadedThisRequest = !whisperFn;

    const transcribeStart = nowMs();
    const result = await transcribe(tmpPath, { language });
    transcribeMs = nowMs() - transcribeStart;

    const cleanupStart = nowMs();
    if (tmpPath) {
      try {
        unlinkSync(tmpPath);
        tmpPath = undefined;
      } catch {
        // Best effort cleanup.
      }
    }
    cleanupMs = nowMs() - cleanupStart;

    const benchmark: BenchmarkBreakdown = {
      cold_start: isColdStart,
      invocation_number: invocationCount,
      addon_loaded_this_request: addonLoadedThisRequest,
      addon_load_ms: addonLoadedThisRequest ? lastAddonLoadMs : 0,
      body_collect_ms: bodyCollectMs,
      multipart_parse_ms: multipartParseMs,
      temp_write_ms: tempWriteMs,
      transcribe_ms: transcribeMs,
      cleanup_ms: cleanupMs,
      total_ms: nowMs() - requestStart,
      audio_bytes: audioBytes,
    };
    setBenchmarkHeaders(res, benchmark);
    console.log("[benchmark] /api/inference", JSON.stringify(benchmark));

    if (responseFormat === "text") {
      res.setHeader("Content-Type", "text/plain");
      res.status(200).send(result.text);
      return;
    }
    res.status(200).json(includeBenchmark ? { ...result, benchmark } : result);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("Inference error:", message);
    res.status(500).json({ error: message });
  } finally {
    if (tmpPath) { try { unlinkSync(tmpPath); } catch {} }
  }
}

export const config = {
  api: { bodyParser: { sizeLimit: "50mb" } },
};
