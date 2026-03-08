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
import { Readable } from "stream";
import Busboy from "busboy";

const BIN_DIR = process.env.BIN_DIR || join(process.cwd(), "bin");
const LIB_DIR = process.env.LIB_DIR || join(BIN_DIR, "lib");
const MODEL_PATH =
  process.env.MODEL_PATH ||
  join(process.cwd(), "models", "ggml-ipa-whisper-small-q5_0.bin");
const TMP_DIR = "/tmp/whisper-inference";
const TMP_LIB_DIR = "/tmp/whisper-libs";

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
  console.log(`[inference] loadAddon: BIN_DIR=${BIN_DIR}, LIB_DIR=${LIB_DIR}`);
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
  console.log(`[inference] LD_LIBRARY_PATH=${process.env.LD_LIBRARY_PATH}`);

  const addonPath = join(BIN_DIR, "whisper-addon.node");
  if (!existsSync(addonPath)) {
    throw new Error(`whisper-addon.node not found at ${addonPath}. Run: npm run build`);
  }

  console.log(`[inference] addon file size: ${statSync(addonPath).size} bytes`);
  console.log(`[inference] loading addon via require()...`);
  const require_ = createRequire(__filename);
  const { whisper } = require_(addonPath);
  console.log(`[inference] addon loaded successfully`);
  return whisper;
}

let whisperFn: ReturnType<typeof loadAddon> | null = null;

function getWhisper() {
  if (!whisperFn) whisperFn = loadAddon();
  return whisperFn;
}

function ensureModel(): string {
  if (!existsSync(MODEL_PATH)) {
    throw new Error(`Model not found at ${MODEL_PATH}. Add it to models/ or set MODEL_PATH.`);
  }
  return MODEL_PATH;
}

interface WhisperResult {
  transcription: [string, string, string][];
  language?: string;
}

async function transcribe(
  audioPath: string,
  opts: { temperature?: number; language?: string; response_format?: string } = {}
) {
  const whisper = getWhisper();
  const whisperAsync = promisify(whisper);
  const model = ensureModel();

  console.log(`[inference] starting whisper transcription: model=${model}, audio=${audioPath}`);
  const t0 = Date.now();
  const result = (await whisperAsync({
    model,
    fname_inp: audioPath,
    language: opts.language || "en",
    use_gpu: false,
    no_prints: true,
    no_timestamps: false,
    comma_in_time: false,
  })) as WhisperResult;
  console.log(`[inference] transcription completed in ${Date.now() - t0}ms`);

  const segments = (result.transcription || []).map(
    ([start, end, text]: [string, string, string]) => ({ start, end, text: text.trim() })
  );

  return {
    segments,
    text: segments.map((s: { text: string }) => s.text).join(" "),
    ...(result.language ? { language: result.language } : {}),
  };
}

interface ParsedForm {
  fields: Record<string, string>;
  filePath: string;
}

function parseMultipart(req: VercelRequest): Promise<ParsedForm> {
  return new Promise((resolve, reject) => {
    if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });
    const tmpPath = join(TMP_DIR, `${Date.now()}-${Math.random().toString(36).slice(2)}.wav`);
    const fields: Record<string, string> = {};
    let fileFound = false;
    let fileStream: import("stream").Writable | null = null;

    const bb = Busboy({ headers: req.headers as Record<string, string> });

    bb.on("file", (_name: string, stream: Readable, _info: { filename: string }) => {
      fileFound = true;
      const { createWriteStream } = require("fs") as typeof import("fs");
      fileStream = createWriteStream(tmpPath);
      stream.pipe(fileStream);
    });

    bb.on("field", (name: string, val: string) => {
      fields[name] = val;
    });

    bb.on("close", () => {
      if (!fileFound) {
        return reject(new Error('Missing "file" field. Send audio as multipart form: -F "file=@audio.wav"'));
      }
      if (fileStream) {
        fileStream.on("finish", () => resolve({ fields, filePath: tmpPath }));
      } else {
        resolve({ fields, filePath: tmpPath });
      }
    });

    bb.on("error", (err: Error) => reject(err));

    // When bodyParser is enabled, req.body is a Buffer.
    // Create a Readable from it and pipe to busboy.
    if (Buffer.isBuffer(req.body)) {
      const bodyStream = Readable.from(req.body);
      bodyStream.pipe(bb);
    } else if (typeof req.body === "string") {
      const bodyStream = Readable.from(Buffer.from(req.body));
      bodyStream.pipe(bb);
    } else {
      req.pipe(bb);
    }
  });
}

export default async function handler(req: VercelRequest, res: VercelResponse): Promise<void> {
  console.log(`[inference] handler invoked: ${req.method} ${req.url}`);

  if (req.method !== "POST") {
    res.status(405).json({
      error: "Method not allowed. Use POST with multipart form (file=audio).",
    });
    return;
  }

  let filePath: string | undefined;

  try {
    console.log("[inference] parsing multipart form data...");
    const { fields, filePath: fp } = await parseMultipart(req);
    filePath = fp;
    console.log(`[inference] form parsed: file=${fp}, fields=${JSON.stringify(fields)}`);

    const temperature = parseFloat(fields.temperature || "0.0");
    const responseFormat = fields.response_format || "json";
    const language = fields.language || "en";

    const result = await transcribe(filePath, {
      temperature,
      language,
      response_format: responseFormat,
    });

    if (responseFormat === "text") {
      res.setHeader("Content-Type", "text/plain");
      res.status(200).send(result.text);
      return;
    }

    res.status(200).json(result);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("Inference error:", message);
    res.status(500).json({ error: message, cwd: process.cwd() });
  } finally {
    if (filePath) {
      try { unlinkSync(filePath); } catch { /* best-effort */ }
    }
  }
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "50mb",
    },
  },
};
