/**
 * Vercel serverless function: transcribes audio via whisper.cpp Node addon.
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
  console.log(`[inference] addon size: ${statSync(addonPath).size}, loading...`);
  const require_ = createRequire(__filename);
  try {
    const { whisper } = require_(addonPath);
    console.log("[inference] addon loaded OK");
    return whisper;
  } catch (err) {
    console.error("[inference] addon load FAILED:", err);
    throw err;
  }
}

let whisperFn: ReturnType<typeof loadAddon> | null = null;
function getWhisper() {
  if (!whisperFn) whisperFn = loadAddon();
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
    if (buf[partStart] === 0x2d && buf[partStart + 1] === 0x2d) break; // --
    const headerEnd = buf.indexOf(Buffer.from("\r\n\r\n"), partStart);
    if (headerEnd === -1) break;
    const headers = buf.subarray(partStart + 2, headerEnd).toString("utf8");
    const bodyStart = headerEnd + 4;
    const nextSep = buf.indexOf(sep, bodyStart);
    const bodyEnd = nextSep === -1 ? buf.length : nextSep - 2; // strip \r\n before next boundary
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

export default async function handler(req: VercelRequest, res: VercelResponse): Promise<void> {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed. Use POST with multipart form (file=audio)." });
    return;
  }

  let tmpPath: string | undefined;
  try {
    // Try reading from the raw body that Vercel's body parser may have stored
    let rawBody: Buffer;
    if (Buffer.isBuffer(req.body)) {
      rawBody = req.body;
    } else if (typeof req.body === "string") {
      rawBody = Buffer.from(req.body, "binary");
    } else {
      // Body parser couldn't parse multipart; body is undefined but
      // the raw bytes may still be available via (req as any).rawBody
      const raw = (req as unknown as { rawBody?: Buffer }).rawBody;
      if (Buffer.isBuffer(raw)) {
        rawBody = raw;
      } else {
        rawBody = await collectBody(req);
      }
    }
    console.log(`[inference] body: ${rawBody.length} bytes`);
    const contentType = req.headers["content-type"] || "";
    const boundaryMatch = contentType.match(/boundary=(.+)/);
    if (!boundaryMatch) {
      res.status(400).json({ error: "Missing multipart boundary in Content-Type" });
      return;
    }
    const { file, fields } = parseMultipartBuffer(rawBody, boundaryMatch[1]);
    if (!file || file.length === 0) {
      res.status(400).json({ error: 'Missing "file" field. Send audio as: -F "file=@audio.wav"' });
      return;
    }

    if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });
    tmpPath = join(TMP_DIR, `${Date.now()}-${Math.random().toString(36).slice(2)}.wav`);
    writeFileSync(tmpPath, file);

    const language = fields.language || "en";
    const responseFormat = fields.response_format || "json";

    const result = await transcribe(tmpPath, { language });

    if (responseFormat === "text") {
      res.setHeader("Content-Type", "text/plain");
      res.status(200).send(result.text);
      return;
    }
    res.status(200).json(result);
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
