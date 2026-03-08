/**
 * Vercel serverless function: transcribes audio via whisper.cpp Node addon.
 *
 * Loads the native addon (bin/whisper-addon.node) and runs inference in-process
 * — no child-process server, no TCP proxy.  The GGML model is loaded once per
 * cold-start and reused across warm invocations (whisper.cpp caches internally
 * within the addon's whisper_init_from_file_with_params call per invocation,
 * but the .node shared object stays loaded in the V8 process).
 *
 * Bundle budget:  addon (~800 KB) + libs (~2.6 MB) + model (~168 MB) ≈ 172 MB
 * well within Vercel's 250 MB compressed limit.
 */

import {
  existsSync, writeFileSync, unlinkSync, mkdirSync,
  statSync, readdirSync, readFileSync, symlinkSync, lstatSync, copyFileSync
} from "fs";
import { join } from "path";
import { promisify } from "util";
import { createRequire } from "module";
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { IncomingForm, type File as FormidableFile } from "formidable";

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

/**
 * Vercel deployments don't preserve symlinks. The linker expects soname
 * aliases (e.g. libwhisper.so.1). We copy the real versioned .so into
 * a writable /tmp dir under every expected alias.
 */
function ensureLibDir(): string {
  if (existsSync(TMP_LIB_DIR) && readdirSync(TMP_LIB_DIR).length > 0) {
    return TMP_LIB_DIR;
  }
  mkdirSync(TMP_LIB_DIR, { recursive: true });

  const entries = readdirSync(LIB_DIR);
  const diag: string[] = [];

  for (const entry of entries) {
    const fullPath = join(LIB_DIR, entry);
    const stat = lstatSync(fullPath);
    diag.push(`${entry}:${stat.size}:${stat.isSymbolicLink() ? "sym" : "file"}`);

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

  console.log(`[inference] lib diag: ${diag.join(", ")}`);
  console.log(`[inference] tmp lib: ${JSON.stringify(readdirSync(TMP_LIB_DIR))}`);
  return TMP_LIB_DIR;
}

function loadAddon(): (params: Record<string, unknown>, cb: (err: Error | null, result?: unknown) => void) => void {
  console.log(`[inference] loadAddon: BIN_DIR=${BIN_DIR}, LIB_DIR=${LIB_DIR}`);
  if (!existsSync(LIB_DIR)) {
    throw new Error(`Shared libraries not found at ${LIB_DIR}. Run: npm run build`);
  }

  const libDir = ensureLibDir();

  // Ensure LD_LIBRARY_PATH includes both the tmp dir (with soname aliases)
  // and the original lib dir. The env var may already be set by Vercel project
  // settings, but we prepend the tmp dir to pick up recreated aliases.
  const sep = ":";
  const current = process.env.LD_LIBRARY_PATH || "";
  const dirs = current ? current.split(sep) : [];
  if (!dirs.includes(libDir)) dirs.unshift(libDir);
  if (!dirs.includes(LIB_DIR)) dirs.unshift(LIB_DIR);
  process.env.LD_LIBRARY_PATH = dirs.join(sep);
  console.log(`[inference] LD_LIBRARY_PATH=${process.env.LD_LIBRARY_PATH}`);

  const addonPath = join(BIN_DIR, "whisper-addon.node");
  if (!existsSync(addonPath)) {
    throw new Error(
      `whisper-addon.node not found at ${addonPath}. Run: npm run build`
    );
  }

  const addonSize = statSync(addonPath).size;
  console.log(`[inference] addon file size: ${addonSize} bytes`);

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
    throw new Error(
      `Model not found at ${MODEL_PATH}. Add it to models/ or set MODEL_PATH.`
    );
  }
  return MODEL_PATH;
}

interface WhisperSegment {
  start: string;
  end: string;
  text: string;
}

interface WhisperResult {
  transcription: [string, string, string][];
  language?: string;
}

async function transcribe(
  audioPath: string,
  opts: {
    temperature?: number;
    language?: string;
    response_format?: string;
  } = {}
): Promise<{ segments: WhisperSegment[]; text: string; language?: string }> {
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

  const segments: WhisperSegment[] = (result.transcription || []).map(
    ([start, end, text]) => ({ start, end, text: text.trim() })
  );

  return {
    segments,
    text: segments.map((s) => s.text).join(" "),
    ...(result.language ? { language: result.language } : {}),
  };
}

function parseForm(req: VercelRequest): Promise<{ fields: Record<string, string>; filePath: string }> {
  return new Promise((resolve, reject) => {
    if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });
    const form = new IncomingForm({
      uploadDir: TMP_DIR,
      keepExtensions: true,
      maxFileSize: 50 * 1024 * 1024,
    });
    form.parse(req, (err, fields, files) => {
      if (err) return reject(err);
      const fileField = files.file;
      if (!fileField) return reject(new Error('Missing "file" field. Send audio as multipart form: -F "file=@audio.wav"'));
      const file: FormidableFile = Array.isArray(fileField) ? fileField[0] : fileField;
      const flatFields: Record<string, string> = {};
      for (const [k, v] of Object.entries(fields)) {
        flatFields[k] = Array.isArray(v) ? v[0] : (v ?? "");
      }
      resolve({ fields: flatFields, filePath: file.filepath });
    });
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
    const { fields, filePath: fp } = await parseForm(req);
    filePath = fp;

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
    const stack = err instanceof Error ? err.stack : undefined;
    console.error("Inference error:", message);
    res.status(500).json({ error: message, stack, cwd: process.cwd() });
  } finally {
    if (filePath) {
      try { unlinkSync(filePath); } catch { /* best-effort cleanup */ }
    }
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
};
