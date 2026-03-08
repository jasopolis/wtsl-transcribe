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

import { existsSync, writeFileSync, unlinkSync, mkdirSync } from "fs";
import { join } from "path";
import { promisify } from "util";
import { createRequire } from "module";

const BIN_DIR = process.env.BIN_DIR || join(process.cwd(), "bin");
const LIB_DIR = process.env.LIB_DIR || join(BIN_DIR, "lib");
const MODEL_PATH =
  process.env.MODEL_PATH ||
  join(process.cwd(), "models", "ggml-ipa-whisper-small-q5_0.bin");
const TMP_DIR = "/tmp/whisper-inference";

function loadAddon(): (params: Record<string, unknown>, cb: (err: Error | null, result?: unknown) => void) => void {
  console.log(`[inference] loadAddon: BIN_DIR=${BIN_DIR}, LIB_DIR=${LIB_DIR}`);
  if (!existsSync(LIB_DIR)) {
    throw new Error(`Shared libraries not found at ${LIB_DIR}. Run: npm run build`);
  }
  const sep = ":";
  const current = process.env.LD_LIBRARY_PATH || "";
  if (!current.split(sep).includes(LIB_DIR)) {
    process.env.LD_LIBRARY_PATH = LIB_DIR + (current ? sep + current : "");
  }
  console.log(`[inference] LD_LIBRARY_PATH=${process.env.LD_LIBRARY_PATH}`);

  const addonPath = join(BIN_DIR, "whisper-addon.node");
  if (!existsSync(addonPath)) {
    throw new Error(
      `whisper-addon.node not found at ${addonPath}. Run: npm run build`
    );
  }

  const { statSync, readdirSync } = require("fs") as typeof import("fs");
  const addonSize = statSync(addonPath).size;
  console.log(`[inference] addon file size: ${addonSize} bytes`);
  const libFiles = readdirSync(LIB_DIR);
  console.log(`[inference] lib dir contents: ${JSON.stringify(libFiles)}`);

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

export default async function handler(req: Request): Promise<Response> {
  console.log(`[inference] handler invoked: ${req.method} ${new URL(req.url).pathname}`);

  if (req.method !== "POST") {
    return new Response(
      JSON.stringify({
        error: "Method not allowed. Use POST with multipart form (file=audio).",
      }),
      { status: 405, headers: { "Content-Type": "application/json" } }
    );
  }

  let formData: FormData;
  try {
    formData = await req.formData();
  } catch {
    return new Response(
      JSON.stringify({ error: "Invalid multipart form data" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const file = formData.get("file");
  if (!file || !(file instanceof File)) {
    return new Response(
      JSON.stringify({
        error: 'Missing "file" field. Send audio as multipart form: -F "file=@audio.wav"',
      }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const temperature = parseFloat(
    (formData.get("temperature") as string) || "0.0"
  );
  const responseFormat =
    (formData.get("response_format") as string) || "json";
  const language = (formData.get("language") as string) || "en";

  if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });
  const tmpPath = join(TMP_DIR, `${Date.now()}-${Math.random().toString(36).slice(2)}.wav`);

  try {
    const buf = Buffer.from(await file.arrayBuffer());
    writeFileSync(tmpPath, buf);

    const result = await transcribe(tmpPath, {
      temperature,
      language,
      response_format: responseFormat,
    });

    if (responseFormat === "text") {
      return new Response(result.text, {
        status: 200,
        headers: { "Content-Type": "text/plain" },
      });
    }

    return new Response(JSON.stringify(result), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("Inference error:", message);
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  } finally {
    try {
      unlinkSync(tmpPath);
    } catch {
      // best-effort cleanup
    }
  }
}
