/**
 * Vercel serverless function: runs bundled whisper-server (CPU-only prebuilt)
 * and proxies POST /inference to it. Stays within 250MB bundle, 1GB memory.
 */

import { existsSync, mkdirSync } from "fs";
import { createServer, createConnection } from "net";
import { join } from "path";
import { spawn, type ChildProcess } from "child_process";

const BIN_DIR = process.env.BIN_DIR || join(process.cwd(), "bin");
const MODEL_PATH =
  process.env.MODEL_PATH ||
  join(process.cwd(), "models", "ggml-ipa-whisper-small-q5_0.bin");

let serverProcess: ChildProcess | null = null;
let serverPort: number | null = null;

function getBinPath(): string {
  const bin = join(BIN_DIR, "whisper-server");
  if (!existsSync(bin)) {
    throw new Error(
      `whisper-server not found at ${bin}. Run npm run build (or ensure bin/ is deployed).`
    );
  }
  return bin;
}

function ensureModel(): string {
  if (!existsSync(MODEL_PATH)) {
    throw new Error(
      `Model not found at ${MODEL_PATH}. Add the model to models/ or set MODEL_PATH.`
    );
  }
  return MODEL_PATH;
}

function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createServer();
    s.listen(0, "127.0.0.1", () => {
      const port = (s.address() as { port: number }).port;
      s.close(() => resolve(port));
    });
    s.on("error", reject);
  });
}

function startServer(): Promise<number> {
  return new Promise((resolve, reject) => {
    if (serverPort !== null && serverProcess?.exitCode === null) {
      return resolve(serverPort);
    }
    const bin = getBinPath();
    const model = ensureModel();
    // Vercel: filesystem is read-only except /tmp (see vercel.com/docs/functions/runtimes#file-system-support)
    const tmpDir = "/tmp/whisper-inference";
    if (!existsSync(tmpDir)) mkdirSync(tmpDir, { recursive: true });

    getFreePort()
      .then((port) => {
        const proc = spawn(
          bin,
          [
            "-m",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            String(port),
            "-l",
            "en",
            // no --convert: no ffmpeg in bundle; client should send WAV
          ],
          {
            cwd: process.cwd(),
            env: { ...process.env, TMPDIR: tmpDir },
            stdio: ["ignore", "pipe", "pipe"],
          }
        );
        serverProcess = proc;
        serverPort = port;

        let stderr = "";
        proc.stderr?.on("data", (c) => (stderr += c.toString()));
        proc.on("error", (err) => reject(err));
        proc.on("exit", (code) => {
          serverProcess = null;
          serverPort = null;
          if (code !== 0 && code !== null) {
            console.error("whisper-server stderr:", stderr);
          }
        });

        // Wait for server to accept TCP connections
        const deadline = Date.now() + 60000;
        const tryConnect = () => {
          const sock = createConnection(
            { host: "127.0.0.1", port },
            () => {
              sock.destroy();
              resolve(port);
            }
          );
          sock.on("error", () => {
            if (Date.now() < deadline) setTimeout(tryConnect, 200);
            else reject(new Error("whisper-server failed to start"));
          });
        };
        setTimeout(tryConnect, 500);
      })
      .catch(reject);
  });
}

async function waitForServer(port: number): Promise<void> {
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/`);
      if (r.ok) return;
    } catch {
      await new Promise((r) => setTimeout(r, 300));
    }
  }
  throw new Error("whisper-server did not become ready");
}

export default async function handler(
  req: Request
): Promise<Response> {
  if (req.method !== "POST") {
    return new Response(
      JSON.stringify({ error: "Method not allowed. Use POST with multipart form (file=audio)." }),
      { status: 405, headers: { "Content-Type": "application/json" } }
    );
  }

  const port = await startServer();
  await waitForServer(port);

  const contentType = req.headers.get("content-type") || "";
  const body = req.body;
  if (!body) {
    return new Response(
      JSON.stringify({ error: "No request body" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const backendUrl = `http://127.0.0.1:${port}/inference`;
  const proxyReq = new Request(backendUrl, {
    method: "POST",
    headers: { "Content-Type": contentType },
    body,
    duplex: "half",
  } as RequestInit);
  const res = await fetch(proxyReq);
  const outHeaders = new Headers();
  res.headers.forEach((v, k) => outHeaders.set(k, v));
  return new Response(res.body, { status: res.status, headers: outHeaders });
}
