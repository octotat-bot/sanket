// The sign classifier, off the main thread (CLAUDE.md Phase 5). Loads <name>.labels.json, checks
// the labels hash against the app's vocabulary (label-drift guard, CLAUDE.md section 8) and the
// model file's sha256, then runs onnxruntime-web: WebGPU when the browser has it, else WASM
// (SIMD, and threads when the page is cross-origin isolated). Runtime files come from our own
// origin (/ort/, copied by scripts/copy-mediapipe-assets.mjs), never a CDN.
/// <reference lib="webworker" />
import type { Backend, FromWorker, ModelMeta, ToWorker } from "./protocol";

type Ort = typeof import("onnxruntime-web");
type Session = Awaited<ReturnType<Ort["InferenceSession"]["create"]>>;

const post = (m: FromWorker, transfer: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(m, transfer);

let ort: Ort | null = null;
let session: Session | null = null;
let meta: ModelMeta | null = null;
let shape: [number, number] = [0, 0];

async function hex(buf: ArrayBuffer | Uint8Array): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", buf as BufferSource);
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchWithProgress(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ kind: "progress", loaded, total });
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

async function init(msg: Extract<ToWorker, { kind: "init" }>) {
  const m = (await (await fetch(msg.metaUrl)).json()) as ModelMeta;
  const recomputed = await hex(new TextEncoder().encode(JSON.stringify(m.labels)));
  if (m.labels_sha256 !== msg.expectedLabelsSha256 || recomputed !== msg.expectedLabelsSha256) {
    throw new Error("model labels do not match the app's vocabulary (labels hash mismatch); refusing to load");
  }
  const hasGpu = "gpu" in navigator && !!(await (navigator as Navigator & { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter().catch(() => null));
  const backend: Backend = msg.prefer === "auto" ? (hasGpu ? "webgpu" : "wasm") : msg.prefer;
  // fp32 on WebGPU (its int8 kernels are incomplete); dynamic int8 on WASM (smaller, CPU-friendly).
  const file = backend === "webgpu" ? m.files.fp32 : m.files.int8;
  const base = new URL(msg.metaUrl, self.location.href);
  const bytes = await fetchWithProgress(new URL(file, base).href);
  if ((await hex(bytes)) !== m.sha256[file]) throw new Error(`${file}: sha256 mismatch`);

  ort = backend === "webgpu" ? await import("onnxruntime-web/webgpu") : await import("onnxruntime-web/wasm");
  ort.env.wasm.wasmPaths = `${self.location.origin}/ort/`;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
    graphOptimizationLevel: "all",
  });
  meta = m;
  shape = [m.input.shape[1], m.input.shape[2]];
  // Warm-up: the first runs compile kernels / shaders; keep them out of any latency number.
  const warmupMs: number[] = [];
  for (let i = 0; i < 3; i++) warmupMs.push((await run(new Float32Array(shape[0] * shape[1]))).ms);
  post({ kind: "ready", backend, file, meta: m, warmupMs });
}

async function run(x: Float32Array): Promise<{ logits: Float32Array; ms: number }> {
  if (!ort || !session || !meta) throw new Error("model not loaded");
  if (x.length !== shape[0] * shape[1]) throw new Error(`expected ${shape[0]}×${shape[1]} input, got ${x.length}`);
  const t0 = performance.now();
  const input = new ort.Tensor("float32", x, [1, shape[0], shape[1]]);
  const out = await session.run({ [meta.input.name]: input });
  const logits = new Float32Array((out.logits.data as Float32Array).slice());
  input.dispose();
  out.logits.dispose();
  return { logits, ms: performance.now() - t0 };
}

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  try {
    if (msg.kind === "init") await init(msg);
    else {
      const r = await run(msg.x);
      post({ kind: "result", id: msg.id, logits: r.logits, ms: r.ms }, [r.logits.buffer]);
    }
  } catch (err) {
    post({ kind: "error", id: msg.kind === "run" ? msg.id : undefined, message: err instanceof Error ? err.message : String(err) });
  }
};
