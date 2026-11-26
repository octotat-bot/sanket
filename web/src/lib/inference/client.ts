// Main-thread handle to the model worker: one promise per request.
import type { Backend, FromWorker, ModelMeta, ToWorker } from "./protocol";

export interface ModelReady {
  backend: Backend;
  file: string;
  meta: ModelMeta;
  warmupMs: number[];
}

export class ModelClient {
  private worker: Worker;
  private next = 1;
  private waiting = new Map<number, { resolve: (r: { logits: Float32Array; ms: number }) => void; reject: (e: Error) => void }>();
  private readyPromise: Promise<ModelReady>;

  constructor(metaUrl: string, expectedLabelsSha256: string, prefer: Backend | "auto", onProgress?: (loaded: number, total: number) => void) {
    this.worker = new Worker(new URL("./model.worker.ts", import.meta.url), { type: "module" });
    this.readyPromise = new Promise((resolve, reject) => {
      this.worker.onmessage = (e: MessageEvent<FromWorker>) => {
        const m = e.data;
        if (m.kind === "progress") onProgress?.(m.loaded, m.total);
        else if (m.kind === "ready") resolve({ backend: m.backend, file: m.file, meta: m.meta, warmupMs: m.warmupMs });
        else if (m.kind === "result") {
          this.waiting.get(m.id)?.resolve({ logits: m.logits, ms: m.ms });
          this.waiting.delete(m.id);
        } else if (m.kind === "error") {
          if (m.id === undefined) reject(new Error(m.message));
          else {
            this.waiting.get(m.id)?.reject(new Error(m.message));
            this.waiting.delete(m.id);
          }
        }
      };
      this.worker.onerror = (e) => reject(new Error(e.message || "model worker failed to start"));
    });
    this.post({ kind: "init", metaUrl, expectedLabelsSha256, prefer });
  }

  ready(): Promise<ModelReady> {
    return this.readyPromise;
  }

  run(x: Float32Array): Promise<{ logits: Float32Array; ms: number }> {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      this.post({ kind: "run", id, x }, [x.buffer]);
    });
  }

  dispose(): void {
    this.worker.terminate();
    for (const w of this.waiting.values()) w.reject(new Error("disposed"));
    this.waiting.clear();
  }

  private post(m: ToWorker, transfer: Transferable[] = []) {
    this.worker.postMessage(m, transfer);
  }
}
