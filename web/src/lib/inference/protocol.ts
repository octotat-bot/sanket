// Messages between the page and the model worker (model.worker.ts).

export type Backend = "webgpu" | "wasm";

export interface ModelMeta {
  labels: string[];
  labels_sha256: string;
  T: number;
  use_velocity: boolean;
  spec_version: string;
  arch: string;
  input: { name: string; shape: [string, number, number] };
  files: { fp32: string; int8: string };
  sha256: Record<string, string>;
  calibration?: { temperature: number; min_prob: number; min_margin: number };
}

export type ToWorker =
  | { kind: "init"; metaUrl: string; expectedLabelsSha256: string; prefer: Backend | "auto" }
  | { kind: "run"; id: number; x: Float32Array };

export type FromWorker =
  | { kind: "progress"; loaded: number; total: number }
  | { kind: "ready"; backend: Backend; file: string; meta: ModelMeta; warmupMs: number[] }
  | { kind: "result"; id: number; logits: Float32Array; ms: number }
  | { kind: "error"; id?: number; message: string };
