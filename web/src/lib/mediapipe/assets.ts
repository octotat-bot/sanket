// Pinned MediaPipe assets, loaded from our own origin and checked against spec/keypoints.json.
// Shared by the extractor benchmark, the recording tool and (later) the app.
import { FilesetResolver } from "@mediapipe/tasks-vision";
import { sha256Hex } from "../bench/stats";

// tasks-vision 1.2.3 does not export the WasmFileset type, so derive it.
export type WasmFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

export interface VerifiedModel {
  name: string;
  bytes: Uint8Array;
  expectedSha256: string;
  actualSha256: string;
  ok: boolean;
}

export interface Spec {
  extractor: {
    mediapipe_version: string;
    models: Record<string, { sha256: string; bytes: number }>;
  };
}

export async function loadSpec(): Promise<Spec> {
  const res = await fetch("/spec/keypoints.json");
  if (!res.ok) throw new Error(`spec/keypoints.json: HTTP ${res.status}`);
  return (await res.json()) as Spec;
}

/** Fetch pinned .task files (all, or only `names`) and check each sha256 against spec/keypoints.json. */
export async function loadVerifiedModels(spec: Spec, names?: string[]): Promise<Record<string, VerifiedModel>> {
  const out: Record<string, VerifiedModel> = {};
  for (const [name, pin] of Object.entries(spec.extractor.models)) {
    if (names && !names.includes(name)) continue;
    const res = await fetch(`/models/mediapipe/${name}.task`);
    if (!res.ok) throw new Error(`${name}.task: HTTP ${res.status} (run \`make models\`)`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    const actual = await sha256Hex(bytes);
    out[name] = {
      name,
      bytes,
      expectedSha256: pin.sha256,
      actualSha256: actual,
      ok: actual === pin.sha256,
    };
  }
  return out;
}

export function loadFileset(): Promise<WasmFileset> {
  return FilesetResolver.forVisionTasks("/mediapipe/wasm");
}

