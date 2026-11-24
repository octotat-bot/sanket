// ADR 002, measure (b): landmark extraction speed in a real browser.
// Runs Holistic vs Hand+Pose, each on the GPU and CPU delegates, on live video frames.
//
// Everything stays on this device: models, WASM and spec are served from our own origin, and
// results are only shown on the page or saved as a local file.
import {
  FilesetResolver,
  HandLandmarker,
  HolisticLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import { mean, percentile, sha256Hex } from "./stats";

// tasks-vision 1.2.3 does not export the WasmFileset type, so derive it.
type WasmFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

export const CONFIG_IDS = ["holistic-gpu", "handpose-gpu", "holistic-cpu", "handpose-cpu"] as const;
export type ConfigId = (typeof CONFIG_IDS)[number];

/** Decision rule from ADR 002: the extractor alone must sustain at least 30 fps. */
export const TARGET_FPS = 30;

export interface VerifiedModel {
  name: string;
  bytes: Uint8Array;
  expectedSha256: string;
  actualSha256: string;
  ok: boolean;
}

export interface ConfigResult {
  id: ConfigId;
  ok: boolean;
  error?: string;
  initMs?: number;
  processedFrames?: number;
  /** Frames the video source presented during the run (from requestVideoFrameCallback). */
  sourceFrames?: number | null;
  seconds?: number;
  processedFps?: number;
  sourceFps?: number | null;
  msP50?: number;
  msP95?: number;
  msMean?: number;
  /** 1000 / p95 ms: the frame rate the extractor could hold if frames arrived faster. */
  headroomFps?: number;
  handRate?: number;
  /** p95 ms <= 1000/30 and processed fps >= 95% of source fps. */
  sustainsTarget?: boolean;
}

interface Spec {
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

/** Fetch each pinned .task file and check its sha256 against spec/keypoints.json. */
export async function loadVerifiedModels(spec: Spec): Promise<Record<string, VerifiedModel>> {
  const out: Record<string, VerifiedModel> = {};
  for (const [name, pin] of Object.entries(spec.extractor.models)) {
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

type Delegate = "GPU" | "CPU";

/** One frame step: returns whether any hand was detected. */
type Step = (video: HTMLVideoElement, tsMs: number) => boolean;

async function createStep(
  id: ConfigId,
  fileset: WasmFileset,
  models: Record<string, VerifiedModel>,
): Promise<{ step: Step; close: () => void }> {
  const delegate: Delegate = id.endsWith("gpu") ? "GPU" : "CPU";
  // Hand MediaPipe a copy: it may take ownership of the buffer it is given.
  const buf = (name: string) => models[name].bytes.slice();
  if (id.startsWith("holistic")) {
    const lm = await HolisticLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: buf("holistic_landmarker"), delegate },
      runningMode: "VIDEO",
      outputFaceBlendshapes: false,
      outputPoseSegmentationMasks: false,
    });
    return {
      step: (video, ts) => {
        let hand = false;
        lm.detectForVideo(video, ts, (r) => {
          hand = r.leftHandLandmarks.length > 0 || r.rightHandLandmarks.length > 0;
        });
        return hand;
      },
      close: () => lm.close(),
    };
  }
  const hand = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: buf("hand_landmarker"), delegate },
    runningMode: "VIDEO",
    numHands: 2,
  });
  const pose = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: buf("pose_landmarker_full"), delegate },
    runningMode: "VIDEO",
    outputSegmentationMasks: false,
  });
  return {
    step: (video, ts) => {
      const h = hand.detectForVideo(video, ts);
      pose.detectForVideo(video, ts, () => {});
      return h.landmarks.length > 0;
    },
    close: () => {
      hand.close();
      pose.close();
    },
  };
}

interface FrameInfo {
  presentedFrames: number | null;
}

/** Resolve on the next *new* video frame (requestVideoFrameCallback, or rAF fallback). */
function nextFrame(video: HTMLVideoElement): Promise<FrameInfo> {
  if ("requestVideoFrameCallback" in video) {
    return new Promise((resolve) =>
      video.requestVideoFrameCallback((_now, meta) =>
        resolve({ presentedFrames: meta.presentedFrames }),
      ),
    );
  }
  const start = (video as HTMLVideoElement).currentTime;
  return new Promise((resolve) => {
    const tick = () => {
      if ((video as HTMLVideoElement).currentTime !== start) resolve({ presentedFrames: null });
      else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

export interface RunOptions {
  warmupSeconds: number;
  seconds: number;
  onProgress?: (msg: string) => void;
}

export async function runConfig(
  id: ConfigId,
  video: HTMLVideoElement,
  fileset: WasmFileset,
  models: Record<string, VerifiedModel>,
  opts: RunOptions,
): Promise<ConfigResult> {
  let created: { step: Step; close: () => void };
  const t0 = performance.now();
  try {
    created = await createStep(id, fileset, models);
  } catch (e) {
    return { id, ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  const initMs = performance.now() - t0;
  let lastTs = 0;
  const ts = () => (lastTs = Math.max(lastTs + 1, performance.now())); // strictly increasing

  try {
    opts.onProgress?.(`${id}: warming up`);
    const warmEnd = performance.now() + opts.warmupSeconds * 1000;
    while (performance.now() < warmEnd) {
      await nextFrame(video);
      created.step(video, ts());
    }

    opts.onProgress?.(`${id}: measuring for ${opts.seconds} s`);
    const times: number[] = [];
    let hands = 0;
    let firstPresented: number | null = null;
    let lastPresented: number | null = null;
    const start = performance.now();
    const end = start + opts.seconds * 1000;
    while (performance.now() < end) {
      const info = await nextFrame(video);
      if (firstPresented === null) firstPresented = info.presentedFrames;
      lastPresented = info.presentedFrames;
      const s = performance.now();
      hands += created.step(video, ts()) ? 1 : 0;
      times.push(performance.now() - s);
    }
    const seconds = (performance.now() - start) / 1000;
    const sourceFrames =
      firstPresented !== null && lastPresented !== null ? lastPresented - firstPresented + 1 : null;
    const processedFps = times.length / seconds;
    const sourceFps = sourceFrames !== null ? sourceFrames / seconds : null;
    const msP95 = percentile(times, 95);
    return {
      id,
      ok: true,
      initMs,
      processedFrames: times.length,
      sourceFrames,
      seconds,
      processedFps,
      sourceFps,
      msP50: percentile(times, 50),
      msP95,
      msMean: mean(times),
      headroomFps: 1000 / msP95,
      handRate: times.length ? hands / times.length : 0,
      sustainsTarget:
        msP95 <= 1000 / TARGET_FPS && (sourceFps === null || processedFps >= 0.95 * sourceFps),
    };
  } catch (e) {
    return { id, ok: false, initMs, error: e instanceof Error ? e.message : String(e) };
  } finally {
    created.close();
  }
}

export interface DeviceInfo {
  userAgent: string;
  hardwareConcurrency: number;
  deviceMemoryGb: number | null;
  webglRenderer: string | null;
  webgpuAdapter: string | null;
  crossOriginIsolated: boolean;
}

export async function deviceInfo(): Promise<DeviceInfo> {
  let webglRenderer: string | null = null;
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    const ext = gl?.getExtension("WEBGL_debug_renderer_info");
    webglRenderer = gl && ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : null;
  } catch {
    webglRenderer = null;
  }
  let webgpuAdapter: string | null = null;
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (gpu) {
    try {
      const adapter = (await gpu.requestAdapter()) as {
        info?: { vendor?: string; architecture?: string; description?: string };
      } | null;
      webgpuAdapter = adapter
        ? [adapter.info?.vendor, adapter.info?.architecture, adapter.info?.description]
            .filter(Boolean)
            .join(" / ") || "available"
        : "no adapter";
    } catch {
      webgpuAdapter = "error";
    }
  }
  return {
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemoryGb: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? null,
    webglRenderer,
    webgpuAdapter,
    crossOriginIsolated: self.crossOriginIsolated,
  };
}
