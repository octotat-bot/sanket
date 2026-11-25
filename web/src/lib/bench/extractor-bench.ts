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

/**
 * Speed rule from ADR 002 (revised 2026-11-25; justified by the frame-rate ablation, not by a
 * benchmark result). Speed is only pass/fail: A vs B is decided on detection quality.
 */
export interface SpeedBar {
  label: string;
  minFps: number;
  maxP95Ms: number;
}
/** What a mid-range device must sustain: the app's real requirement. */
export const MID_RANGE_FLOOR: SpeedBar = { label: "mid-range floor", minFps: 15, maxP95Ms: 66.7 };
/** Stricter bar on the high-end benchmark machine (M3 Pro), leaving headroom for mid-range. */
export const BENCH_MACHINE_BAR: SpeedBar = { label: "M3 Pro bar", minFps: 20, maxP95Ms: 50 };

export function meetsBar(processedFps: number, msP95: number, bar: SpeedBar): boolean {
  return processedFps >= bar.minFps && msP95 <= bar.maxP95Ms;
}

export interface VerifiedModel {
  name: string;
  bytes: Uint8Array;
  expectedSha256: string;
  actualSha256: string;
  ok: boolean;
}

export interface ConfigResult {
  id: ConfigId;
  /** "camera" or "file:<name>" */
  source: string;
  round: number;
  ok: boolean;
  /** Frames processed during warm-up (excluded from the timings below). */
  warmupFrames?: number;
  /** ms of the first (up to) 10 frames after init: shows shader-compile / start-up cost. */
  firstFramesMs?: number[];
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
  /** ADR 002 speed rule: processed fps and p95 against each bar. */
  meetsBenchBar?: boolean;
  meetsFloor?: boolean;
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

/** One frame step: whether any hand was detected, plus hand points (normalized) for the overlay. */
type Point = [number, number];
interface StepResult {
  hand: boolean;
  points: Point[];
}
type Step = (video: HTMLVideoElement, tsMs: number) => StepResult;

const toPoints = (hands: { x: number; y: number }[][]): Point[] =>
  hands.flatMap((h) => h.map((p): Point => [p.x, p.y]));

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
        let out: StepResult = { hand: false, points: [] };
        lm.detectForVideo(video, ts, (r) => {
          const hands = [...r.leftHandLandmarks, ...r.rightHandLandmarks];
          out = { hand: hands.length > 0, points: toPoints(hands) };
        });
        return out;
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
      return { hand: h.landmarks.length > 0, points: toPoints(h.landmarks) };
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

// ---- run plan: randomized config order, recorded so a run can be reproduced ----

/** Small seeded PRNG (mulberry32): same seed → same order. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface PlannedRun {
  id: ConfigId;
  round: number;
}

/** Each round runs every selected config once, in a freshly shuffled order (Fisher-Yates). */
export function planRuns(configs: readonly ConfigId[], rounds: number, seed: number): PlannedRun[] {
  const rand = mulberry32(seed);
  const plan: PlannedRun[] = [];
  for (let round = 1; round <= rounds; round++) {
    const order = [...configs];
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    plan.push(...order.map((id) => ({ id, round })));
  }
  return plan;
}

export interface RunOptions {
  source: string;
  round: number;
  warmupSeconds: number;
  /** Warm-up lasts at least this many frames too (slow configs see few frames per second). */
  minWarmupFrames: number;
  seconds: number;
  /** Called before each config: e.g. rewind a video file so every config sees the same input. */
  restart?: () => Promise<void>;
  onProgress?: (msg: string) => void;
  onFrame?: (points: Point[]) => void;
}

export async function runConfig(
  id: ConfigId,
  video: HTMLVideoElement,
  fileset: WasmFileset,
  models: Record<string, VerifiedModel>,
  opts: RunOptions,
): Promise<ConfigResult> {
  const tag = { id, source: opts.source, round: opts.round };
  let created: { step: Step; close: () => void };
  const t0 = performance.now();
  try {
    created = await createStep(id, fileset, models);
  } catch (e) {
    return { ...tag, ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  const initMs = performance.now() - t0;
  let lastTs = 0;
  const ts = () => (lastTs = Math.max(lastTs + 1, performance.now())); // strictly increasing
  const timed = (): number => {
    const s = performance.now();
    const r = created.step(video, ts());
    const ms = performance.now() - s;
    opts.onFrame?.(r.points);
    lastHand = r.hand;
    return ms;
  };
  let lastHand = false;

  try {
    await opts.restart?.();
    // Warm-up: shader compilation and graph start-up happen on the first frames. They are
    // recorded (firstFramesMs) but kept out of the measured window.
    opts.onProgress?.(`${id} (${opts.source}, round ${opts.round}): warming up`);
    const firstFramesMs: number[] = [];
    let warmupFrames = 0;
    const warmEnd = performance.now() + opts.warmupSeconds * 1000;
    while (performance.now() < warmEnd || warmupFrames < opts.minWarmupFrames) {
      await nextFrame(video);
      const ms = timed();
      if (firstFramesMs.length < 10) firstFramesMs.push(ms);
      warmupFrames++;
    }

    opts.onProgress?.(`${id} (${opts.source}, round ${opts.round}): measuring for ${opts.seconds} s`);
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
      times.push(timed());
      hands += lastHand ? 1 : 0;
    }
    const seconds = (performance.now() - start) / 1000;
    const sourceFrames =
      firstPresented !== null && lastPresented !== null ? lastPresented - firstPresented + 1 : null;
    const processedFps = times.length / seconds;
    const sourceFps = sourceFrames !== null ? sourceFrames / seconds : null;
    const msP95 = percentile(times, 95);
    return {
      ...tag,
      ok: true,
      initMs,
      warmupFrames,
      firstFramesMs,
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
      meetsBenchBar: meetsBar(processedFps, msP95, BENCH_MACHINE_BAR),
      meetsFloor: meetsBar(processedFps, msP95, MID_RANGE_FLOOR),
    };
  } catch (e) {
    return { ...tag, ok: false, initMs, error: e instanceof Error ? e.message : String(e) };
  } finally {
    created.close();
    opts.onFrame?.([]);
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
