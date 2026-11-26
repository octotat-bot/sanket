"use client";
// The Live page: camera → 480-row downscale (ADR 007) → MediaPipe Holistic → sign spotter →
// features (same code as training, golden-tested) → model worker → rejection → UI.
// Everything runs on this device; the page can only talk to its own origin (CSP, ADR 006).
import type { HolisticLandmarker } from "@mediapipe/tasks-vision";
import { useEffect, useRef, useState } from "react";
import { frameForMediaPipe } from "@/lib/frames/downscale";
import { ModelClient, type ModelReady } from "@/lib/inference/client";
import { clipToModelInput, featureSpecFrom, type FeatureSpec } from "@/lib/inference/features";
import { DEFAULT_DECIDE, decide, labelsSha256, type Decision, type DecideConfig } from "@/lib/live/decide";
import { Spotter, type SpotterState } from "@/lib/live/spotter";
import { loadFileset, loadSpec, loadVerifiedModels } from "@/lib/mediapipe/assets";
import { createHolistic, toStoredFrame, type Delegate, type StoredFrame } from "@/lib/mediapipe/holistic";

const MODEL_META_URL = "/models/include50.labels.json";

type Boot =
  | { phase: "preparing"; step: string; progress?: number }
  | { phase: "ready" }
  | { phase: "error"; message: string };

type Camera = "off" | "starting" | "on" | "denied" | "missing" | "busy" | "failed";

interface Word {
  id: number;
  label: string;
  display: string;
  category: string;
}

interface Stats {
  fps: number;
  mpMs: number;
  inferMs: number | null;
  speed: number;
}

interface Heard {
  key: number;
  decision: Decision;
}

const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const BODY_EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];

function cameraError(e: unknown): Camera {
  const name = e instanceof DOMException ? e.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return "denied";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "missing";
  if (name === "NotReadableError" || name === "AbortError") return "busy";
  return "failed";
}

const CAMERA_HELP: Partial<Record<Camera, string>> = {
  denied: "Camera access is blocked. Allow the camera for this site in your browser's address bar, then press Start camera again.",
  missing: "No camera was found. Connect a webcam and press Start camera again.",
  busy: "The camera is in use by another app or tab. Close it there, then press Start camera again.",
  failed: "The camera could not start. Reload the page and try again.",
};

export default function LiveClient({ words }: { words: Word[] }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HolisticLandmarker | null>(null);
  const modelRef = useRef<ModelClient | null>(null);
  const readyRef = useRef<ModelReady | null>(null);
  const fsRef = useRef<FeatureSpec | null>(null);
  const decideRef = useRef<DecideConfig>(DEFAULT_DECIDE);
  const streamRef = useRef<MediaStream | null>(null);
  const runningRef = useRef(false);
  const lastTsRef = useRef(0);
  const sizeRef = useRef<[number, number]>([640, 480]);
  const spotterRef = useRef<Spotter | null>(null);
  const busyRef = useRef(false);
  const statsRef = useRef({ frames: 0, since: 0, mpMs: 0, inferMs: null as number | null, speed: 0 });
  const keyRef = useRef(0);

  const [boot, setBoot] = useState<Boot>({ phase: "preparing", step: "Loading the hand and body tracker" });
  const [camera, setCamera] = useState<Camera>("off");
  const [state, setState] = useState<SpotterState>("idle");
  const [recognizing, setRecognizing] = useState(false);
  const [heard, setHeard] = useState<Heard | null>(null);
  const [history, setHistory] = useState<Heard[]>([]);
  const [debug, setDebug] = useState(false);
  const [stats, setStats] = useState<Stats>({ fps: 0, mpMs: 0, inferMs: null, speed: 0 });
  const [runtime, setRuntime] = useState<{ delegate: Delegate; backend: string; file: string } | null>(null);
  const [paused, setPaused] = useState(false);

  // Page load: verify and warm up MediaPipe and the model before the camera goes live
  // (the first GPU frame took 9.4 s on an M3 Pro; CLAUDE.md section 8).
  useEffect(() => {
    let cancelled = false;
    spotterRef.current = new Spotter(() => sizeRef.current);
    (async () => {
      try {
        const spec = await loadSpec();
        const specJson = (await (await fetch("/spec/keypoints.json")).json()) as Parameters<typeof featureSpecFrom>[0] & {
          spec_version: string;
        };
        fsRef.current = featureSpecFrom(specJson);
        const models = await loadVerifiedModels(spec, ["holistic_landmarker"]); // 13.7 MB, the only one used
        const bad = Object.values(models).filter((m) => !m.ok);
        if (bad.length) throw new Error(`MediaPipe model check failed (sha256): ${bad.map((m) => m.name).join(", ")}`);
        const fileset = await loadFileset();
        const { landmarker, delegate } = await createHolistic(fileset, models, "GPU");
        if (cancelled) return landmarker.close();
        landmarkerRef.current = landmarker;
        setBoot({ phase: "preparing", step: "Warming up the tracker (first frames compile GPU shaders)" });
        const blank = new OffscreenCanvas(854, 480);
        blank.getContext("2d")?.fillRect(0, 0, 854, 480);
        for (let i = 0; i < 3; i++) {
          const bmp = await createImageBitmap(blank);
          lastTsRef.current += 33;
          landmarker.detectForVideo(bmp, lastTsRef.current, () => {});
          bmp.close();
        }

        setBoot({ phase: "preparing", step: "Loading the sign model", progress: 0 });
        const expected = await labelsSha256([...words].sort((a, b) => a.id - b.id).map((w) => w.label));
        const client = new ModelClient(MODEL_META_URL, expected, "auto", (loaded, total) =>
          setBoot({ phase: "preparing", step: "Loading the sign model", progress: total ? loaded / total : undefined }),
        );
        modelRef.current = client;
        const ready = await client.ready();
        if (cancelled) return;
        if (ready.meta.spec_version !== specJson.spec_version) {
          throw new Error(
            `model was trained with keypoint spec ${ready.meta.spec_version}, this app uses ${specJson.spec_version}; refusing to run`,
          );
        }
        const cal = ready.meta.calibration;
        if (cal) decideRef.current = { ...DEFAULT_DECIDE, temperature: cal.temperature, minProb: cal.min_prob, minMargin: cal.min_margin };
        readyRef.current = ready;
        setRuntime({ delegate, backend: ready.backend, file: ready.file });
        setBoot({ phase: "ready" });
      } catch (e) {
        if (!cancelled) setBoot({ phase: "error", message: e instanceof Error ? e.message : String(e) });
      }
    })();
    return () => {
      cancelled = true;
      runningRef.current = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
      modelRef.current?.dispose();
    };
  }, [words]);

  // Pause when the tab is hidden; resume cleanly (fresh spotter state) when it comes back.
  useEffect(() => {
    const onVisibility = () => {
      const video = videoRef.current;
      if (document.hidden) {
        runningRef.current = false;
        spotterRef.current?.reset();
        setPaused(true);
      } else if (streamRef.current && video && !runningRef.current) {
        runningRef.current = true;
        setPaused(false);
        loop(video);
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  });

  function draw(f: StoredFrame | null) {
    const c = canvasRef.current;
    const v = videoRef.current;
    if (!c || !v) return;
    const w = v.clientWidth;
    const h = v.clientHeight;
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const g = c.getContext("2d");
    if (!g) return;
    g.clearRect(0, 0, w, h);
    if (!f) return;
    // The video element uses object-cover: map normalized coords through the same crop.
    const vw = v.videoWidth || 4;
    const vh = v.videoHeight || 3;
    const s = Math.max(w / vw, h / vh);
    const ox = (w - vw * s) / 2;
    const oy = (h - vh * s) / 2;
    const P = (p: number[]): [number, number] => [ox + p[0] * vw * s, oy + p[1] * vh * s];
    const lines = (pts: number[][], edges: number[][], color: string, width: number) => {
      g.strokeStyle = color;
      g.lineWidth = width;
      g.lineCap = "round";
      g.beginPath();
      for (const [a, b] of edges) {
        const [x1, y1] = P(pts[a]);
        const [x2, y2] = P(pts[b]);
        g.moveTo(x1, y1);
        g.lineTo(x2, y2);
      }
      g.stroke();
    };
    if (f.pose_present) lines(f.pose, BODY_EDGES, "rgba(236,232,245,0.45)", 3);
    if (f.right_present) lines(f.right, HAND_EDGES, "#f2a93b", 2.5);
    if (f.left_present) lines(f.left, HAND_EDGES, "#45bfb4", 2.5);
  }

  async function classify(clip: NonNullable<ReturnType<Spotter["update"]>["clip"]>) {
    const model = modelRef.current;
    const ready = readyRef.current;
    const fs = fsRef.current;
    if (!model || !ready || !fs || busyRef.current) return;
    busyRef.current = true;
    setRecognizing(true);
    try {
      const rows = clipToModelInput(clip, fs, ready.meta.use_velocity, ready.meta.T);
      const x = new Float32Array(rows.length * rows[0].length);
      rows.forEach((r, i) => x.set(r, i * r.length));
      const { logits, ms } = await model.run(x);
      statsRef.current.inferMs = ms;
      const d = decide(logits, ready.meta.labels, decideRef.current);
      const item = { key: ++keyRef.current, decision: d };
      setHeard(item);
      if (d.accepted) setHistory((h) => [item, ...h].slice(0, 8));
    } finally {
      busyRef.current = false;
      setRecognizing(false);
    }
  }

  function loop(video: HTMLVideoElement) {
    if (!runningRef.current) return;
    video.requestVideoFrameCallback(async (now) => {
      const lm = landmarkerRef.current;
      if (!runningRef.current || !lm) return;
      // Unmirrored camera frame (only the preview is mirrored, via CSS) → 480 rows → Holistic.
      const t0 = performance.now();
      const { bitmap, width, height } = await frameForMediaPipe(video, video.videoWidth, video.videoHeight);
      sizeRef.current = [width, height];
      const ts = (lastTsRef.current = Math.max(lastTsRef.current + 1, now));
      let frame: StoredFrame | null = null;
      lm.detectForVideo(bitmap, ts, (r) => {
        frame = toStoredFrame(r);
      });
      bitmap.close();
      const st = statsRef.current;
      st.mpMs = 0.8 * st.mpMs + 0.2 * (performance.now() - t0);
      const f = frame as StoredFrame | null;
      draw(f);
      if (f && spotterRef.current) {
        const u = spotterRef.current.update(ts, f);
        st.speed = u.speed;
        setState((s) => (s === u.state ? s : u.state));
        if (u.clip) void classify(u.clip);
      }
      st.frames++;
      if (now - st.since >= 1000) {
        setStats({ fps: (st.frames * 1000) / (now - st.since), mpMs: st.mpMs, inferMs: st.inferMs, speed: st.speed });
        st.frames = 0;
        st.since = now;
      }
      loop(video);
    });
  }

  async function startCamera() {
    const video = videoRef.current;
    if (!video || boot.phase !== "ready") return;
    setCamera("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();
      setCamera("on");
      runningRef.current = true;
      statsRef.current.since = performance.now();
      loop(video);
    } catch (e) {
      console.error("camera start failed:", e);
      setCamera(cameraError(e));
    }
  }

  function stopCamera() {
    runningRef.current = false;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    spotterRef.current?.reset();
    draw(null);
    setCamera("off");
    setState("idle");
  }

  const status = recognizing ? "Recognising" : state === "signing" ? "Signing" : camera === "on" ? "Watching for a sign" : null;
  const d = heard?.decision;

  return (
    <section aria-label="Live recognition" className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(18rem,2fr)]">
      <div>
        <div className="relative aspect-[4/3] overflow-hidden rounded-2xl bg-surface ring-1 ring-line">
          {/* Mirrored preview only: MediaPipe receives the unmirrored frames. */}
          <video ref={videoRef} playsInline muted className="absolute inset-0 h-full w-full -scale-x-100 object-cover" />
          <canvas ref={canvasRef} aria-hidden className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100" />
          {camera !== "on" && (
            <div className="absolute inset-0 flex flex-col items-start justify-end gap-4 p-6 sm:p-8">
              {boot.phase === "preparing" && (
                <div role="status" aria-live="polite" className="w-full max-w-sm">
                  <p className="text-lg">{boot.step}…</p>
                  <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-line">
                    <div
                      className={`h-full rounded-full bg-saffron ${boot.progress === undefined ? "w-1/3 animate-pulse" : ""}`}
                      style={boot.progress === undefined ? undefined : { width: `${Math.round(boot.progress * 100)}%` }}
                    />
                  </div>
                </div>
              )}
              {boot.phase === "error" && (
                <p role="alert" className="max-w-md text-lg">
                  Sanket can&apos;t run here: {boot.message}
                </p>
              )}
              {boot.phase === "ready" && (
                <>
                  {CAMERA_HELP[camera] && (
                    <p role="alert" className="max-w-md text-lg">
                      {CAMERA_HELP[camera]}
                    </p>
                  )}
                  <button
                    type="button"
                    onClick={startCamera}
                    disabled={camera === "starting"}
                    className="rounded-full bg-saffron px-6 py-3 text-lg font-bold text-ink transition hover:brightness-110 disabled:opacity-60"
                  >
                    {camera === "starting" ? "Starting camera…" : "Start camera"}
                  </button>
                  <p className="max-w-md text-muted">Your video stays on this device. Nothing is uploaded.</p>
                </>
              )}
            </div>
          )}
          {camera === "on" && (
            <div className="absolute bottom-4 left-4 flex items-center gap-2 rounded-full bg-ink/80 px-4 py-2 backdrop-blur">
              <span
                aria-hidden
                className={`h-2.5 w-2.5 rounded-full ${recognizing ? "bg-paper" : state === "signing" ? "bg-saffron" : "bg-teal"}`}
              />
              <span role="status" aria-live="polite">
                {paused ? "Paused while the tab is hidden" : status}
              </span>
            </div>
          )}
          {camera === "on" && (
            <button
              type="button"
              onClick={stopCamera}
              className="absolute right-4 bottom-4 rounded-full bg-ink/80 px-4 py-2 backdrop-blur hover:bg-ink"
            >
              Stop camera
            </button>
          )}
        </div>
        <div className="mt-4 min-h-12" aria-label="Recently recognised words">
          {history.length > 0 && (
            <ol className="flex flex-wrap gap-2">
              {history.map((h, i) => (
                <li key={h.key} className={`rounded-full px-3 py-1 ring-1 ring-line ${i === 0 ? "bg-surface" : "text-muted"}`}>
                  {words.find((w) => w.label === h.decision.top[0].label)?.display ?? h.decision.top[0].label}
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-6">
        <div aria-live="polite" aria-atomic className="min-h-[9rem]">
          {!d && (
            <p className="text-xl text-muted">
              Raise your hands into view and sign one word from the list below. Lower your hands when you finish.
            </p>
          )}
          {d && d.accepted && (
            <p key={heard.key} className="word-in font-display text-6xl font-extrabold leading-none tracking-tight sm:text-7xl xl:text-8xl">
              {words.find((w) => w.label === d.top[0].label)?.display ?? d.top[0].label}
            </p>
          )}
          {d && !d.accepted && (
            <p key={heard.key} className="word-in font-display text-5xl font-bold leading-none text-muted">
              Not sure
              <span className="mt-3 block font-sans text-lg font-normal">
                {d.reason === "ambiguous" ? "It looked like more than one sign. Try again a little slower." : "That didn't match a word clearly. Try again."}
              </span>
            </p>
          )}
        </div>
        {d && (
          <ul className="flex flex-col gap-3" aria-label="Top guesses with confidence">
            {d.top.map((r) => (
              <li key={r.label}>
                <div className="flex justify-between text-lg">
                  <span>{words.find((w) => w.label === r.label)?.display ?? r.label}</span>
                  <span className="tabular-nums text-muted">{Math.round(r.p * 100)}%</span>
                </div>
                <div className="mt-1 h-2 overflow-hidden rounded-full bg-surface">
                  <div className="bar h-full rounded-full bg-paper/80 transition-[width] duration-500" style={{ width: `${r.p * 100}%` }} />
                </div>
              </li>
            ))}
          </ul>
        )}
        <div className="mt-auto">
          <button type="button" onClick={() => setDebug((v) => !v)} aria-expanded={debug} className="text-muted underline underline-offset-4">
            {debug ? "Hide" : "Show"} performance
          </button>
          {debug && (
            <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-muted tabular-nums" data-testid="debug">
              <dt>Frame rate</dt>
              <dd>{stats.fps.toFixed(1)} fps</dd>
              <dt>Tracker per frame</dt>
              <dd>{stats.mpMs.toFixed(1)} ms</dd>
              <dt>Model per sign</dt>
              <dd>{stats.inferMs === null ? "–" : `${stats.inferMs.toFixed(1)} ms`}</dd>
              <dt>Tracker runs on</dt>
              <dd>{runtime?.delegate ?? "–"}</dd>
              <dt>Model runs on</dt>
              <dd>{runtime ? `${runtime.backend === "webgpu" ? "WebGPU" : "WebAssembly"} (${runtime.file})` : "–"}</dd>
              <dt>Hand speed</dt>
              <dd>{stats.speed.toFixed(2)} shoulder widths/s</dd>
            </dl>
          )}
        </div>
      </div>
    </section>
  );
}
