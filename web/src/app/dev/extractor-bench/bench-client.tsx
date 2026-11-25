"use client";

import { useRef, useState } from "react";
import {
  CONFIG_IDS,
  type ConfigId,
  type ConfigResult,
  type DeviceInfo,
  deviceInfo,
  loadFileset,
  loadSpec,
  loadVerifiedModels,
  planRuns,
  runConfig,
  BENCH_MACHINE_BAR,
  MID_RANGE_FLOOR,
} from "@/lib/bench/extractor-bench";

type SourceMode = "file" | "camera" | "both";

interface Report {
  createdAt: string;
  mediapipeVersion: string;
  sourceMode: SourceMode;
  file: string | null;
  seed: number;
  rounds: number;
  order: string[];
  warmupSeconds: number;
  minWarmupFrames: number;
  seconds: number;
  videoSizes: Record<string, [number, number]>;
  device: DeviceInfo;
  models: { name: string; sha256: string; verified: boolean }[];
  results: ConfigResult[];
}

const LABELS: Record<ConfigId, string> = {
  "holistic-gpu": "A · Holistic · GPU",
  "holistic-cpu": "A · Holistic · CPU",
  "handpose-gpu": "B · Hand + Pose · GPU",
  "handpose-cpu": "B · Hand + Pose · CPU",
};
const MIN_WARMUP_FRAMES = 60;

function fmt(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined || Number.isNaN(x) ? "–" : x.toFixed(digits);
}

export default function BenchClient() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<SourceMode>("file");
  const [file, setFile] = useState<File | null>(null);
  const [seconds, setSeconds] = useState(15);
  const [warmup, setWarmup] = useState(8);
  const [rounds, setRounds] = useState(2);
  // 0 = pick a random seed when the run starts (render must stay deterministic for prerendering).
  const [seed, setSeed] = useState(0);
  const [selected, setSelected] = useState<ConfigId[]>([...CONFIG_IDS]);
  const [status, setStatus] = useState("Idle.");
  const [running, setRunning] = useState(false);
  const [mirrored, setMirrored] = useState(false);
  const [report, setReport] = useState<Report | null>(null);

  /** Detected hand points, drawn on screen only (never stored), to see *why* a config misses. */
  function drawOverlay(points: [number, number][]) {
    const c = overlayRef.current;
    const v = videoRef.current;
    if (!c || !v) return;
    c.width = v.clientWidth;
    c.height = v.clientHeight;
    const g = c.getContext("2d");
    if (!g) return;
    g.clearRect(0, 0, c.width, c.height);
    g.fillStyle = "#22d3ee";
    for (const [x, y] of points) g.fillRect(x * c.width - 2, y * c.height - 2, 4, 4);
  }

  async function attachCamera(video: HTMLVideoElement): Promise<() => void> {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    video.srcObject = stream;
    video.loop = false;
    await video.play();
    return () => {
      stream.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    };
  }

  async function attachFile(video: HTMLVideoElement, f: File): Promise<() => void> {
    const url = URL.createObjectURL(f);
    video.srcObject = null;
    video.src = url;
    video.loop = true;
    video.muted = true;
    await video.play();
    return () => {
      video.pause();
      video.removeAttribute("src");
      video.load();
      URL.revokeObjectURL(url);
    };
  }

  /** Rewind to the first frame so every config sees exactly the same input. */
  async function rewind(video: HTMLVideoElement) {
    video.pause();
    video.currentTime = 0;
    await new Promise<void>((r) => video.addEventListener("seeked", () => r(), { once: true }));
    await video.play();
  }

  async function start() {
    const video = videoRef.current;
    if (!video || running) return;
    if (mode !== "camera" && !file) {
      setStatus("Error: choose an INCLUDE video file first.");
      return;
    }
    setRunning(true);
    setReport(null);
    let detach: (() => void) | null = null;
    try {
      setStatus("Loading spec and models, and checking their sha256…");
      const spec = await loadSpec();
      const models = await loadVerifiedModels(spec);
      const bad = Object.values(models).filter((m) => !m.ok);
      if (bad.length) throw new Error(`sha256 mismatch: ${bad.map((m) => m.name).join(", ")}`);
      const fileset = await loadFileset();
      const runSeed = seed || crypto.getRandomValues(new Uint32Array(1))[0];
      setSeed(runSeed);
      const plan = planRuns(selected, rounds, runSeed);
      const sources: ("file" | "camera")[] = mode === "both" ? ["file", "camera"] : [mode];
      const results: ConfigResult[] = [];
      const videoSizes: Record<string, [number, number]> = {};
      for (const src of sources) {
        const label = src === "file" ? `file:${file!.name}` : "camera";
        setMirrored(src === "camera");
        setStatus(src === "camera" ? "Starting camera…" : "Starting video file…");
        detach = src === "camera" ? await attachCamera(video) : await attachFile(video, file!);
        videoSizes[label] = [video.videoWidth, video.videoHeight];
        for (const run of plan) {
          results.push(
            await runConfig(run.id, video, fileset, models, {
              source: label,
              round: run.round,
              warmupSeconds: warmup,
              minWarmupFrames: MIN_WARMUP_FRAMES,
              seconds,
              restart: src === "file" ? () => rewind(video) : undefined,
              onProgress: (m) => setStatus(m),
              onFrame: drawOverlay,
            }),
          );
        }
        detach();
        detach = null;
      }
      const r: Report = {
        createdAt: new Date().toISOString(),
        mediapipeVersion: spec.extractor.mediapipe_version,
        sourceMode: mode,
        file: file?.name ?? null,
        seed: runSeed,
        rounds,
        order: plan.map((p) => `${p.round}:${p.id}`),
        warmupSeconds: warmup,
        minWarmupFrames: MIN_WARMUP_FRAMES,
        seconds,
        videoSizes,
        device: await deviceInfo(),
        models: Object.values(models).map((m) => ({
          name: m.name,
          sha256: m.actualSha256,
          verified: m.ok,
        })),
        results,
      };
      setReport(r);
      (window as unknown as { __benchReport?: Report }).__benchReport = r;
      setStatus("Done.");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(`Error: ${msg}`);
      (window as unknown as { __benchError?: string }).__benchError = msg;
    } finally {
      detach?.();
      setRunning(false);
    }
  }

  function download() {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `extractor-bench-${report.createdAt.replace(/[:.]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const perSource = mode === "both" ? 2 : 1;
  const estSeconds = perSource * rounds * selected.length * (seconds + warmup + 2) + 5;

  return (
    <div className="flex flex-col gap-6">
      <section className="grid gap-4 rounded-lg border border-zinc-300 p-4 dark:border-zinc-700 sm:grid-cols-2">
        <fieldset className="flex flex-col gap-2" disabled={running}>
          <legend className="font-medium">Video source</legend>
          {(
            [
              ["file", "INCLUDE video file (same input for every configuration)"],
              ["camera", "Camera (640×480 requested)"],
              ["both", "Both: the file first, then the camera"],
            ] as const
          ).map(([m, text]) => (
            <label key={m} className="flex items-center gap-2">
              <input type="radio" name="source" checked={mode === m} onChange={() => setMode(m)} />
              {text}
            </label>
          ))}
          {mode !== "camera" && (
            <input
              type="file"
              accept="video/*,.mov,.MOV"
              aria-label="Video file"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          )}
          <p className="text-sm text-zinc-500">
            The file is played locally in a loop and never leaves this device. Each configuration
            starts again from its first frame.
          </p>
        </fieldset>
        <fieldset className="flex flex-col gap-2" disabled={running}>
          <legend className="font-medium">Configurations</legend>
          {CONFIG_IDS.map((id) => (
            <label key={id} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={selected.includes(id)}
                onChange={(e) =>
                  setSelected((s) =>
                    e.target.checked
                      ? CONFIG_IDS.filter((c) => c === id || s.includes(c))
                      : s.filter((c) => c !== id),
                  )
                }
              />
              {LABELS[id]}
            </label>
          ))}
          <div className="grid grid-cols-2 gap-2 pt-2">
            {(
              [
                ["Seconds each", seconds, setSeconds, 1],
                ["Warm-up", warmup, setWarmup, 0],
                ["Rounds", rounds, setRounds, 1],
                ["Order seed (0 = random)", seed, setSeed, 0],
              ] as const
            ).map(([label, value, set, min]) => (
              <label key={label} className="flex items-center justify-between gap-2">
                {label}
                <input
                  type="number"
                  min={min}
                  className="w-28 rounded border border-zinc-400 bg-transparent px-1"
                  value={value}
                  onChange={(e) => set(Number(e.target.value))}
                />
              </label>
            ))}
          </div>
          <p className="text-sm text-zinc-500">
            Speed rule (ADR 002): on the M3 Pro benchmark machine ≥ {BENCH_MACHINE_BAR.minFps} fps with
            p95 ≤ {BENCH_MACHINE_BAR.maxP95Ms} ms; any mid-range device ≥ {MID_RANGE_FLOOR.minFps} fps with p95 ≤{" "}
            {MID_RANGE_FLOOR.maxP95Ms} ms. Judged on the faster delegate, in camera mode only (file mode is the
            same-input quality check).
          </p>
          <p className="text-sm text-zinc-500">
            Each round runs every configuration once, in an order shuffled from the seed. Warm-up
            lasts at least {MIN_WARMUP_FRAMES} frames and is excluded from the timings.
          </p>
        </fieldset>
      </section>

      <div className="flex flex-wrap items-center gap-4">
        <button
          type="button"
          onClick={() => void start()}
          disabled={running || selected.length === 0}
          className="rounded-md bg-zinc-900 px-4 py-2 font-medium text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
        >
          {running ? "Running…" : "Run benchmark"}
        </button>
        <p role="status" aria-live="polite" className="text-zinc-600 dark:text-zinc-300">
          {status}
        </p>
      </div>
      <p className="text-sm text-zinc-500">
        About {Math.round(estSeconds / 60)} min. Keep this tab in front and the laptop plugged in.
        Background tabs are throttled.
      </p>
      {mode !== "file" && (
        <p className="text-sm font-medium text-amber-700 dark:text-amber-400">
          Camera runs: stand where the camera sees you from the waist up, and keep both hands
          raised and moving (sign any words). The dots show which hands each configuration finds.
          If one never finds your hands, note how you were standing and the lighting.
        </p>
      )}

      {/* Preview (and overlay) are mirrored for the camera only; landmarks always come from the unmirrored frames. */}
      <div className={`relative w-full max-w-md ${mirrored ? "-scale-x-100" : ""}`}>
        <video ref={videoRef} playsInline muted className="w-full rounded-lg bg-black" />
        <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
      </div>

      {report && (
        <section className="flex flex-col gap-4">
          <h2 className="text-xl font-semibold">Results</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-zinc-400">
                  <th className="py-1 pr-3">Source</th>
                  <th className="pr-3">Configuration</th>
                  <th className="pr-3">Round</th>
                  <th className="pr-3">ms/frame p50</th>
                  <th className="pr-3">p95</th>
                  <th className="pr-3">first frames ms (max)</th>
                  <th className="pr-3">processed / source fps</th>
                  <th className="pr-3">frames w/ hand</th>
                  <th className="pr-3">init ms</th>
                  <th className="pr-3">
                    ≥ {BENCH_MACHINE_BAR.minFps} fps, p95 ≤ {BENCH_MACHINE_BAR.maxP95Ms} ms?
                  </th>
                  <th>
                    ≥ {MID_RANGE_FLOOR.minFps} fps, p95 ≤ {MID_RANGE_FLOOR.maxP95Ms} ms?
                  </th>
                </tr>
              </thead>
              <tbody>
                {report.results.map((r, i) => (
                  <tr key={i} className="border-b border-zinc-200 dark:border-zinc-800">
                    <td className="py-1 pr-3">{r.source}</td>
                    <td className="pr-3">{LABELS[r.id]}</td>
                    <td className="pr-3">{r.round}</td>
                    {r.ok ? (
                      <>
                        <td className="pr-3">{fmt(r.msP50)}</td>
                        <td className="pr-3">{fmt(r.msP95)}</td>
                        <td className="pr-3">{fmt(Math.max(...(r.firstFramesMs ?? [NaN])), 0)}</td>
                        <td className="pr-3">
                          {fmt(r.processedFps)} / {fmt(r.sourceFps)}
                        </td>
                        <td className="pr-3">{fmt((r.handRate ?? 0) * 100, 0)}%</td>
                        <td className="pr-3">{fmt(r.initMs, 0)}</td>
                        <td className="pr-3">{r.meetsBenchBar ? "yes" : "no"}</td>
                        <td>{r.meetsFloor ? "yes" : "no"}</td>
                      </>
                    ) : (
                      <td colSpan={8} className="text-red-600">
                        failed: {r.error}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-zinc-500">Video</dt>
            <dd>
              {Object.entries(report.videoSizes)
                .map(([k, [w, h]]) => `${k} ${w}×${h}`)
                .join(" · ")}
            </dd>
            <dt className="text-zinc-500">Order (seed {report.seed})</dt>
            <dd>{report.order.join(", ")}</dd>
            <dt className="text-zinc-500">Cross-origin isolated</dt>
            <dd>{String(report.device.crossOriginIsolated)}</dd>
            <dt className="text-zinc-500">GPU (WebGL)</dt>
            <dd>{report.device.webglRenderer ?? "unknown"}</dd>
            <dt className="text-zinc-500">WebGPU adapter</dt>
            <dd>{report.device.webgpuAdapter ?? "not available"}</dd>
            <dt className="text-zinc-500">CPU threads</dt>
            <dd>{report.device.hardwareConcurrency}</dd>
            <dt className="text-zinc-500">Models</dt>
            <dd>
              {report.models
                .map((m) => `${m.name} ${m.sha256.slice(0, 12)}… ${m.verified ? "✓" : "✗"}`)
                .join(" · ")}
            </dd>
          </dl>
          <div className="flex gap-3">
            <button type="button" onClick={download} className="rounded-md border border-zinc-400 px-3 py-1">
              Save results as JSON
            </button>
            <button
              type="button"
              onClick={() => void navigator.clipboard.writeText(JSON.stringify(report, null, 2))}
              className="rounded-md border border-zinc-400 px-3 py-1"
            >
              Copy JSON
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
