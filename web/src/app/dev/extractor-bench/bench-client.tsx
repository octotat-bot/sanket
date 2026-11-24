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
  runConfig,
  TARGET_FPS,
} from "@/lib/bench/extractor-bench";

type Source = "camera" | "file";

interface Report {
  createdAt: string;
  mediapipeVersion: string;
  source: Source;
  videoSize: [number, number];
  warmupSeconds: number;
  seconds: number;
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

function fmt(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined || Number.isNaN(x) ? "–" : x.toFixed(digits);
}

export default function BenchClient() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [source, setSource] = useState<Source>("camera");
  const [file, setFile] = useState<File | null>(null);
  const [seconds, setSeconds] = useState(15);
  const [warmup, setWarmup] = useState(3);
  const [selected, setSelected] = useState<ConfigId[]>([...CONFIG_IDS]);
  const [status, setStatus] = useState("Idle.");
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<Report | null>(null);

  async function attachSource(video: HTMLVideoElement): Promise<() => void> {
    if (source === "camera") {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      return () => stream.getTracks().forEach((t) => t.stop());
    }
    if (!file) throw new Error("Choose a video file first.");
    const url = URL.createObjectURL(file);
    video.src = url;
    video.loop = true;
    video.muted = true;
    await video.play();
    return () => {
      video.pause();
      URL.revokeObjectURL(url);
    };
  }

  async function start() {
    const video = videoRef.current;
    if (!video || running) return;
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
      setStatus(source === "camera" ? "Starting camera…" : "Starting video…");
      detach = await attachSource(video);
      const results: ConfigResult[] = [];
      for (const id of selected) {
        results.push(
          await runConfig(id, video, fileset, models, {
            warmupSeconds: warmup,
            seconds,
            onProgress: (m) => setStatus(m),
          }),
        );
      }
      const r: Report = {
        createdAt: new Date().toISOString(),
        mediapipeVersion: spec.extractor.mediapipe_version,
        source,
        videoSize: [video.videoWidth, video.videoHeight],
        warmupSeconds: warmup,
        seconds,
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

  return (
    <div className="flex flex-col gap-6">
      <section className="grid gap-4 rounded-lg border border-zinc-300 p-4 dark:border-zinc-700 sm:grid-cols-2">
        <fieldset className="flex flex-col gap-2" disabled={running}>
          <legend className="font-medium">Video source</legend>
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="source"
              checked={source === "camera"}
              onChange={() => setSource("camera")}
            />
            Camera (640×480 requested)
          </label>
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="source"
              checked={source === "file"}
              onChange={() => setSource("file")}
            />
            Local video file (looped, stays on this device)
          </label>
          {source === "file" && (
            <input
              type="file"
              accept="video/*"
              aria-label="Video file"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          )}
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
                    e.target.checked ? CONFIG_IDS.filter((c) => c === id || s.includes(c)) : s.filter((c) => c !== id),
                  )
                }
              />
              {LABELS[id]}
            </label>
          ))}
          <div className="flex gap-4 pt-2">
            <label className="flex items-center gap-2">
              Seconds each
              <input
                type="number"
                min={1}
                className="w-16 rounded border border-zinc-400 bg-transparent px-1"
                value={seconds}
                onChange={(e) => setSeconds(Number(e.target.value))}
              />
            </label>
            <label className="flex items-center gap-2">
              Warm-up
              <input
                type="number"
                min={0}
                step={0.5}
                className="w-16 rounded border border-zinc-400 bg-transparent px-1"
                value={warmup}
                onChange={(e) => setWarmup(Number(e.target.value))}
              />
            </label>
          </div>
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
        Keep this tab in front and the laptop plugged in while it runs (about{" "}
        {selected.length * (seconds + warmup) + 5} s). Background tabs are throttled.
      </p>

      {/* Preview is mirrored for the user only; landmarks are computed on the unmirrored frames. */}
      <video
        ref={videoRef}
        playsInline
        muted
        className={`w-full max-w-md rounded-lg bg-black ${source === "camera" ? "-scale-x-100" : ""}`}
      />

      {report && (
        <section className="flex flex-col gap-4">
          <h2 className="text-xl font-semibold">Results</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-zinc-400">
                  <th className="py-1 pr-3">Configuration</th>
                  <th className="pr-3">ms/frame p50</th>
                  <th className="pr-3">p95</th>
                  <th className="pr-3">headroom fps (1000/p95)</th>
                  <th className="pr-3">processed fps</th>
                  <th className="pr-3">source fps</th>
                  <th className="pr-3">frames w/ hand</th>
                  <th className="pr-3">init ms</th>
                  <th>≥ {TARGET_FPS} fps?</th>
                </tr>
              </thead>
              <tbody>
                {report.results.map((r) => (
                  <tr key={r.id} className="border-b border-zinc-200 dark:border-zinc-800">
                    <td className="py-1 pr-3">{LABELS[r.id]}</td>
                    {r.ok ? (
                      <>
                        <td className="pr-3">{fmt(r.msP50)}</td>
                        <td className="pr-3">{fmt(r.msP95)}</td>
                        <td className="pr-3">{fmt(r.headroomFps, 0)}</td>
                        <td className="pr-3">{fmt(r.processedFps)}</td>
                        <td className="pr-3">{fmt(r.sourceFps)}</td>
                        <td className="pr-3">{fmt((r.handRate ?? 0) * 100, 0)}%</td>
                        <td className="pr-3">{fmt(r.initMs, 0)}</td>
                        <td>{r.sustainsTarget ? "yes" : "no"}</td>
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
              {report.source}, {report.videoSize[0]}×{report.videoSize[1]}
            </dd>
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
            <button
              type="button"
              onClick={download}
              className="rounded-md border border-zinc-400 px-3 py-1"
            >
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
