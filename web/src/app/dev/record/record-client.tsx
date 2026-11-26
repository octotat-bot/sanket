"use client";

import type { HolisticLandmarker } from "@mediapipe/tasks-vision";
import { useEffect, useRef, useState } from "react";
import { sha256Hex } from "@/lib/bench/stats";
import { frameForMediaPipe } from "@/lib/frames/downscale";
import { loadFileset, loadSpec, loadVerifiedModels } from "@/lib/mediapipe/assets";
import { createHolistic, type StoredFrame, toStoredFrame } from "@/lib/mediapipe/holistic";
import { REFERENCE_CREDIT } from "@/lib/record/reference";
import {
  appendFrame,
  pickWords,
  buildPlan,
  buildSessionFile,
  CONSENT_VERSION,
  type ClipQuality,
  clipQuality,
  type Fluency,
  participantId,
  type Prompt,
  type RecordedClip,
  type SessionMeta,
  type Word,
} from "@/lib/record/session";

/** Shown to the participant and hashed into the session file (consent.text_sha256). */
export const CONSENT_TEXT = [
  "You must be 18 or older to take part.",
  "This tool records keypoints only: numbers describing where your body and hands are in each video frame. No video, images or audio are recorded or stored.",
  "Your keypoints are used only to measure how accurately the Sanket ISL word recognizer works on real webcams, and for nothing else.",
  "Your keypoint files are kept private by the project and are never published or shared. Only overall accuracy numbers, combined across all participants, may appear in the project's results.",
  "No name or contact details are recorded. After you agree, you get a random participant code. Write it down: to withdraw later, send the code to Mukund and every recording with that code is deleted.",
  "Taking part is voluntary. You can stop at any time, and anything not saved is discarded.",
].join("\n");

type Phase = "consent" | "setup" | "session";
type ClipState = "idle" | "countdown" | "recording" | "review";

interface Live {
  processed: number;
  shoulders: boolean;
  hands: boolean;
}

const WARMUP_FRAMES = 30;

function randomId(prefix: string): string {
  return prefix + participantId(crypto.getRandomValues(new Uint8Array(6))).slice(2);
}

export default function RecordClient() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HolisticLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const runningRef = useRef(false);
  const lastTsRef = useRef(0);
  const clipRef = useRef<RecordedClip | null>(null); // the clip being recorded
  const pendingRef = useRef<RecordedClip | null>(null); // the last clip, awaiting accept/retake
  const clipStartRef = useRef(0);
  const metaRef = useRef<SessionMeta | null>(null);
  const liveRef = useRef<Live>({ processed: 0, shoulders: false, hands: false });

  const [phase, setPhase] = useState<Phase>("consent");
  const [agreed, setAgreed] = useState(false);
  const [adult, setAdult] = useState(false);
  const [fluency, setFluency] = useState<Fluency | null>(null);
  const [referenceOk, setReferenceOk] = useState<boolean | null>(null);
  const [takes, setTakes] = useState(2);
  const [noSign, setNoSign] = useState(10);
  const [seconds, setSeconds] = useState(4);
  const [nWords, setNWords] = useState(50);
  const [mirrorRef, setMirrorRef] = useState(true);
  const [status, setStatus] = useState("");
  const [live, setLive] = useState<Live>({ processed: 0, shoulders: false, hands: false });
  const [plan, setPlan] = useState<Prompt[]>([]);
  const [index, setIndex] = useState(0);
  const [clipState, setClipState] = useState<ClipState>("idle");
  const [countdown, setCountdown] = useState(0);
  const [quality, setQuality] = useState<ClipQuality | null>(null);
  const [clips, setClips] = useState<RecordedClip[]>([]);
  const [saved, setSaved] = useState(0);
  const [pid, setPid] = useState("");

  // Warn before leaving with clips that haven't been saved to a file.
  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (clips.length > saved) e.preventDefault();
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [clips.length, saved]);

  // Stop the camera and MediaPipe when leaving the page.
  useEffect(
    () => () => {
      runningRef.current = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
    },
    [],
  );

  // Test hook (e2e): the exact frame path the recorder uses, to check frames are never mirrored.
  useEffect(() => {
    (window as unknown as { __sanketFrameForMediaPipe?: typeof frameForMediaPipe }).__sanketFrameForMediaPipe =
      frameForMediaPipe;
  }, []);

  function drawOverlay(f: StoredFrame | null) {
    const c = overlayRef.current;
    const v = videoRef.current;
    if (!c || !v) return;
    c.width = v.clientWidth;
    c.height = v.clientHeight;
    const g = c.getContext("2d");
    if (!g) return;
    g.clearRect(0, 0, c.width, c.height);
    if (!f) return;
    const dot = (p: number[], color: string) => {
      g.fillStyle = color;
      g.fillRect(p[0] * c.width - 2, p[1] * c.height - 2, 4, 4);
    };
    if (f.pose_present) [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16].forEach((i) => dot(f.pose[i], "#f59e0b"));
    if (f.left_present) f.left.forEach((p) => dot(p, "#22d3ee"));
    if (f.right_present) f.right.forEach((p) => dot(p, "#a78bfa"));
  }

  function processLoop(video: HTMLVideoElement) {
    if (!runningRef.current) return;
    video.requestVideoFrameCallback(async (now) => {
      const lm = landmarkerRef.current;
      if (!runningRef.current || !lm) return;
      // Unmirrored camera frame → 480 rows (ADR 007) → Holistic.
      const { bitmap, width, height } = await frameForMediaPipe(video, video.videoWidth, video.videoHeight);
      const ts = (lastTsRef.current = Math.max(lastTsRef.current + 0.001, now));
      let frame: StoredFrame | null = null;
      lm.detectForVideo(bitmap, ts, (r) => {
        frame = toStoredFrame(r);
      });
      bitmap.close();
      const f = frame as StoredFrame | null;
      drawOverlay(f);
      const clip = clipRef.current;
      if (clip && f) {
        if (clip.t_ms.length === 0) {
          clipStartRef.current = ts;
          clip.frame_size = [width, height];
          clip.camera_size = [video.videoWidth, video.videoHeight];
        }
        appendFrame(clip, ts - clipStartRef.current, f);
      }
      const l = liveRef.current;
      l.processed++;
      l.shoulders = !!f && f.pose_present && f.pose[11][3] >= 0.5 && f.pose[12][3] >= 0.5;
      l.hands = !!f && (f.left_present || f.right_present);
      if (l.processed % 10 === 0 || l.processed === WARMUP_FRAMES) setLive({ ...l });
      processLoop(video);
    });
  }

  async function begin() {
    const video = videoRef.current;
    if (!video) return;
    const consentAt = new Date().toISOString();
    const id = participantId(crypto.getRandomValues(new Uint8Array(6)));
    setPid(id);
    setPhase("setup");
    try {
      setStatus("Loading and verifying the pinned models (sha256)…");
      const spec = await loadSpec();
      const models = await loadVerifiedModels(spec);
      const bad = Object.values(models).filter((m) => !m.ok);
      if (bad.length) throw new Error(`model sha256 mismatch: ${bad.map((m) => m.name).join(", ")}`);
      const fileset = await loadFileset();
      const { landmarker, delegate } = await createHolistic(fileset, models, "GPU");
      landmarkerRef.current = landmarker;
      setStatus("Starting the camera…");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();
      metaRef.current = {
        session_id: randomId("S-"),
        participant_id: id,
        participant: { adult_confirmed: true, fluent_isl_signer: fluency ?? "not_stated" },
        reference_clips: "none", // set from an actual check when the session starts
        consent: {
          version: CONSENT_VERSION,
          accepted_at: consentAt,
          text_sha256: await sha256Hex(new TextEncoder().encode(CONSENT_TEXT)),
        },
        created_at: new Date().toISOString(),
        app: { commit: process.env.NEXT_PUBLIC_GIT_SHA ?? null, user_agent: navigator.userAgent },
        spec_version: (spec as unknown as { spec_version: string }).spec_version,
        mediapipe_version: spec.extractor.mediapipe_version,
        model_sha256: { holistic_landmarker: models.holistic_landmarker.actualSha256 },
        delegate,
        extract_height: 480,
      };
      runningRef.current = true;
      processLoop(video);
      setStatus(`Running on ${delegate}. Preparing (the first GPU frame can take several seconds)…`);
    } catch (e) {
      setStatus(`Error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function startSession() {
    const vocab = (await (await fetch("/spec/vocab_include50.json")).json()).words as Word[];
    // Reference clips exist only on the dev server, after `make reference-clips` (never deployed).
    const probe = await fetch(`/dev/reference/${vocab[0].label}`, { method: "HEAD" }).catch(() => null);
    const available = !!probe && probe.ok;
    setReferenceOk(available);
    if (metaRef.current) metaRef.current.reference_clips = available ? "include50_train_local" : "none";
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    if (metaRef.current) metaRef.current.reference_mirrored = mirrorRef;
    setPlan(buildPlan(pickWords(vocab, nWords, seed), takes, noSign, seed));
    setIndex(0);
    setPhase("session");
  }

  function record() {
    const p = plan[index];
    if (!p || clipState === "countdown" || clipState === "recording") return;
    setQuality(null);
    setClipState("countdown");
    let n = 3;
    setCountdown(n);
    const tick = () => {
      n -= 1;
      if (n > 0) {
        setCountdown(n);
        setTimeout(tick, 1000);
        return;
      }
      clipRef.current = {
        clip_id: randomId("C-"),
        kind: p.kind,
        label: p.label,
        label_id: p.labelId,
        take: p.take,
        prompt: p.display,
        started_at: new Date().toISOString(),
        frame_size: [0, 0],
        camera_size: [0, 0],
        t_ms: [],
        pose: [],
        left: [],
        right: [],
        pose_present: [],
        left_present: [],
        right_present: [],
      };
      setClipState("recording");
      setTimeout(() => {
        const c = clipRef.current;
        clipRef.current = null;
        if (c) {
          setQuality(clipQuality(c));
          pendingRef.current = c;
        }
        setClipState("review");
      }, seconds * 1000);
    };
    setTimeout(tick, 1000);
  }

  function accept() {
    const c = pendingRef.current;
    if (!c || clipState !== "review") return;
    setClips((cs) => [...cs, c]);
    pendingRef.current = null;
    setClipState("idle");
    setQuality(null);
    setIndex((i) => i + 1);
  }

  function retake() {
    pendingRef.current = null;
    setQuality(null);
    setClipState("idle");
  }

  function skip() {
    if (clipState === "countdown" || clipState === "recording") return;
    pendingRef.current = null;
    setQuality(null);
    setClipState("idle");
    setIndex((i) => i + 1);
  }

  function save() {
    const meta = metaRef.current;
    if (!meta) return;
    const file = buildSessionFile(meta, clips);
    const blob = new Blob([JSON.stringify(file)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `sanket-rec-${meta.participant_id}-${meta.session_id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    setSaved(clips.length);
    (window as unknown as { __recordSession?: unknown }).__recordSession = file;
  }

  // Keyboard: Space = record, Enter = accept, R = retake, S = skip.
  useEffect(() => {
    if (phase !== "session") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.code === "Space") {
        e.preventDefault();
        record();
      } else if (e.key === "Enter") accept();
      else if (e.key.toLowerCase() === "r") retake();
      else if (e.key.toLowerCase() === "s") skip();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const ready = live.processed >= WARMUP_FRAMES;
  const prompt = plan[index];
  const done = phase === "session" && index >= plan.length;

  return (
    <div className="flex flex-col gap-6">
      {phase === "consent" && (
        <section aria-labelledby="consent-h" className="flex flex-col gap-4 rounded-lg border border-zinc-300 p-5 dark:border-zinc-700">
          <h2 id="consent-h" className="text-xl font-semibold">
            Before you start
          </h2>
          <ul className="list-disc space-y-2 pl-5 text-zinc-700 dark:text-zinc-200">
            {CONSENT_TEXT.split("\n").map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
          <p className="text-sm text-zinc-500">
            Also sign the paper consent form (tools/record/CONSENT.md). It does not contain your
            participant code, so your recordings can&apos;t be linked to your name.
          </p>
          <label className="flex items-center gap-2 font-medium">
            <input type="checkbox" checked={adult} onChange={(e) => setAdult(e.target.checked)} />
            I am 18 or older
          </label>
          <label className="flex items-center gap-2 font-medium">
            <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} />
            I have read this and agree to take part
          </label>
          <fieldset className="flex flex-col gap-1">
            <legend className="font-medium">Are you a fluent ISL signer?</legend>
            {(
              [
                ["yes", "Yes"],
                ["no", "No"],
                ["not_stated", "Prefer not to say"],
              ] as const
            ).map(([value, label]) => (
              <label key={value} className="flex items-center gap-2">
                <input type="radio" name="fluency" checked={fluency === value} onChange={() => setFluency(value)} />
                {label}
              </label>
            ))}
          </fieldset>
          <div className="flex flex-wrap gap-4 text-sm">
            {(
              [
                ["Words", nWords, setNWords, 1, 50],
                ["Takes per word", takes, setTakes, 1, 5],
                ["No-sign clips", noSign, setNoSign, 0, 30],
                ["Seconds per clip", seconds, setSeconds, 2, 6],
              ] as const
            ).map(([label, value, set, min, max]) => (
              <label key={label} className="flex items-center gap-2">
                {label}
                <input
                  type="number"
                  min={min}
                  max={max}
                  value={value}
                  onChange={(e) => set(Number(e.target.value))}
                  className="w-16 rounded border border-zinc-400 bg-transparent px-1"
                />
              </label>
            ))}
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={mirrorRef} onChange={(e) => setMirrorRef(e.target.checked)} />
            Show reference clips mirrored (copy them like a mirror)
          </label>
          <button
            type="button"
            disabled={!agreed || !adult || fluency === null}
            onClick={() => void begin()}
            className="self-start rounded-md bg-zinc-900 px-4 py-2 font-medium text-white disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
          >
            Agree and start the camera
          </button>
        </section>
      )}

      {/* Preview and overlay are mirrored for the participant only; keypoints come from unmirrored frames. */}
      <div className={`relative w-full max-w-xl -scale-x-100 ${phase === "consent" ? "hidden" : ""}`}>
        <video ref={videoRef} playsInline muted className="w-full rounded-lg bg-black" />
        <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
      </div>

      {phase !== "consent" && (
        <p role="status" aria-live="polite" className="text-sm text-zinc-600 dark:text-zinc-300">
          Participant code: <strong>{pid}</strong> · {status} · {ready ? "ready" : "preparing"} ·
          shoulders {live.shoulders ? "✓" : "✗"} · hands {live.hands ? "✓" : "✗"}
        </p>
      )}

      {phase === "setup" && (
        <section className="flex flex-col gap-3">
          <p>
            Sit so the camera sees your head, shoulders and both hands, with your elbows in view. Good,
            even light from the front helps.
          </p>
          <button
            type="button"
            disabled={!ready}
            onClick={() => void startSession()}
            className="self-start rounded-md bg-zinc-900 px-4 py-2 font-medium text-white disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
          >
            Start the session
          </button>
        </section>
      )}

      {phase === "session" && !done && prompt && (
        <section aria-live="polite" className="flex flex-col gap-3 rounded-lg border border-zinc-300 p-5 dark:border-zinc-700">
          <p className="text-sm text-zinc-500">
            Clip {index + 1} of {plan.length} · {clips.length} accepted · {clips.length - saved} unsaved
          </p>
          {prompt.kind === "sign" ? (
            <p className="text-4xl font-semibold" data-testid="prompt">
              Sign: {prompt.display} <span className="text-lg text-zinc-500">({prompt.category}, take {prompt.take})</span>
            </p>
          ) : (
            <p className="text-2xl font-semibold" data-testid="prompt">
              No sign: {prompt.display}
            </p>
          )}
          {prompt.kind === "sign" && referenceOk && (
            <figure className="flex flex-col gap-1">
              {/* Dev only: INCLUDE train clip from the local cache (never in the deployed app). */}
              <video
                key={prompt.label}
                src={`/dev/reference/${prompt.label}`}
                autoPlay
                loop
                muted
                playsInline
                className={`w-full max-w-sm rounded-md bg-black ${mirrorRef ? "-scale-x-100" : ""}`}
                aria-label={`INCLUDE reference clip for ${prompt.display}${mirrorRef ? " (mirrored)" : ""}`}
              />
              <figcaption className="text-xs text-zinc-500">{REFERENCE_CREDIT}</figcaption>
            </figure>
          )}
          {prompt.kind === "sign" && referenceOk === false && (
            <p className="text-sm text-zinc-500">
              No reference clip: they exist only on the dev server after <code>make reference-clips</code>.
            </p>
          )}
          {clipState === "countdown" && <p className="text-5xl font-bold">{countdown}</p>}
          {clipState === "recording" && <p className="text-2xl font-bold text-red-600">● Recording</p>}
          {clipState === "review" && quality && (
            <div data-testid="quality">
              <p>
                {quality.frames} frames · {quality.fps.toFixed(0)} fps · hands in {(quality.handRate * 100).toFixed(0)}% ·
                shoulders in {(quality.shoulderRate * 100).toFixed(0)}%
              </p>
              {quality.warnings.map((w) => (
                <p key={w} className="text-amber-700 dark:text-amber-400">
                  ⚠ {w}
                </p>
              ))}
            </div>
          )}
          <div className="flex flex-wrap gap-3">
            <button type="button" onClick={record} disabled={clipState !== "idle"} className="rounded-md bg-red-600 px-4 py-2 font-medium text-white disabled:opacity-40">
              Record (Space)
            </button>
            <button type="button" onClick={accept} disabled={clipState !== "review"} className="rounded-md border border-zinc-400 px-4 py-2 disabled:opacity-40">
              Accept (Enter)
            </button>
            <button type="button" onClick={retake} disabled={clipState !== "review"} className="rounded-md border border-zinc-400 px-4 py-2 disabled:opacity-40">
              Retake (R)
            </button>
            <button type="button" onClick={skip} disabled={clipState === "countdown" || clipState === "recording"} className="rounded-md border border-zinc-400 px-4 py-2 disabled:opacity-40">
              Skip (S)
            </button>
          </div>
        </section>
      )}

      {phase === "session" && done && <p className="text-xl font-semibold">All prompts done. Thank you! Save the session file below.</p>}

      {phase === "session" && (
        <button
          type="button"
          onClick={save}
          disabled={clips.length === 0}
          className="self-start rounded-md border border-zinc-400 px-4 py-2 disabled:opacity-40"
        >
          Save session file ({clips.length} clips)
        </button>
      )}
    </div>
  );
}
