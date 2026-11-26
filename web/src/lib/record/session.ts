// Phase 6 recording sessions: the plan of prompts, anonymous ids, per-clip quality, and the
// session file format (spec/recording_v1.json). Pure functions, unit-tested in
// tests/unit/record-session.test.ts; the page is src/app/dev/record/.
import { mulberry32 } from "../bench/stats";
import type { StoredFrame } from "../mediapipe/holistic";

export const RECORDING_FORMAT = "sanket-recording-v1";
export const CONSENT_VERSION = "1";

/** What participants are asked to do in "no sign" clips: everyday movement that is not signing. */
export const NO_SIGN_ACTIVITIES = [
  "Rest your hands on the desk and look at the screen",
  "Scratch your nose or cheek",
  "Adjust your hair or glasses",
  "Talk out loud as if on a phone call (no signing)",
  "Type a few words on the keyboard",
  "Pick up and put down a cup or phone",
  "Stretch your arms and then rest them",
  "Cross your arms, then uncross them",
];

export interface Word {
  id: number;
  label: string;
  display: string;
  category: string;
}

export interface Prompt {
  kind: "sign" | "no_sign";
  label: string;
  labelId: number; // -1 for no_sign
  display: string;
  category: string;
  take: number;
}

/** Anonymous participant id: "P-" + 6 base32 characters. Never derived from a name. */
export function participantId(bytes: Uint8Array): string {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
  return "P-" + Array.from(bytes.slice(0, 6), (b) => alphabet[b % 32]).join("");
}

/** Every word `takes` times plus `noSign` no-sign clips, shuffled (seeded, so a session can be re-planned). */
export function buildPlan(words: Word[], takes: number, noSign: number, seed: number): Prompt[] {
  const plan: Prompt[] = [];
  for (const w of words) {
    for (let t = 1; t <= takes; t++) {
      plan.push({ kind: "sign", label: w.label, labelId: w.id, display: w.display, category: w.category, take: t });
    }
  }
  for (let i = 0; i < noSign; i++) {
    const activity = NO_SIGN_ACTIVITIES[i % NO_SIGN_ACTIVITIES.length];
    plan.push({ kind: "no_sign", label: "no_sign", labelId: -1, display: activity, category: "no sign", take: i + 1 });
  }
  const rand = mulberry32(seed);
  for (let i = plan.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [plan[i], plan[j]] = [plan[j], plan[i]];
  }
  return plan;
}

export interface RecordedClip {
  clip_id: string;
  kind: "sign" | "no_sign";
  label: string;
  label_id: number;
  take: number;
  prompt: string;
  started_at: string;
  frame_size: [number, number]; // as processed by MediaPipe (480 rows)
  camera_size: [number, number];
  t_ms: number[];
  pose: number[][][];
  left: number[][][];
  right: number[][][];
  pose_present: boolean[];
  left_present: boolean[];
  right_present: boolean[];
}

export function appendFrame(clip: RecordedClip, tMs: number, f: StoredFrame): void {
  clip.t_ms.push(tMs);
  clip.pose.push(f.pose);
  clip.left.push(f.left);
  clip.right.push(f.right);
  clip.pose_present.push(f.pose_present);
  clip.left_present.push(f.left_present);
  clip.right_present.push(f.right_present);
}

export interface ClipQuality {
  frames: number;
  durationMs: number;
  fps: number;
  handRate: number;
  shoulderRate: number; // frames where both shoulders are visible (≥ 0.5): the body reference
  warnings: string[];
}

export function clipQuality(c: RecordedClip): ClipQuality {
  const n = c.t_ms.length;
  const durationMs = n > 1 ? c.t_ms[n - 1] - c.t_ms[0] : 0;
  const fps = durationMs > 0 ? ((n - 1) * 1000) / durationMs : 0;
  const handRate = n ? c.left_present.filter((v, i) => v || c.right_present[i]).length / n : 0;
  const shoulderRate = n
    ? c.pose.filter((p, i) => c.pose_present[i] && p[11][3] >= 0.5 && p[12][3] >= 0.5).length / n
    : 0;
  const warnings: string[] = [];
  if (n < 10) warnings.push("very few frames");
  if (fps > 0 && fps < 12) warnings.push(`low frame rate (${fps.toFixed(0)} fps)`);
  if (shoulderRate < 0.8) warnings.push("shoulders not visible in many frames: sit a little further back");
  if (c.kind === "sign" && handRate < 0.5) warnings.push("hands not found in most frames: check lighting and framing");
  return { frames: n, durationMs, fps, handRate, shoulderRate, warnings };
}

const round = (x: number, d: number) => Math.round(x * 10 ** d) / 10 ** d;

export type Fluency = "yes" | "no" | "not_stated";

export interface SessionMeta {
  session_id: string;
  participant_id: string;
  /** 18+ only; fluent_isl_signer splits fluent signers from non-native imitators in reports. */
  participant: { adult_confirmed: true; fluent_isl_signer: Fluency };
  /** "include50_train_local": INCLUDE train clips were shown from the local cache (dev server). */
  reference_clips: "include50_train_local" | "none";
  consent: { version: string; accepted_at: string; text_sha256: string };
  created_at: string;
  app: { commit: string | null; user_agent: string };
  spec_version: string;
  mediapipe_version: string;
  model_sha256: Record<string, string>;
  delegate: string;
  extract_height: number;
}

/** The session file (spec/recording_v1.json): keypoints rounded to `decimals`, nothing else. */
export function buildSessionFile(meta: SessionMeta, clips: RecordedClip[], decimals = 5) {
  const r3 = (a: number[][][]) => a.map((f) => f.map((p) => p.map((v) => round(v, decimals))));
  return {
    format: RECORDING_FORMAT,
    ...meta,
    clips: clips.map((c) => ({
      ...c,
      t_ms: c.t_ms.map((t) => round(t, 3)),
      pose: r3(c.pose),
      left: r3(c.left),
      right: r3(c.right),
    })),
  };
}

export interface RecordingSpec {
  format: string;
  session_required: string[];
  consent_required: string[];
  clip_required: string[];
  kinds: string[];
  shapes: Record<string, [number, number]>;
  participant_id_pattern: string;
  forbidden_keys: string[];
  participant_required: Record<string, unknown[]>;
  reference_clips: string[];
}

/** Errors (empty = valid). Mirrors ml/src/sanket/data/recorded.py `validate_session`. */
export function validateSession(s: Record<string, unknown>, spec: RecordingSpec): string[] {
  const errs: string[] = [];
  const json = JSON.stringify(s);
  for (const k of spec.forbidden_keys) if (json.includes(`"${k}":`)) errs.push(`forbidden key ${k}`);
  if (s.format !== spec.format) errs.push(`format ${String(s.format)}`);
  for (const k of spec.session_required) if (!(k in s)) errs.push(`missing ${k}`);
  if (!new RegExp(spec.participant_id_pattern).test(String(s.participant_id))) errs.push("participant_id");
  const participant = (s.participant ?? {}) as Record<string, unknown>;
  for (const [k, allowed] of Object.entries(spec.participant_required))
    if (!allowed.includes(participant[k])) errs.push(`participant.${k}`);
  if (!spec.reference_clips.includes(String(s.reference_clips))) errs.push("reference_clips");
  const consent = (s.consent ?? {}) as Record<string, unknown>;
  for (const k of spec.consent_required) if (!(k in consent)) errs.push(`consent.${k}`);
  const clips = (s.clips ?? []) as Record<string, unknown>[];
  clips.forEach((c, i) => {
    for (const k of spec.clip_required) if (!(k in c)) errs.push(`clip ${i}: missing ${k}`);
    if (!spec.kinds.includes(String(c.kind))) errs.push(`clip ${i}: kind`);
    const t = (c.t_ms ?? []) as number[];
    if (t.some((v, j) => j > 0 && v <= t[j - 1])) errs.push(`clip ${i}: t_ms not increasing`);
    for (const [k, [a, b]] of Object.entries(spec.shapes)) {
      const arr = (c[k] ?? []) as number[][][];
      if (arr.length !== t.length || arr.some((f) => f.length !== a || f.some((p) => p.length !== b)))
        errs.push(`clip ${i}: ${k} shape`);
    }
  });
  return errs;
}
