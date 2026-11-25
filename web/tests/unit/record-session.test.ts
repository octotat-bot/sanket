import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { StoredFrame } from "@/lib/mediapipe/holistic";
import {
  appendFrame,
  buildPlan,
  buildSessionFile,
  clipQuality,
  participantId,
  type RecordedClip,
  type RecordingSpec,
  type SessionMeta,
  validateSession,
  type Word,
} from "@/lib/record/session";

const root = resolve(__dirname, "../../..");
const recSpec = JSON.parse(readFileSync(resolve(root, "spec/recording_v1.json"), "utf8")) as RecordingSpec;
const vocab = JSON.parse(readFileSync(resolve(root, "spec/vocab_include50.json"), "utf8")).words as Word[];

const frame = (hand: boolean, shoulders = 0.9): StoredFrame => {
  const pose = Array.from({ length: 25 }, () => [0.5, 0.5, 0, 0.9]);
  pose[11][3] = shoulders;
  pose[12][3] = shoulders;
  const h = Array.from({ length: 21 }, () => [0.4, 0.6, 0]);
  const z = Array.from({ length: 21 }, () => [0, 0, 0]);
  return { pose, left: hand ? h : z, right: z, pose_present: true, left_present: hand, right_present: false };
};

const emptyClip = (kind: "sign" | "no_sign" = "sign"): RecordedClip => ({
  clip_id: "c1",
  kind,
  label: kind === "sign" ? "hello" : "no_sign",
  label_id: kind === "sign" ? 22 : -1,
  take: 1,
  prompt: "Hello",
  started_at: "2026-11-25T10:00:00.000Z",
  frame_size: [640, 480],
  camera_size: [640, 480],
  t_ms: [],
  pose: [],
  left: [],
  right: [],
  pose_present: [],
  left_present: [],
  right_present: [],
});

const meta: SessionMeta = {
  session_id: "S-TEST",
  participant_id: "P-ABC234",
  consent: { version: "1", accepted_at: "2026-11-25T10:00:00.000Z", text_sha256: "x".repeat(64) },
  created_at: "2026-11-25T10:00:00.000Z",
  app: { commit: null, user_agent: "test" },
  spec_version: "0.7.0",
  mediapipe_version: "1.2.3",
  model_sha256: { holistic_landmarker: "y".repeat(64) },
  delegate: "GPU",
  extract_height: 480,
};

describe("plan and ids", () => {
  it("covers every word × takes plus no-sign clips, shuffled reproducibly", () => {
    const plan = buildPlan(vocab, 2, 8, 1);
    expect(plan).toHaveLength(50 * 2 + 8);
    expect(plan).toEqual(buildPlan(vocab, 2, 8, 1));
    expect(plan.filter((p) => p.kind === "no_sign")).toHaveLength(8);
    expect(new Set(plan.filter((p) => p.kind === "sign").map((p) => p.labelId)).size).toBe(50);
    expect(plan.slice(0, 10).map((p) => p.label)).not.toEqual(buildPlan(vocab, 2, 8, 2).slice(0, 10).map((p) => p.label));
  });

  it("participant ids are anonymous and match the spec pattern", () => {
    const id = participantId(new Uint8Array([0, 1, 2, 31, 32, 255]));
    expect(id).toMatch(new RegExp(recSpec.participant_id_pattern));
  });
});

describe("quality", () => {
  it("flags missing hands in sign clips but not in no-sign clips", () => {
    const s = emptyClip("sign");
    const n = emptyClip("no_sign");
    for (let i = 0; i < 30; i++) {
      appendFrame(s, i * 33.3, frame(false));
      appendFrame(n, i * 33.3, frame(false));
    }
    expect(clipQuality(s).warnings.some((w) => w.includes("hands"))).toBe(true);
    expect(clipQuality(n).warnings.some((w) => w.includes("hands"))).toBe(false);
    expect(clipQuality(s).fps).toBeCloseTo(30, 0);
  });

  it("flags invisible shoulders (the body reference)", () => {
    const c = emptyClip();
    for (let i = 0; i < 30; i++) appendFrame(c, i * 33, frame(true, 0.2));
    expect(clipQuality(c).warnings.some((w) => w.includes("shoulders"))).toBe(true);
  });
});

describe("session file", () => {
  const clip = emptyClip();
  for (let i = 0; i < 12; i++) appendFrame(clip, i * 33.3333333, frame(true));
  const file = buildSessionFile(meta, [clip]);

  it("validates against spec/recording_v1.json", () => {
    expect(validateSession(file as unknown as Record<string, unknown>, recSpec)).toEqual([]);
  });

  it("rounds keypoints and never contains image/video/name fields", () => {
    expect(JSON.stringify(file)).not.toMatch(/"(image|video|dataUrl|name|email)":/);
    expect(file.clips[0].t_ms[1]).toBe(33.333);
  });

  it("rejects broken files", () => {
    const bad = { ...file, participant_id: "Mukund", clips: [{ ...file.clips[0], t_ms: [5, 3] }] };
    const errs = validateSession(bad as unknown as Record<string, unknown>, recSpec);
    expect(errs).toContain("participant_id");
    expect(errs.some((e) => e.includes("t_ms"))).toBe(true);
    const withImage = { ...file, image: "data:..." };
    expect(validateSession(withImage as unknown as Record<string, unknown>, recSpec)).toContain("forbidden key image");
  });
});
