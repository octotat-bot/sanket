import { describe, expect, it } from "vitest";
import type { StoredFrame } from "@/lib/mediapipe/holistic";
import { DEFAULT_SPOTTER, Spotter, type SpotterUpdate } from "@/lib/live/spotter";
import { DEFAULT_DECIDE, decide, labelsSha256, softmax } from "@/lib/live/decide";

const FPS = 30;
const W = 640;
const H = 480;

/** A seated person: shoulders at y=0.4; `hand` = wrist position in body units (sw), or null. */
function frame(hand: [number, number] | null, scale = 1, size: [number, number] = [W, H]): StoredFrame {
  const ax = size[0] / size[1];
  const cx = 0.5;
  const cy = 0.4;
  const halfSwX = (0.1 * scale) / ax; // shoulder width = 0.2·scale in aspect-corrected units
  const sw = 0.2 * scale;
  const pose = Array.from({ length: 25 }, () => [0.5, 0.5, 0, 0.9]);
  pose[11] = [cx + halfSwX, cy, 0, 0.99]; // signer's left shoulder (image right)
  pose[12] = [cx - halfSwX, cy, 0, 0.99];
  const zeros = Array.from({ length: 21 }, () => [0, 0, 0]);
  if (!hand) return { pose, left: zeros, right: zeros, pose_present: true, left_present: false, right_present: false };
  const pts = Array.from({ length: 21 }, (_, i) => [
    cx + ((hand[0] + 0.01 * i) * sw) / ax,
    cy + (hand[1] - 0.02 * i) * sw,
    0,
  ]);
  return { pose, left: zeros, right: pts, pose_present: true, left_present: false, right_present: true };
}

type Seg = (t: number) => [number, number] | null;
const rest: Seg = () => null; // hands below the frame / on the desk out of view
const sign: Seg = (t) => [0.3 * Math.cos(2 * Math.PI * 2 * t), 0.5 + 0.3 * Math.sin(2 * Math.PI * 2 * t)];
const stillRaised: Seg = () => [0.2, 0.4];

function run(plan: [Seg, number][], scale = 1, size: [number, number] = [W, H]) {
  const s = new Spotter(() => size);
  const events: SpotterUpdate[] = [];
  let t = 0;
  for (const [seg, seconds] of plan) {
    for (let i = 0; i < seconds * FPS; i++, t += 1000 / FPS) {
      const u = s.update(t, frame(seg(i / FPS), scale, size));
      if (u.endReason) events.push(u);
    }
  }
  return events;
}

describe("sign spotting state machine", () => {
  it("one sign between rests gives exactly one clip covering the sign", () => {
    const ev = run([[rest, 1], [sign, 1.5], [rest, 1]]);
    expect(ev.map((e) => e.endReason)).toEqual(["lowered"]);
    const c = ev[0].clip!;
    const dur = c.t_ms[c.t_ms.length - 1];
    expect(dur).toBeGreaterThan(1300);
    expect(dur).toBeLessThan(1500 + DEFAULT_SPOTTER.prerollMs + 300);
    expect(c.frame_size).toEqual([W, H]);
  });

  it("a raised but still hand (resting on the chin) never starts a sign", () => {
    expect(run([[stillRaised, 3]])).toEqual([]);
  });

  it("ends when the hands stop moving, even if still raised", () => {
    const ev = run([[rest, 0.5], [sign, 1.2], [stillRaised, 1.5]]);
    expect(ev.map((e) => e.endReason)).toEqual(["still"]);
  });

  it("drops flicks shorter than minMs and caps long signing at maxMs", () => {
    expect(run([[rest, 0.5], [sign, 0.2], [rest, 1]]).map((e) => e.endReason)).toEqual(["too_short"]);
    const long = run([[rest, 0.5], [sign, 6]]);
    expect(long[0].endReason).toBe("max");
    expect(long[0].clip!.t_ms.at(-1)!).toBeLessThanOrEqual(DEFAULT_SPOTTER.maxMs + DEFAULT_SPOTTER.prerollMs + 50);
  });

  it("is invariant to distance from the camera and to the frame's aspect ratio", () => {
    const plan: [Seg, number][] = [[rest, 1], [sign, 1.5], [stillRaised, 1], [rest, 1], [sign, 1], [rest, 1]];
    const base = run(plan).map((e) => e.endReason);
    expect(base).toEqual(["still", "lowered"]);
    expect(run(plan, 0.5).map((e) => e.endReason)).toEqual(base);
    expect(run(plan, 1, [854, 480]).map((e) => e.endReason)).toEqual(base);
  });

  it("never starts without visible shoulders (no body reference)", () => {
    const s = new Spotter(() => [W, H]);
    for (let i = 0; i < 60; i++) {
      const f = frame(sign(i / FPS));
      f.pose[11][3] = 0.1;
      expect(s.update((i * 1000) / FPS, f).state).toBe("idle");
    }
  });
});

describe("decision and rejection", () => {
  const labels = ["a", "b", "c", "d"];
  it("softmax is stable and sums to one", () => {
    const p = softmax([1000, 1001, 999, -1000]);
    expect(p.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 12);
    expect(p[1]).toBeGreaterThan(p[0]);
  });
  it("accepts a clear winner, rejects low confidence and close calls", () => {
    const ok = decide([5, 0, 0, 0], labels);
    expect(ok.accepted).toBe(true);
    expect(ok.top[0].label).toBe("a");
    expect(decide([0.1, 0, 0, 0], labels)).toMatchObject({ accepted: false, reason: "low_confidence" });
    expect(decide([3, 2.9, -5, -5], labels)).toMatchObject({ accepted: false, reason: "ambiguous" });
    expect(decide([5, 0, 0, 0], labels).top).toHaveLength(DEFAULT_DECIDE.topK);
  });
  it("temperature > 1 lowers confidence", () => {
    expect(softmax([2, 0], 2)[0]).toBeLessThan(softmax([2, 0], 1)[0]);
  });
  it("labels hash matches Python's labels_sha256 for INCLUDE-50", async () => {
    const { readFileSync } = await import("node:fs");
    const { resolve } = await import("node:path");
    const vocab = JSON.parse(readFileSync(resolve(__dirname, "../../../spec/vocab_include50.json"), "utf8")).words as {
      id: number;
      label: string;
    }[];
    const ids = vocab.sort((x, y) => x.id - y.id).map((w) => w.label);
    // python: hashlib.sha256(json.dumps(labels, separators=(",", ":")).encode()).hexdigest()
    expect(await labelsSha256(ids)).toBe("572dda29ef10ff0061ec2a68338d624bb7b252321bdf1449d5dff030bae6e81e");
  });
});
