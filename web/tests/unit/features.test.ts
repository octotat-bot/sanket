// TypeScript features must equal the Python reference on the shared golden file (to 1e-9).
// Golden: spec/golden/features_v1.json, written by `python -m sanket.keypoints.golden`.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { type Clip, clipToModelInput, featureSpecFrom, fillShortGaps, layout } from "@/lib/inference/features";

const root = resolve(__dirname, "../../..");
const spec = JSON.parse(readFileSync(resolve(root, "spec/keypoints.json"), "utf8"));
const golden = JSON.parse(readFileSync(resolve(root, "spec/golden/features_v1.json"), "utf8"));
const fs = featureSpecFrom(spec);

function maxAbsDiff(a: Float64Array[], b: number[][]): number {
  expect(a.length).toBe(b.length);
  let m = 0;
  a.forEach((row, i) => {
    expect(row.length).toBe(b[i].length);
    row.forEach((v, j) => (m = Math.max(m, Math.abs(v - b[i][j]))));
  });
  return m;
}

describe("features v1 vs Python golden file", () => {
  it("has the spec's feature count", () => {
    expect(layout(fs).nFeatures).toBe(spec.normalization.features_v1.n_features);
    expect(golden.n_features).toBe(spec.normalization.features_v1.n_features);
  });

  golden.clips.forEach((c: { input: Clip; expected: number[][]; expected_velocity: number[][] }, k: number) => {
    it(`clip ${k}: ${c.input.frame_size.join("x")}, ${c.input.t_ms.length} frames`, () => {
      expect(maxAbsDiff(clipToModelInput(c.input, fs, false, golden.T), c.expected)).toBeLessThan(1e-9);
      expect(maxAbsDiff(clipToModelInput(c.input, fs, true, golden.T), c.expected_velocity)).toBeLessThan(1e-9);
    });
  });
});

describe("aspect ratio and padding (same body, different frames)", () => {
  // The same body in pixels, observed by MediaPipe in frames of different size.
  const base = golden.clips[0].input as Clip;
  const [W0, H0] = base.frame_size;
  const reframe = (W: number, H: number): Clip => {
    const conv = (frames: number[][][]) =>
      frames.map((f) => f.map(([x, y, ...rest]) => [(x * W0) / W, (y * H0) / H, ...rest]));
    return { ...base, frame_size: [W, H], pose: conv(base.pose), left: conv(base.left), right: conv(base.right) };
  };
  const ref = clipToModelInput(base, fs, true, 12);
  for (const [W, H] of [
    [1920, 1088],
    [2560, 1440],
  ]) {
    it(`${W}x${H} gives the same features`, () => {
      const got = clipToModelInput(reframe(W, H), fs, true, 12);
      expect(maxAbsDiff(got, ref.map((r) => Array.from(r)))).toBeLessThan(1e-9);
    });
  }
});

describe("gap rule is in milliseconds (same meaning at any frame rate)", () => {
  const base = golden.clips[0].input as Clip;
  // 120 ms fills 1 missing frame at 15 fps, 3 at 25 fps and 3 (not 4) at 30 fps.
  for (const [fps, maxFilled] of [
    [15, 1],
    [25, 3],
    [30, 3],
  ]) {
    it(`${fps} fps fills runs of up to ${maxFilled} frames`, () => {
      for (let k = 1; k <= 5; k++) {
        const n = k + 2;
        const c: Clip = {
          ...base,
          t_ms: Array.from({ length: n }, (_, i) => (i * 1000) / fps),
          pose: base.pose.slice(0, n),
          left: base.left.slice(0, n).map((f) => f.map((p) => [...p])),
          right: base.right.slice(0, n),
          pose_present: Array(n).fill(true),
          left_present: Array.from({ length: n }, (_, i) => i === 0 || i === n - 1),
          right_present: Array(n).fill(true),
        };
        c.left[0] = c.left[0].map(() => [0.3, 0.5, 0]);
        c.left[n - 1] = c.left[n - 1].map(() => [0.5, 0.5, 0]);
        const out = fillShortGaps(c, fs.maxGapMs);
        const filled = out.left_present.slice(1, n - 1).every(Boolean);
        expect(filled, `fps ${fps}, k ${k}`).toBe(k <= maxFilled);
      }
    });
  }
});
