// TypeScript features must equal the Python reference on the shared golden file (to 1e-9).
// Golden: spec/golden/features_v1.json, written by `python -m sanket.keypoints.golden`.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { type Clip, clipToModelInput, featureSpecFrom, layout } from "@/lib/inference/features";

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
