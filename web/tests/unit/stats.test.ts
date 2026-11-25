import { describe, expect, it } from "vitest";
import { blockBootstrapMeanCI, mean, percentile, sha256Hex } from "@/lib/bench/stats";

describe("percentile", () => {
  it("interpolates between ranks", () => {
    expect(percentile([1, 2, 3, 4, 5], 50)).toBe(3);
    expect(percentile([1, 2, 3, 4], 50)).toBe(2.5);
    expect(percentile([10, 20], 95)).toBeCloseTo(19.5);
    expect(percentile([5, 1, 3], 0)).toBe(1); // order of input doesn't matter
    expect(percentile([5, 1, 3], 100)).toBe(5);
  });
  it("is NaN for no samples", () => {
    expect(percentile([], 50)).toBeNaN();
    expect(mean([])).toBeNaN();
  });
});

describe("sha256Hex", () => {
  it("matches the known digest of 'abc'", async () => {
    expect(await sha256Hex(new TextEncoder().encode("abc"))).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });
});

describe("blockBootstrapMeanCI", () => {
  const noise = Array.from({ length: 200 }, (_, i) => Math.sin(i * 12.9898) * 0.5); // mean ≈ 0

  it("is deterministic for a seed", () => {
    expect(blockBootstrapMeanCI(noise, { seed: 3 })).toEqual(blockBootstrapMeanCI(noise, { seed: 3 }));
  });

  it("includes zero for zero-mean noise", () => {
    const r = blockBootstrapMeanCI(noise);
    expect(r.lo).toBeLessThan(0);
    expect(r.hi).toBeGreaterThan(0);
    expect(r.excludesZero).toBe(false);
  });

  it("excludes zero for a clear offset", () => {
    const r = blockBootstrapMeanCI(noise.map((x) => x + 1));
    expect(r.excludesZero).toBe(true);
    expect(r.lo).toBeGreaterThan(0.8);
  });

  it("gives a wider CI for correlated data than independent resampling would", () => {
    // 20 runs of 10 identical values: effectively 20 samples, not 200.
    const runs = Array.from({ length: 200 }, (_, i) => Math.sin(Math.floor(i / 10) * 7.1));
    const blocky = blockBootstrapMeanCI(runs, { block: 10 });
    const naive = blockBootstrapMeanCI(runs, { block: 1 });
    expect(blocky.hi - blocky.lo).toBeGreaterThan(naive.hi - naive.lo);
  });
});
