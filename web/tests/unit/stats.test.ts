import { describe, expect, it } from "vitest";
import { mean, percentile, sha256Hex } from "@/lib/bench/stats";

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
