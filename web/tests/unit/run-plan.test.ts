import { describe, expect, it } from "vitest";
import { CONFIG_IDS, planRuns } from "@/lib/bench/extractor-bench";

describe("planRuns", () => {
  it("is reproducible from the seed", () => {
    expect(planRuns(CONFIG_IDS, 3, 42)).toEqual(planRuns(CONFIG_IDS, 3, 42));
  });

  it("runs every config exactly once per round", () => {
    const plan = planRuns(CONFIG_IDS, 3, 7);
    for (const round of [1, 2, 3]) {
      const ids = plan.filter((p) => p.round === round).map((p) => p.id);
      expect([...ids].sort()).toEqual([...CONFIG_IDS].sort());
    }
  });

  it("actually shuffles (different seeds give different orders)", () => {
    const orders = new Set(
      Array.from({ length: 20 }, (_, s) => planRuns(CONFIG_IDS, 1, s).map((p) => p.id).join()),
    );
    expect(orders.size).toBeGreaterThan(5);
  });
});

describe("ADR 002 speed rule", () => {
  it("applies the M3 Pro bar and the mid-range floor", async () => {
    const { meetsBar, BENCH_MACHINE_BAR, MID_RANGE_FLOOR } = await import("@/lib/bench/extractor-bench");
    expect(meetsBar(20, 50, BENCH_MACHINE_BAR)).toBe(true);
    expect(meetsBar(19.9, 40, BENCH_MACHINE_BAR)).toBe(false); // too few frames
    expect(meetsBar(25, 50.1, BENCH_MACHINE_BAR)).toBe(false); // p95 too slow
    // Run 1's Holistic CPU (19.8 fps, p95 64 ms): passes the floor, fails the M3 Pro bar.
    expect(meetsBar(19.8, 64, MID_RANGE_FLOOR)).toBe(true);
    expect(meetsBar(19.8, 64, BENCH_MACHINE_BAR)).toBe(false);
  });
});
