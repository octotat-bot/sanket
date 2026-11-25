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
