// Python and the browser must compute the same downscaled size (spec "downscale").
// The same cases are checked in ml/tests/test_extract_pipeline.py.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { EXTRACT_HEIGHT, targetSize } from "@/lib/frames/downscale";

const spec = JSON.parse(readFileSync(resolve(__dirname, "../../../spec/keypoints.json"), "utf8"));

describe("downscale size (shared with Python)", () => {
  it("uses the spec's height", () => {
    expect(EXTRACT_HEIGHT).toBe(spec.downscale.height);
  });

  it.each(spec.downscale.shared_test_cases as [number, number, number, number][])(
    "%i x %i -> %i x %i",
    (w, h, ew, eh) => {
      expect(targetSize(w, h, spec.downscale.height)).toEqual([ew, eh]);
    },
  );
});
