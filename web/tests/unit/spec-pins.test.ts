// The browser must use the same keypoint extractor as training (CLAUDE.md, Phase 0 gate).
// Mirrors ml/tests/test_spec.py from the web side.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const root = resolve(__dirname, "../../..");
const spec = JSON.parse(readFileSync(resolve(root, "spec/keypoints.json"), "utf8"));
const pkg = JSON.parse(readFileSync(resolve(root, "web/package.json"), "utf8"));
const installed = JSON.parse(
  readFileSync(resolve(root, "web/node_modules/@mediapipe/tasks-vision/package.json"), "utf8"),
);

describe("keypoint extractor pins", () => {
  const version: string = spec.extractor.mediapipe_version;

  it("package.json pins @mediapipe/tasks-vision exactly to the spec version", () => {
    expect(pkg.dependencies["@mediapipe/tasks-vision"]).toBe(version);
  });

  it("the installed package is that version", () => {
    expect(installed.version).toBe(version);
  });

  it("the spec's web package string agrees with mediapipe_version", () => {
    expect(spec.extractor.packages.web).toBe(`@mediapipe/tasks-vision@${version}`);
    expect(spec.extractor.packages.python).toBe(`mediapipe==${version}`);
  });

  it("every dependency is pinned to an exact version", () => {
    const all = { ...pkg.dependencies, ...pkg.devDependencies } as Record<string, string>;
    for (const [name, range] of Object.entries(all)) {
      expect(range, name).toMatch(/^\d+\.\d+\.\d+(-[\w.]+)?$/);
    }
  });
});
