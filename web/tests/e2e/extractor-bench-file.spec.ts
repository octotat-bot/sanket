// The real benchmark page, in INCLUDE-video-file mode, finds hands at the rates Python does.
//
// Uses the INCLUDE-50 test video fetched by `sanket.checks.browser_parity_fixture` (gitignored
// cache; never committed) and the Python reference rates it computed. Playback samples frames
// as they are presented (not every decoded frame, and from wherever the loop is), so the
// tolerance is wider than in extractor-parity.spec.ts, which compares frame by frame.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";

const root = resolve(__dirname, "../../..");
const expectedPath = resolve(root, "ml/.cache/parity/expected.json");

interface Result {
  id: string;
  source: string;
  ok: boolean;
  error?: string;
  handRate?: number;
  processedFrames?: number;
  warmupFrames?: number;
  firstFramesMs?: number[];
}

test("file mode: Holistic and Hand+Pose hand rates match Python on the same INCLUDE video", async ({
  page,
}) => {
  test.skip(!existsSync(expectedPath), "run sanket.checks.browser_parity_fixture first");
  const expected = JSON.parse(readFileSync(expectedPath, "utf8"));
  const video = resolve(root, "ml/.cache/videos", expected.video);

  await page.goto("/dev/extractor-bench");
  // Linux builds of Playwright's Chromium may lack H.264; then this test can't decode INCLUDE
  // videos. extractor-parity.spec.ts (decoded frames, no video codec) still runs everywhere.
  const h264 = await page.evaluate(() =>
    document.createElement("video").canPlayType('video/mp4; codecs="avc1.640028"'),
  );
  test.skip(h264 === "", "this browser build cannot decode H.264 (INCLUDE videos)");
  await page.getByLabel("INCLUDE video file (same input for every configuration)").check();
  await page.getByLabel("Video file", { exact: true }).setInputFiles(video);
  await page.getByLabel("A · Holistic · GPU").uncheck();
  await page.getByLabel("B · Hand + Pose · GPU").uncheck();
  await page.getByLabel("Seconds each").fill("6");
  await page.getByLabel("Warm-up").fill("1");
  await page.getByLabel("Rounds").fill("1");
  await page.getByRole("button", { name: "Run benchmark" }).click();
  await expect(page.getByRole("status")).toHaveText("Done.", { timeout: 110_000 });

  const report = await page.evaluate(
    () => (window as unknown as { __benchReport: { results: Result[]; device: { crossOriginIsolated: boolean } } }).__benchReport,
  );
  expect(report.device.crossOriginIsolated).toBe(true);
  const rate = (id: string) => report.results.find((r) => r.id === id)!;
  for (const [id, py] of [
    ["holistic-cpu", expected.holistic.hand_rate],
    ["handpose-cpu", expected.hand_pose.hand_rate],
  ] as const) {
    const r = rate(id);
    expect(r.ok, r.error).toBe(true);
    expect(r.source).toMatch(/^file:/);
    expect(r.warmupFrames).toBeGreaterThanOrEqual(60);
    expect(r.firstFramesMs?.length).toBe(10);
    console.log(id, "browser", r.handRate?.toFixed(3), "python", py.toFixed(3), "frames", r.processedFrames);
    expect(Math.abs((r.handRate ?? 0) - py)).toBeLessThanOrEqual(0.15);
  }
});
