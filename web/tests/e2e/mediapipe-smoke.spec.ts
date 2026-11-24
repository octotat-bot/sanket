// Phase 0 gate (browser side): the pinned @mediapipe/tasks-vision runs the SAME .task files
// (sha256-checked into ml/.cache/models by `make models`) in Chromium.
// Python side: ml/tests/test_mediapipe_smoke.py.
//
// No Next.js build needed: Playwright serves a tiny page on a fake origin and answers every
// request from local files, so the test is fast and has no network dependency at run time.
import { existsSync, readFileSync } from "node:fs";
import { extname, resolve } from "node:path";
import { expect, test } from "@playwright/test";

const root = resolve(__dirname, "../../..");
const tasksVision = resolve(root, "web/node_modules/@mediapipe/tasks-vision");
const modelsDir = resolve(root, "ml/.cache/models");
const ORIGIN = "http://sanket.test";

const MIME: Record<string, string> = {
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".task": "application/octet-stream",
  ".html": "text/html",
};

const PAGE = `<!doctype html><html><body><script type="module">
import { FilesetResolver, HolisticLandmarker, HandLandmarker, PoseLandmarker }
  from "/tv/vision_bundle.mjs";
window.runSmoke = async () => {
  const fileset = await FilesetResolver.forVisionTasks("/tv/wasm");
  const canvas = document.createElement("canvas");
  canvas.width = 640; canvas.height = 480;
  canvas.getContext("2d").fillRect(0, 0, 640, 480);  // blank frame: no person
  const out = {};
  for (const [name, Cls] of [["holistic_landmarker", HolisticLandmarker],
                             ["hand_landmarker", HandLandmarker],
                             ["pose_landmarker_full", PoseLandmarker]]) {
    const lm = await Cls.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "/models/" + name + ".task", delegate: "CPU" },
      runningMode: "VIDEO",
    });
    let result;
    for (let i = 0; i < 3; i++) result = lm.detectForVideo(canvas, i * 33);
    out[name] = Object.keys(result).sort();
    lm.close();
  }
  return out;
};
window.ready = true;
</script></body></html>`;

test("pinned MediaPipe landmarkers load and run in the browser", async ({ page }) => {
  test.skip(!existsSync(modelsDir), "run `make models` first");

  await page.route(`${ORIGIN}/**`, async (route) => {
    const path = new URL(route.request().url()).pathname;
    let file: string | null = null;
    if (path === "/") return route.fulfill({ body: PAGE, contentType: MIME[".html"] });
    if (path.startsWith("/tv/")) file = resolve(tasksVision, path.slice(4));
    if (path.startsWith("/models/")) file = resolve(modelsDir, path.slice(8));
    if (!file || !existsSync(file)) return route.fulfill({ status: 404 });
    return route.fulfill({ body: readFileSync(file), contentType: MIME[extname(file)] });
  });

  await page.goto(`${ORIGIN}/`);
  await page.waitForFunction(() => (window as unknown as { ready?: boolean }).ready === true);
  const keys = await page.evaluate(() =>
    (window as unknown as { runSmoke: () => Promise<Record<string, string[]>> }).runSmoke(),
  );

  expect(keys.holistic_landmarker).toEqual(
    expect.arrayContaining(["poseLandmarks", "leftHandLandmarks", "rightHandLandmarks"]),
  );
  expect(keys.hand_landmarker).toEqual(expect.arrayContaining(["landmarks", "handedness"]));
  expect(keys.pose_landmarker_full).toEqual(expect.arrayContaining(["landmarks"]));
});
