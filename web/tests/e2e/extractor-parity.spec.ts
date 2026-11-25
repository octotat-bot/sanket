// Python vs browser MediaPipe on the SAME frames of an INCLUDE-50 test video (ADR 002 follow-up;
// an early, narrow form of Phase 4 parity test 4).
//
// Fixture: `uv run python -m sanket.checks.browser_parity_fixture` writes OpenCV-decoded
// lossless frames + Python results into ml/.cache/parity/ (gitignored; INCLUDE frames are
// never committed). The browser runs both extractors on those exact pixels with the same
// timestamps, and we compare hand rates, per-frame presence and hand landmark positions.
import { existsSync, readFileSync } from "node:fs";
import { extname, resolve } from "node:path";
import { expect, test } from "@playwright/test";

const root = resolve(__dirname, "../../..");
const parity = resolve(root, "ml/.cache/parity");
const tasksVision = resolve(root, "web/node_modules/@mediapipe/tasks-vision");
const modelsDir = resolve(root, "ml/.cache/models");
const ORIGIN = "http://parity.test";
const MIME: Record<string, string> = {
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".task": "application/octet-stream",
  ".png": "image/png",
  ".json": "application/json",
};

type Pt = [number, number];
interface Side {
  hand_rate: number;
  left_present: boolean[];
  right_present: boolean[];
  left: Pt[][];
  right: Pt[][];
}

const PAGE = `<!doctype html><html><body><script type="module">
import { FilesetResolver, HolisticLandmarker, HandLandmarker, PoseLandmarker } from "/tv/vision_bundle.mjs";
const xy = (lms) => lms.map((p) => [p.x, p.y]);
// mode "480": downscale in the browser exactly as the app will (spec "downscale": even width,
// halves round up; createImageBitmap resizeQuality "high"), vs Python's INTER_AREA reference.
window.runParity = async (name, delegate, mode) => {
  const fs = await FilesetResolver.forVisionTasks("/tv/wasm");
  const stamps = await (await fetch("/parity/stamps.json")).json();
  const base = (m) => ({ modelAssetPath: "/models/" + m + ".task", delegate });
  const out = { left_present: [], right_present: [], left: [], right: [] };
  const push = (l, r) => {
    out.left_present.push(!!l); out.right_present.push(!!r);
    out.left.push(l ? xy(l) : []); out.right.push(r ? xy(r) : []);
  };
  let step, close;
  if (name === "holistic") {
    const lm = await HolisticLandmarker.createFromOptions(fs, { baseOptions: base("holistic_landmarker"), runningMode: "VIDEO" });
    step = (img, t) => lm.detectForVideo(img, t, (r) => push(r.leftHandLandmarks[0], r.rightHandLandmarks[0]));
    close = () => lm.close();
  } else {
    const hand = await HandLandmarker.createFromOptions(fs, { baseOptions: base("hand_landmarker"), runningMode: "VIDEO", numHands: 2 });
    const pose = await PoseLandmarker.createFromOptions(fs, { baseOptions: base("pose_landmarker_full"), runningMode: "VIDEO" });
    step = (img, t) => {
      const r = hand.detectForVideo(img, t);
      pose.detectForVideo(img, t, () => {});
      // Same rule as Python: the signer's own side; if two hands share a label keep the more confident.
      const ranked = r.landmarks.map((l, i) => [l, r.handedness[i][0]]).sort((a, b) => b[1].score - a[1].score);
      const sides = {};
      for (const [l, c] of ranked) { const s = c.categoryName.toLowerCase(); if (!sides[s]) sides[s] = l; }
      push(sides.left, sides.right);
    };
    close = () => { hand.close(); pose.close(); };
  }
  for (let i = 0; i < stamps.length; i++) {
    const blob = await (await fetch("/parity/frames/" + String(i).padStart(4, "0") + ".png")).blob();
    const full = await createImageBitmap(blob);
    let bmp = full;
    if (mode === "480" && full.height > 480) {
      const w = 2 * Math.floor((full.width * 480) / full.height / 2 + 0.5);
      bmp = await createImageBitmap(full, { resizeWidth: w, resizeHeight: 480, resizeQuality: "high" });
      full.close();
    }
    step(bmp, stamps[i]);
    bmp.close();
  }
  close();
  return out;
};
window.ready = true;
</script></body></html>`;

function compare(py: Side, br: Omit<Side, "hand_rate">) {
  const n = py.left_present.length;
  let agree = 0;
  const dists: number[] = [];
  let brHand = 0;
  for (let i = 0; i < n; i++) {
    if (br.left_present[i] || br.right_present[i]) brHand++;
    for (const s of ["left", "right"] as const) {
      const pp = py[`${s}_present`][i];
      const bp = br[`${s}_present`][i];
      if (pp === bp) agree++;
      if (pp && bp) {
        const d = py[s][i].map((p, k) => Math.hypot(p[0] - br[s][i][k][0], p[1] - br[s][i][k][1]));
        dists.push(d.reduce((a, b) => a + b, 0) / d.length);
      }
    }
  }
  dists.sort((a, b) => a - b);
  const q = (p: number) => (dists.length ? dists[Math.min(dists.length - 1, Math.floor(p * dists.length))] : 0);
  return {
    pyRate: py.hand_rate,
    brRate: brHand / n,
    presenceAgreement: agree / (2 * n),
    landmarkDistMedian: q(0.5),
    landmarkDistP95: q(0.95),
    comparedHands: dists.length,
  };
}

// native: identical pixels on both sides. 480: each side downscales the same native frames its
// own way (Python INTER_AREA, browser createImageBitmap "high"), as offline extraction and the
// live app will (ADR 007).
// Tolerances, from the first measurements (2026-11-25, recorded in ADR 007):
// * native: identical pixels → near-identical results (measured presence 99.4% / 100%).
// * 480: the two resizers differ slightly (pixel MAE 0.29 / 255). Holistic measured presence
//   96.4%, so it keeps the 95% bar. Hand+Pose measured 94.0%: its palm detector flips on
//   marginal frames and tracking carries the flip, so its bar is 92%. Landmark positions agree
//   closely for both where both detect (median ≤ 0.001, p95 ≤ 0.007).
const TOL: Record<string, Record<"native" | "480", { rate: number; presence: number; median: number; p95: number }>> = {
  holistic: {
    native: { rate: 0.03, presence: 0.95, median: 0.002, p95: 0.01 },
    "480": { rate: 0.03, presence: 0.95, median: 0.002, p95: 0.01 },
  },
  hand_pose: {
    native: { rate: 0.03, presence: 0.95, median: 0.002, p95: 0.01 },
    "480": { rate: 0.03, presence: 0.92, median: 0.002, p95: 0.01 },
  },
};

for (const [name, mode] of [
  ["holistic", "native"],
  ["hand_pose", "native"],
  ["holistic", "480"],
  ["hand_pose", "480"],
] as const) {
  const title =
    mode === "native"
      ? `browser ${name} matches Python on identical INCLUDE frames`
      : `browser ${name}: browser-downscaled 480 rows matches Python INTER_AREA 480 rows`;
  test(title, async ({ page }) => {
    test.skip(!existsSync(resolve(parity, "expected.json")), "run sanket.checks.browser_parity_fixture first");
    const expected = JSON.parse(readFileSync(resolve(parity, "expected.json"), "utf8"));

    await page.route(`${ORIGIN}/**`, async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (path === "/") return route.fulfill({ body: PAGE, contentType: "text/html" });
      const file = path.startsWith("/tv/")
        ? resolve(tasksVision, path.slice(4))
        : path.startsWith("/models/")
          ? resolve(modelsDir, path.slice(8))
          : path.startsWith("/parity/")
            ? resolve(parity, path.slice(8))
            : null;
      if (!file || !existsSync(file)) return route.fulfill({ status: 404 });
      return route.fulfill({ body: readFileSync(file), contentType: MIME[extname(file)] });
    });
    await page.goto(`${ORIGIN}/`);
    await page.waitForFunction(() => (window as unknown as { ready?: boolean }).ready === true);
    const br = await page.evaluate(
      ([n, m]) =>
        (window as unknown as { runParity: (n: string, d: string, m: string) => Promise<Omit<Side, "hand_rate">> }).runParity(n === "hand_pose" ? "handpose" : n, "CPU", m),
      [name, mode] as const,
    );
    const r = compare(expected[mode === "native" ? name : `${name}_480`] as Side, br);
    console.log(name, mode, JSON.stringify(r));
    test.info().annotations.push({ type: "parity", description: JSON.stringify(r) });

    const t = TOL[name][mode];
    expect(Math.abs(r.brRate - r.pyRate)).toBeLessThanOrEqual(t.rate);
    expect(r.presenceAgreement).toBeGreaterThanOrEqual(t.presence);
    expect(r.comparedHands).toBeGreaterThan(20);
    expect(r.landmarkDistMedian).toBeLessThan(t.median); // normalized image units
    expect(r.landmarkDistP95).toBeLessThan(t.p95);
  });
}
