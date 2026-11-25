// ADR 007: which browser downscale best matches offline extraction (OpenCV INTER_AREA, 480 rows)?
// Usage (from web/, after `make parity-fixture`): node scripts/downscale-probe.mjs
// For every candidate method, downscale the parity fixture's native INCLUDE frames to 854x480,
// compare pixels with OpenCV INTER_AREA frames, run Holistic (CPU) on them and compare hand
// landmarks with the Python 480-row reference. Runs in Playwright Chromium and installed Chrome.
// Writes ../reports/phase1_browser_downscale.json.
import { writeFileSync } from "node:fs";
import { chromium } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import { extname, resolve } from "node:path";
const root = resolve(import.meta.dirname, "../.."), parity = resolve(root, "ml/.cache/parity"), tv = resolve(import.meta.dirname, "../node_modules/@mediapipe/tasks-vision");
const MIME = { ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm", ".task": "application/octet-stream", ".png": "image/png", ".json": "application/json" };
const PAGE = `<!doctype html><script type=module>
import { FilesetResolver, HolisticLandmarker } from "/tv/vision_bundle.mjs";
const W = 854, H = 480;
const pad = (i) => String(i).padStart(4, "0");
const blob = async (p) => (await fetch(p)).blob();
async function down(method, full) {
  if (method.startsWith("bitmap-")) return createImageBitmap(full, { resizeWidth: W, resizeHeight: H, resizeQuality: method.slice(7) });
  const c = new OffscreenCanvas(W, H), g = c.getContext("2d");
  if (method === "canvas-off") g.imageSmoothingEnabled = false; else g.imageSmoothingQuality = method.slice(7);
  g.drawImage(full, 0, 0, W, H); return c.transferToImageBitmap();
}
function rgba(bmp) { const c = new OffscreenCanvas(W, H), g = c.getContext("2d"); g.drawImage(bmp, 0, 0); return g.getImageData(0, 0, W, H).data; }
window.measure = async (method) => {
  const stamps = await (await fetch("/p/stamps.json")).json();
  const exp = await (await fetch("/p/expected.json")).json();
  const fs = await FilesetResolver.forVisionTasks("/tv/wasm");
  const lm = await HolisticLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: "/models/holistic_landmarker.task", delegate: "CPU" }, runningMode: "VIDEO" });
  let diff = 0, nPix = 0, hands = 0, agree = 0; const d = [];
  for (let i = 0; i < stamps.length; i++) {
    const full = await createImageBitmap(await blob("/p/frames/" + pad(i) + ".png"));
    const mine = await down(method, full); full.close();
    const ref = await createImageBitmap(await blob("/p/frames480/" + pad(i) + ".png"));
    const a = rgba(mine), b = rgba(ref); ref.close();
    for (let k = 0; k < a.length; k += 4) { diff += Math.abs(a[k] - b[k]) + Math.abs(a[k+1] - b[k+1]) + Math.abs(a[k+2] - b[k+2]); nPix += 3; }
    let L, R; lm.detectForVideo(mine, stamps[i], (r) => { L = r.leftHandLandmarks[0]; R = r.rightHandLandmarks[0]; }); mine.close();
    if (L || R) hands++;
    for (const [side, got] of [["left", L], ["right", R]]) {
      const ep = exp.holistic_480[side + "_present"][i];
      if (ep === !!got) agree++;
      if (ep && got) { const e = exp.holistic_480[side][i]; d.push(got.reduce((s, p, k) => s + Math.hypot(p.x - e[k][0], p.y - e[k][1]), 0) / got.length); }
    }
  }
  lm.close(); d.sort((x, y) => x - y);
  return { pixelMAE: +(diff / nPix).toFixed(3), handRate: +(hands / stamps.length).toFixed(3), pyHandRate480: exp.holistic_480.hand_rate.toFixed(3),
    presenceAgree: +(agree / (2 * stamps.length)).toFixed(3), lmMedian: +d[Math.floor(d.length / 2)].toFixed(5), lmP95: +d[Math.floor(0.95 * d.length)].toFixed(5) };
};
window.ready = true;</script>`;
const results = {};
const methods = ["bitmap-pixelated", "bitmap-low", "bitmap-medium", "bitmap-high", "canvas-off", "canvas-low", "canvas-medium", "canvas-high"];
for (const channel of [undefined, "chrome"]) {
  const b = await chromium.launch({ channel }); const page = await b.newPage();
  await page.route("http://rs.test/**", (route) => {
    const p = new URL(route.request().url()).pathname;
    if (p === "/") return route.fulfill({ body: PAGE, contentType: "text/html" });
    const f = p.startsWith("/tv/") ? resolve(tv, p.slice(4)) : p.startsWith("/models/") ? resolve(root, "ml/.cache/models", p.slice(8)) : p.startsWith("/p/") ? resolve(parity, p.slice(3)) : null;
    if (!f || !existsSync(f)) return route.fulfill({ status: 404 });
    return route.fulfill({ body: readFileSync(f), contentType: MIME[extname(f)] });
  });
  await page.goto("http://rs.test/"); await page.waitForFunction(() => window.ready);
  for (const m of methods) {
    const r = await page.evaluate((x) => window.measure(x), m);
    (results[channel ?? "playwright-chromium"] ??= {})[m] = r;
    console.log(channel ?? "chromium", m.padEnd(17), JSON.stringify(r));
  }
  await b.close();
}
writeFileSync(resolve(root, "reports/phase1_browser_downscale.json"), JSON.stringify(results, null, 2) + "\n");
