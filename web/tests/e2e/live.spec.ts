// The Live page with Chromium's fake camera and a tiny UNTRAINED fixture model served in place of
// /models/include50.* (ml/src/sanket/export/fixture.py): the pipeline loads, warms up, runs, and
// refuses a model whose labels don't match the app's vocabulary. Predictions are not checked
// here (the fixture is untrained); the trained-model check uses a recorded sign video (Phase 5).
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, type Page, test } from "@playwright/test";

const fixture = resolve(__dirname, "../fixtures/model");

async function serveFixture(page: Page, mutateMeta?: (m: Record<string, unknown>) => void) {
  await page.route("**/models/include50.*", async (route) => {
    const name = new URL(route.request().url()).pathname.split("/").pop()!;
    const body = readFileSync(resolve(fixture, name));
    if (name.endsWith(".json") && mutateMeta) {
      const m = JSON.parse(body.toString());
      mutateMeta(m);
      return route.fulfill({ json: m });
    }
    return route.fulfill({ body, contentType: name.endsWith(".json") ? "application/json" : "application/octet-stream" });
  });
}

test("live page: warms up, starts the camera, runs the pipeline, stays on its own origin", async ({ page, baseURL }) => {
  const foreign: string[] = [];
  page.on("request", (r) => {
    if (!r.url().startsWith(baseURL!) && !r.url().startsWith("data:") && !r.url().startsWith("blob:")) foreign.push(r.url());
  });
  page.on("console", (m) => m.type() === "error" && console.log("[page]", m.text()));
  await serveFixture(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /The 50 words it knows/ })).toBeVisible();
  const start = page.getByRole("button", { name: "Start camera" });
  await expect(start).toBeVisible({ timeout: 90_000 });
  expect(await page.evaluate(() => self.crossOriginIsolated)).toBe(true);
  await start.click();
  await expect(page.getByRole("status")).toContainText("Watching for a sign", { timeout: 30_000 });
  await page.getByRole("button", { name: "Show performance" }).click();
  const dbg = page.getByTestId("debug");
  await expect(dbg).toContainText(/WebGPU|WebAssembly/);
  await expect(dbg).toContainText(/include50\.(fp32|int8)\.onnx/);
  await expect.poll(async () => Number((await dbg.innerText()).match(/([\d.]+) fps/)?.[1] ?? 0), { timeout: 20_000 }).toBeGreaterThan(0.5); // CI runner (no GPU) ~2 fps
  await page.getByRole("button", { name: "Stop camera" }).click();
  await expect(start).toBeVisible();
  expect(foreign).toEqual([]);
});

test("live page refuses a model whose labels don't match the vocabulary", async ({ page }) => {
  await serveFixture(page, (m) => {
    const labels = m.labels as string[];
    [labels[0], labels[1]] = [labels[1], labels[0]]; // label-mapping drift
  });
  await page.goto("/");
  await expect(page.getByText(/labels hash mismatch/)).toBeVisible({ timeout: 90_000 });
  await expect(page.getByRole("button", { name: "Start camera" })).toHaveCount(0);
});
