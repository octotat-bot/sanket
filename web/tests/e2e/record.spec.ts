// The Phase 6 recording tool: consent gating, a full record → accept → save cycle with Chromium's
// fake camera, the saved file's format and privacy, and that frames are never mirrored.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { type RecordingSpec, validateSession } from "../../src/lib/record/session";

const root = resolve(__dirname, "../../..");
const recSpec = JSON.parse(readFileSync(resolve(root, "spec/recording_v1.json"), "utf8")) as RecordingSpec;

test("frames reach MediaPipe unmirrored (camera size and downscaled)", async ({ page }) => {
  await page.goto("/dev/record");
  await page.waitForFunction(() => "__sanketFrameForMediaPipe" in window);
  for (const [w, h] of [
    [640, 480],
    [1280, 720],
  ]) {
    const r = await page.evaluate(
      async ([w, h]) => {
        const c = new OffscreenCanvas(w, h);
        const g = c.getContext("2d")!;
        g.fillStyle = "black";
        g.fillRect(0, 0, w, h);
        g.fillStyle = "white";
        g.fillRect(0, 0, w / 2, h); // left half white
        const src = c.transferToImageBitmap();
        const fn = (window as unknown as {
          __sanketFrameForMediaPipe: (s: ImageBitmap, w: number, h: number) => Promise<{ bitmap: ImageBitmap; width: number; height: number }>;
        }).__sanketFrameForMediaPipe;
        const out = await fn(src, w, h);
        const k = new OffscreenCanvas(out.width, out.height).getContext("2d")!;
        k.drawImage(out.bitmap, 0, 0);
        const px = (x: number) => k.getImageData(x, Math.floor(out.height / 2), 1, 1).data[0];
        return { size: [out.width, out.height], left: px(5), right: px(out.width - 5) };
      },
      [w, h],
    );
    expect(r.size[1]).toBe(480); // always 480 rows into MediaPipe (ADR 007)
    expect(r.left).toBeGreaterThan(200); // left stays left: not mirrored
    expect(r.right).toBeLessThan(50);
  }
});

test("consent first, then record, accept and save a keypoints-only session file", async ({ page, baseURL }) => {
  test.setTimeout(150_000);
  const completedForeign: string[] = [];
  page.on("requestfinished", (req) => {
    const u = new URL(req.url());
    if (u.origin !== new URL(baseURL!).origin && !["blob:", "data:"].includes(u.protocol)) completedForeign.push(req.url());
  });

  await page.goto("/dev/record");
  const start = page.getByRole("button", { name: "Agree and start the camera" });
  await expect(start).toBeDisabled();
  expect(await page.evaluate(() => document.querySelector("video")?.srcObject ?? null)).toBeNull(); // no camera yet
  await page.getByLabel("Takes per word").fill("1");
  await page.getByLabel("No-sign clips").fill("1");
  await page.getByLabel("Seconds per clip").fill("2");
  await page.getByLabel("I have read this and agree to take part").check();
  await start.click();

  await expect(page.getByRole("status")).toContainText("ready", { timeout: 120_000 });
  await expect(page.getByRole("status")).toContainText(/Participant code: P-[A-Z2-7]{6}/);
  await page.getByRole("button", { name: "Start the session" }).click();
  await expect(page.getByTestId("prompt")).toBeVisible();
  await expect(page.getByText(/Clip 1 of 51/)).toBeVisible(); // 50 words × 1 take + 1 no-sign

  await page.getByRole("button", { name: "Record (Space)" }).click();
  await expect(page.getByTestId("quality")).toBeVisible({ timeout: 20_000 });
  await page.getByRole("button", { name: "Accept (Enter)" }).click();
  await expect(page.getByText(/Clip 2 of 51/)).toBeVisible();

  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: /Save session file \(1 clips\)/ }).click();
  const file = JSON.parse(readFileSync((await (await download).path())!, "utf8"));

  expect(validateSession(file, recSpec)).toEqual([]);
  expect(file.clips).toHaveLength(1);
  const clip = file.clips[0];
  expect(clip.frame_size).toEqual([640, 480]); // processed size; the fake camera is already 480 rows
  expect(clip.t_ms.length).toBeGreaterThan(10);
  expect(clip.t_ms[0]).toBe(0);
  expect(file.consent.version).toBe("1");
  expect(file.consent.text_sha256).toMatch(/^[0-9a-f]{64}$/);
  expect(JSON.stringify(file)).not.toMatch(/data:image|data:video|base64/);
  expect(completedForeign).toEqual([]); // nothing left the device
});
