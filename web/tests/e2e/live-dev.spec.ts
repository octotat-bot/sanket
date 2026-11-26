// DEV SERVER ONLY (run with SANKET_DEV_SERVER=1 against `next dev -p 3100`): the reference-clip
// panel appears on the Live page, plays a real clip, and switches words. The production suite
// checks the opposite (live.spec.ts: the panel is absent from the build).
import { expect, test } from "@playwright/test";

test("dev: reference panel plays the chosen word's INCLUDE clip", async ({ page, request }) => {
  test.skip(process.env.SANKET_DEV_SERVER !== "1", "needs `next dev` and `make reference-clips`");
  const panel = page.getByTestId("sanket-dev-reference-panel");
  await page.goto("/");
  await expect(panel).toBeVisible({ timeout: 120_000 });
  await expect(panel).toContainText("INCLUDE (CC BY 4.0)");
  await page.getByLabel("Learn a sign").selectOption("hello");
  const playing = () =>
    panel.locator("video").evaluate((v: HTMLVideoElement) => v.readyState >= 2 && v.videoWidth === 854 && v.loop);
  await expect.poll(playing, { timeout: 30_000 }).toBe(true);
  expect((await request.get("/dev/reference/hello")).headers()["content-type"]).toBe("video/mp4");
  expect((await request.get("/dev/reference/..%2F..%2Fetc")).status()).toBe(404);
});
