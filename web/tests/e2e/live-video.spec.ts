// Phase 5 check: the real exported model recognises a real INCLUDE *test-split* sign fed through
// Chromium's fake camera (a .y4m made from the video, padded with still frames before and after).
// Local-only until CI builds the .y4m: skipped when the file is missing. The video never leaves
// ml/.cache (gitignored).
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";

const y4m = resolve(__dirname, "../../../ml/.cache/live/bird_test.y4m");

test.use({
  launchOptions: {
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${y4m}`],
  },
});

test("the shipped model recognises an INCLUDE test-split sign through the Live page", async ({ page }) => {
  test.skip(!existsSync(y4m) || !existsSync(resolve(__dirname, "../../public/models/include50.labels.json")), "needs the .y4m and the exported model");
  page.on("console", (m) => m.type() === "error" && console.log("[page]", m.text()));
  await page.goto("/");
  await page.getByRole("button", { name: "Start camera" }).click({ timeout: 90_000 });
  await page.getByRole("button", { name: "Show performance" }).click();
  // The history strip lists accepted words; the clip loops, so wait for the first one.
  const history = page.getByLabel("Recently recognised words").locator("li");
  await expect(history.first()).toBeVisible({ timeout: 60_000 });
  const words = await history.allInnerTexts();
  console.log("recognised:", words, "|", (await page.getByTestId("debug").innerText()).replace(/\n/g, " "));
  expect(words[0]).toBe("Bird");
});
