// The ADR 002 benchmark page works end to end: real Next.js page, Chromium fake camera,
// CPU delegates only (headless CI has no GPU). It also checks the privacy rule: every
// request stays on our own origin.
import { expect, test } from "@playwright/test";

interface Result {
  id: string;
  ok: boolean;
  error?: string;
  processedFrames?: number;
  msP50?: number;
}
interface Report {
  videoSize: [number, number];
  models: { name: string; verified: boolean }[];
  results: Result[];
}

test("extractor benchmark runs Holistic and Hand+Pose on the camera", async ({
  page,
  baseURL,
}) => {
  const isForeign = (u: string) => {
    const url = new URL(u);
    return url.origin !== new URL(baseURL!).origin && !["blob:", "data:"].includes(url.protocol);
  };
  const completedForeign: string[] = [];
  page.on("requestfinished", (req) => {
    if (isForeign(req.url())) completedForeign.push(req.url());
  });

  const response = await page.goto("/dev/extractor-bench");
  expect(response?.headers()["content-security-policy"]).toContain("connect-src 'self'");
  await page.getByLabel("A · Holistic · GPU").uncheck();
  await page.getByLabel("B · Hand + Pose · GPU").uncheck();
  await page.getByLabel("Seconds each").fill("2");
  await page.getByLabel("Warm-up").fill("0.5");
  await page.getByRole("button", { name: "Run benchmark" }).click();

  await expect(page.getByRole("status")).toHaveText("Done.", { timeout: 90_000 });
  const report = (await page.evaluate(
    () => (window as unknown as { __benchReport: Report }).__benchReport,
  )) as Report;

  expect(report.videoSize).toEqual([640, 480]);
  expect(report.models.every((m) => m.verified)).toBe(true);
  expect(report.results.map((r) => r.id)).toEqual(["holistic-cpu", "handpose-cpu"]);
  for (const r of report.results) {
    expect(r.ok, r.error).toBe(true);
    expect(r.processedFrames).toBeGreaterThan(5);
    expect(r.msP50).toBeGreaterThan(0);
  }
  // Nothing reached another origin (MediaPipe's usage logger is blocked by the CSP).
  expect(completedForeign).toEqual([]);
});
