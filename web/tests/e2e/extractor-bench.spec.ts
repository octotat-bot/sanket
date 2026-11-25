// The ADR 002 benchmark page works end to end: real Next.js page, Chromium fake camera,
// CPU delegates only (headless CI has no GPU). It also checks the privacy rule: every
// request stays on our own origin.
import { expect, test } from "@playwright/test";

interface Result {
  id: string;
  source: string;
  ok: boolean;
  error?: string;
  processedFrames?: number;
  msP50?: number;
}
interface Report {
  videoSizes: Record<string, [number, number]>;
  device: { crossOriginIsolated: boolean };
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
  const headers = response?.headers() ?? {};
  expect(headers["content-security-policy"]).toContain("connect-src 'self'");
  expect(headers["cross-origin-opener-policy"]).toBe("same-origin");
  expect(headers["cross-origin-embedder-policy"]).toBe("require-corp");
  await page.getByLabel("Camera (640×480 requested)").check();
  await page.getByLabel("A · Holistic · GPU").uncheck();
  await page.getByLabel("B · Hand + Pose · GPU").uncheck();
  await page.getByLabel("Seconds each").fill("2");
  await page.getByLabel("Warm-up").fill("0.5");
  await page.getByLabel("Rounds").fill("1");
  await page.getByRole("button", { name: "Run benchmark" }).click();

  await expect(page.getByRole("status")).toHaveText("Done.", { timeout: 90_000 });
  const report = (await page.evaluate(
    () => (window as unknown as { __benchReport: Report }).__benchReport,
  )) as Report;

  expect(report.videoSizes).toEqual({ camera: [640, 480] });
  expect(report.device.crossOriginIsolated).toBe(true); // COOP/COEP headers are served
  expect(report.models.every((m) => m.verified)).toBe(true);
  expect(report.results.map((r) => r.id).sort()).toEqual(["handpose-cpu", "holistic-cpu"]);
  for (const r of report.results) {
    expect(r.ok, r.error).toBe(true);
    expect(r.processedFrames).toBeGreaterThan(5);
    expect(r.msP50).toBeGreaterThan(0);
  }
  // Nothing reached another origin (MediaPipe's usage logger is blocked by the CSP).
  expect(completedForeign).toEqual([]);
});
