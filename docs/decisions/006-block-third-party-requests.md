# ADR 006: Block all third-party network requests with a Content-Security-Policy

- Status: **Proposed** (implemented on branch `phase-1`, awaiting Mukund's approval)
- Date: 2026-11-24

## Context

The privacy promise is that no video or keypoints leave the device, verified by a network
check in Playwright (CLAUDE.md section 8).

While building the ADR 002 benchmark page, that check failed. **`@mediapipe/tasks-vision`
1.2.3 POSTs to `https://odml.pa.googleapis.com/v1/log`** from every task (every 60 s and on
`close()`). Reading the bundle (`vision_bundle.mjs`, logger class constructed in each task's
`createFromOptions`), the payload is a protobuf of usage metrics only:

- task type (for example HolisticLandmarker), running mode
- OS family (parsed from the user agent), library version `1.2.3`, host environment
- init time, call count, total and max latency, dropped-frame count

It contains no images, landmarks, or results. But it is still traffic from our page to a third
party that the user didn't agree to. The typings declare `enableLogging()` and offer no
documented way to turn logging off. The Python wheel has no such endpoint (checked with
`strings` on `libmediapipe.dylib`), and Python only runs on our own machine anyway.

## Options

1. Accept it and soften the privacy statement ("no video or keypoints leave the device").
2. Patch or monkey-patch the minified bundle to drop the logger. Fragile, and it breaks on
   every upgrade.
3. **Send `Content-Security-Policy: connect-src 'self'`** for every route. The browser then
   refuses fetch/XHR/WebSocket to any other origin, whatever a library tries to do.

## Decision (proposed)

Option 3, in `web/next.config.ts`. The logger's send fails and the library handles that by
stopping its timer. Landmark detection is unaffected: both e2e tests pass with the header on.
`web/tests/e2e/extractor-bench.spec.ts` asserts that the header is present and that **no request
to another origin completes**. The privacy statement can then say "this page cannot send data
to any other server", backed by a browser-enforced rule and a test.

## Tradeoff

- Any future third-party fetch (an analytics script, a CDN font) is blocked until it is added
  to the policy on purpose. That is the intent.
- The header must also be set by the host. Next.js `headers()` covers `next start` and Vercel.
  Phase 8 extends the same header list with COOP/COEP for WASM threads.
- Blocked requests may log a CSP violation in the browser console. That is harmless, and
  honest about what's happening.
