# ADR 009: Live recognition: sign spotting, rejection, and model runtime

Status: accepted (2026-11-26, demo track; thresholds provisional until Phase 6 data)

## Context

The Live page sees an endless camera stream, but the model classifies one isolated sign resampled
to T = 64 frames (ADR 008). Something has to decide where a sign starts and ends, the app must
not show a confident wrong word, and the model has to run in a browser.

## Decisions

**Sign spotting: a motion state machine** (`web/src/lib/live/spotter.ts`).
`idle → signing` when a hand is raised (wrist less than 1.6 shoulder widths below the shoulder
line) and its smoothed speed is ≥ 0.8 sw/s for 3 frames. `signing → cooldown` when the hands are
lowered for 6 frames, or are still (< 0.25 sw/s) for 600 ms, or after 4 s. Clips shorter than
500 ms are dropped; 250 ms of pre-roll is kept; 400 ms cooldown. Everything is in body units
(shoulder midpoint, shoulder width, aspect-corrected x), so the thresholds hold at any distance and
any camera aspect ratio (unit-tested with synthetic motion at two scales and two aspects).
Options considered: a sliding window classified continuously (needs a trained "no sign" class to
avoid spamming words, and costs a model run per frame); a learned boundary detector (no labelled
boundary data). The state machine is explainable, cheap, and the training augmentation already
includes temporal crop for imprecise boundaries.

**Rejection: threshold on calibrated probabilities, not a "no sign" class** (`decide.ts`).
Show "Not sure" unless top-1 p ≥ 0.45 and p1 − p2 ≥ 0.15. Probabilities use a temperature fitted
on INCLUDE-50 **val** when available (`calibration` in `<name>.labels.json`; 1.0 until then).
Why not an idle class: INCLUDE has no non-sign clips, and the Phase 6 recordings may be used
*only* to measure accuracy (consent form), so they can't train one. The spotter already filters
most non-sign movement (it needs raised, moving hands).

**Runtime and model format** (`model.worker.ts`). onnxruntime-web in a Web Worker. WebGPU (fp32
model) when `navigator.gpu` gives an adapter, else WebAssembly (dynamic-int8 model, SIMD, threads
when cross-origin isolated). fp32 on WebGPU because its kernels don't cover the int8 ops
(MatMulInteger / DynamicQuantizeLinear) or GRU; int8 on WASM because it is smaller and CPU-friendly.
The WebGPU build of the runtime is 7.1 MB gzipped vs 3.8 MB for the WASM build, so it is loaded
only when WebGPU is available. The model and the runtime files are served from our own origin.

**Guards.** The worker refuses a model whose labels hash differs from the app's vocabulary
(`spec/vocab_include50.json`), whose file sha256 differs from its metadata, or whose keypoint
spec version differs from the app's (training/serving skew).

## Tradeoffs and open questions

- The thresholds are hand-set; tune them on the real-webcam recordings (Phase 6) and report the
  values used. A signer who keeps their hands raised between signs ends signs by stillness only.
- Two signs without a pause merge into one clip (capped at 4 s).
