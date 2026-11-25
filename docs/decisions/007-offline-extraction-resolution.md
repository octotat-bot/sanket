# ADR 007: Offline extraction at 480 rows, matching what the browser sees

- Status: Accepted (Mukund, 2026-11-25): extract at 480 rows, and the app always resizes camera
  frames to 480 rows before MediaPipe, downscaling the same way as Python (below).
- Date: 2026-11-25

## Context

INCLUDE videos are 1920×1080. The live app's camera is requested at 640×480, so MediaPipe in the
browser sees **480 rows**. If training keypoints come from 1080p frames and live keypoints
from 480-row frames, the model trains on slightly different inputs than it serves on. That is a
form of training/serving skew.

The Python-vs-browser parity test (`web/tests/e2e/extractor-parity.spec.ts`) shows identical
keypoints for identical pixels. So "what the browser sees" for a 480-row camera is MediaPipe on
480-row frames, and the question becomes: how far is native-1080p extraction from 480-row extraction?

## Measurement

`python -m sanket.checks.extraction_resolution` → `reports/phase1_extraction_resolution.md`.
First 20 `extractor_bench` videos, same decoded frames, downscaled with `INTER_AREA` (the
pipeline's own `resize_to_height`). Distances are mean landmark distance after normalization
(aspect-correct, shoulder-centred, shoulder-width units; ADR 002 / `normalize.py`), only on frames
where both runs detect the part, compared with the 480-row run.

| Holistic, extracted at | hand rate | hands vs 480: median / p90 | upper body vs 480: median / p90 |
|---|---|---|---|
| native 1080p | 87.3% | 0.010 / 0.025 | 0.010 / 0.019 |
| 720 rows | 87.2% | 0.008 / 0.022 | 0.008 / 0.018 |
| **480 rows** | 87.7% | reference | reference |

| Hand+Pose, extracted at | hand rate | hands vs 480: median / p90 | upper body vs 480: median / p90 |
|---|---|---|---|
| native 1080p | 82.3% | 0.009 / 0.022 | 0.009 / 0.017 |
| 720 rows | 81.4% | 0.006 / 0.018 | 0.008 / 0.015 |
| **480 rows** | 83.6% | reference | reference |

Reading:

- **Detection doesn't depend on resolution** in this range: Holistic 87.2–87.7%, Hand+Pose
  81.4–83.6%.
- **Keypoints shift a little.** Native vs 480 differs by about 0.010 shoulder widths (median,
  p90 0.025). That is the same order as MediaPipe's own frame-to-frame tracking jitter in the fps
  ablation (p90 0.016–0.022). Small, but systematic in the sense that it is avoidable.
- 720 rows sits in between, as expected.

## Options

1. **Native 1080p:** the most pixels per hand, but the furthest from serving conditions.
2. **720 rows:** a compromise, matching nothing in particular.
3. **480 rows, the same height as the live camera:** closest to what the browser sees (zero
   resolution gap by construction), no loss in detection, and less work per frame.

## Decision

**Option 3: extract at 480 rows** (`extract_height: 480` in `ml/configs/extract.yaml`). INCLUDE's
1920×1080 frames become **854×480**, the same size as the ADR 002 run 3 benchmark copy. The
processed size is stored per video as `frame_size` and used by normalization.

## Tradeoff and dependencies

- **This couples training to the app's camera height.** If Phase 5 changes the camera request
  (for example to 1280×720), the app still resizes frames to 480 rows before MediaPipe. **Decided: the app always feeds
  MediaPipe 480-row frames, whatever the camera gives**, so serving stays matched by construction.
- `INTER_AREA` downscaling of studio 1080p is not the same as a webcam's own sensor and ISP
  (noise, sharpening, compression). The resolution gap is closed. The *camera* gap isn't, and only
  the Phase 6 real-webcam test set measures it.
- Fewer pixels give slightly coarser finger landmarks, but the measurement shows detection
  unchanged and differences at the jitter level.
- Aspect ratio differs (854×480 is 16:9, the camera is 4:3). Normalization handles that
  (`ml/tests/test_normalize.py`).

## Python and the browser downscale the same way (added 2026-11-25, Mukund)

**Offline:** `cv2.resize(..., interpolation=cv2.INTER_AREA)` (`resize_to_height` in
`sanket.pipeline.extract`). It averages all source pixels covering each output pixel.

**Browser candidates**, measured by `node web/scripts/downscale-probe.mjs` →
`reports/phase1_browser_downscale.json`. Each downscales the parity fixture's 83 native INCLUDE
frames to 854×480. We compare pixels with OpenCV's INTER_AREA frames, then run Holistic (CPU) and
compare hand landmarks with the Python 480-row reference:

| Method | Pixel MAE (0–255), Chromium / Chrome | Hand presence agreement | Landmark dist. median / p95 |
|---|---|---|---|
| **`createImageBitmap`, `resizeQuality: "high"`** (= "medium") | **0.29 / 0.29** | 96.4% | 0.00098 / 0.0070 |
| `createImageBitmap`, "low" | 0.445 / 0.445 | 97.0% | 0.00091 / 0.0094 |
| canvas `drawImage`, smoothing "high" | 0.29 / **0.407** | 96.4% / 98.2% | browser-dependent |
| canvas, smoothing "low" | 0.543 / 0.366 | 97.0% / 97.6% | browser-dependent |
| no smoothing / "pixelated" | 0.92–1.48 | 94.0–96.4% | ≈ 0.0015–0.0022 / 0.009–0.013 |

**Decision: `createImageBitmap(source, {resizeWidth, resizeHeight, resizeQuality: "high"})`**
(`web/src/lib/frames/downscale.ts`). It is closest to INTER_AREA in pixels, and identical in
Playwright's Chromium and Google Chrome. The canvas path gave different pixels in the two browsers
(likely GPU vs CPU canvas), which is a skew we would not control.

**Same output size on both sides:** spec `downscale.width_rule`: even width, halves round up.
Python's `round()` rounds halves to even, so `target_size` uses `floor(x + 0.5)` instead. Shared
cases in `spec/keypoints.json` are tested in both `ml/tests/test_extract_pipeline.py` and
`web/tests/unit/downscale.test.ts` (1282×960 → 642×480 is the half-way case).

**In parity test 4** (`web/tests/e2e/extractor-parity.spec.ts`, mode `480`): the same native frames
are downscaled in Python (INTER_AREA) and in the browser (createImageBitmap "high"), then run through
both extractors. Measured: Holistic hand rate 0.880 vs 0.867, presence agreement 96.4%, landmark
distance median 0.00098 / p95 0.0070 (normalized). Hand+Pose: presence 94.0%, median 0.00044 /
p95 0.0060. Tolerances: Holistic keeps the 95% presence bar. Hand+Pose's is 92%, because its palm
detector flips on marginal frames (the same weakness as in ADR 002). Landmarks: median < 0.002,
p95 < 0.01 for both.

Residual: the two resizers are close but not identical (MAE 0.29 / 255). That adds a small
amount of noise (presence 96.4% vs 99.4% at native, Holistic).

### Systematic offset: ⚠ flagged for Holistic (small)

Parity test 4 also reports the **mean signed offset, browser − Python**, of hand landmarks, x and y
separately, in **shoulder widths**. x is made aspect-correct with the processed frame size first.
Per frame, the offset is the mean over the 21 landmarks of each hand present in both runs. The 95% CI
is a **moving-block bootstrap** (blocks of 10 frames, 2,000 resamples, seed 0), because consecutive
frames are correlated. Output: `reports/parity_offsets/*.json`, from
`web/tests/e2e/extractor-parity.spec.ts`. One INCLUDE-50 test video (`MVI_2988`), 83 frames.

| Extractor, mode | frames | x: mean [95% CI] | y: mean [95% CI] |
|---|---|---|---|
| **Holistic, 480 rows** (each side downscales) | 71 | **+0.00076 [+0.00007, +0.00141]** ⚠ | **−0.00171 [−0.00323, −0.00060]** ⚠ |
| Holistic, native (identical pixels) | 73 | **−0.00060 [−0.00146, −0.00011]** ⚠ | −0.00061 [−0.00111, +0.00052] |
| Hand+Pose, 480 rows | 53 | +0.00017 [−0.00033, +0.00090] | +0.00231 [−0.00073, +0.00802] |
| Hand+Pose, native | 57 | −0.00002 [−0.00010, +0.00000] | +0.00008 [−0.00005, +0.00030] |

**Flag (per Mukund's rule: the CI excludes zero).** Holistic at 480 rows has a systematic offset in
both x and y. Hand+Pose has none measured, at native or at 480.

Context, so the flag is read at the right size:

- **Magnitude:** at most 0.0017 shoulder widths, about 0.6 mm for a 35–40 cm adult shoulder width
  (approximate). That is 6–17% of the native-vs-480 resolution difference (median 0.010) and below
  MediaPipe's frame-to-frame tracking jitter (median 0.002–0.005). Phase 2's translation and jitter
  augmentations will be orders of magnitude larger.
- **Not only the resizer.** Holistic shows an x offset (−0.0006) even on **identical pixels**, so part
  of it comes from the runtimes (browser WASM vs Python's native build), not from the downscale. The
  x offset even changes sign between native and 480. These are small numbers that a single video can't
  pin down.
- **Weak evidence.** One video, 71–73 frames with hands, and 10-frame blocks, so only about 7
  independent blocks. Eight CIs were computed, so about 0.4 false "excludes zero" are expected by
  chance alone. The Holistic pattern is consistent enough to flag, but not to estimate precisely.

Action: no correction is applied (it is far below any augmentation, and the estimate is too
uncertain to correct with). **Phase 4 parity test 4 repeats this on several videos**, and if the
offset holds at a meaningful size (for example > 0.005 shoulder widths) it gets investigated.
Until then, Holistic's browser-vs-Python agreement is "systematic offset measured, ≤ 0.002
shoulder widths", never "no offset".
