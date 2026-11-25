# ADR 007: Offline extraction at 480 rows, matching what the browser sees

- Status: **Proposed** (Mukund asked for the measurement, 2026-11-25; awaiting approval)
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

## Decision (proposed)

**Option 3: extract at 480 rows** (`extract_height: 480` in `ml/configs/extract.yaml`). INCLUDE's
1920×1080 frames become **854×480**, the same size as the ADR 002 run 3 benchmark copy. The
processed size is stored per video as `frame_size` and used by normalization.

## Tradeoff and dependencies

- **This couples training to the app's camera height.** If Phase 5 changes the camera request
  (for example to 1280×720), either the app resizes frames to 480 rows before MediaPipe, or this
  ADR is revisited and the data re-extracted. Recommended: the app always feeds MediaPipe
  480-row frames, whatever the camera gives, so serving stays matched by construction.
- `INTER_AREA` downscaling of studio 1080p is not the same as a webcam's own sensor and ISP
  (noise, sharpening, compression). The resolution gap is closed. The *camera* gap isn't, and only
  the Phase 6 real-webcam test set measures it.
- Fewer pixels give slightly coarser finger landmarks, but the measurement shows detection
  unchanged and differences at the jitter level.
- Aspect ratio differs (854×480 is 16:9, the camera is 4:3). Normalization handles that
  (`ml/tests/test_normalize.py`).
