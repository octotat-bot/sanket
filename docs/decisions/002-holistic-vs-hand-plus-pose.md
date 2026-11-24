# ADR 002: Holistic Landmarker vs separate Hand + Pose Landmarkers

- Status: **Measure (a) done. Measure (b) waits on Mukund's browser run. No choice yet, and no
  full extraction until Mukund approves.**
- Date: 2026-11-24

## Context

We need upper-body pose plus 21 landmarks per hand, per frame, identically in Python (training
data) and the browser (live). Both options work in both runtimes at MediaPipe 1.2.3
(`ml/tests/test_mediapipe_smoke.py`, `web/tests/e2e/mediapipe-smoke.spec.ts`).

- **A. `HolisticLandmarker`** (`holistic_landmarker.task`, 13.7 MB). One graph: pose + face
  mesh + both hands. The face mesh can't be switched off, but blendshapes and segmentation
  masks can, and they are.
- **B. `HandLandmarker(numHands=2)` + `PoseLandmarker`** (`hand_landmarker.task` 7.8 MB +
  `pose_landmarker_full.task` 9.4 MB). Two graphs per frame. Hands aren't tied to a body.

Mukund's criteria (2026-11-24). Decide on two measures:

- **(a)** hand-detection rate and left/right swap rate on about 50 INCLUDE videos, in Python;
- **(b)** fps in a real browser on Mukund's laptop with its GPU (not headless Chromium).

## Measure (a): results

Script: `python -m sanket.checks.extractor_compare` (`make check-extractor`). Report:
`reports/phase1_extractor_compare.md` / `.json`. 50 seeded random INCLUDE videos (`extractor_bench`
in `ml/data/samples/phase1_checks.json`), 3,427 frames, all 1920×1080 at 25 fps. Both extractors
get the **same decoded frames and timestamps** (OpenCV/FFmpeg, unmirrored, RGB), in VIDEO mode
with default thresholds. Metric definitions are in `sanket.keypoints.hand_tracks`, with
crafted-case tests:

- *Body agreement*: a hand's label matches the pose wrist (landmark 15 = person's left,
  16 = right) it is clearly nearer to (near < 0.6 × far).
- *Swap*: between consecutive frames, the labels exchange although the hands didn't (crossing
  cost < 0.5 × keeping cost), or a single, barely moving hand changes label.

| Metric (pooled over frames) | A: Holistic | B: Hand+Pose |
|---|---|---|
| Frames with any hand | 85.8% | 84.4% |
| Frames with both hands | 57.7% | 53.7% |
| Label agrees with the body | 99.3% (of 4,901) | 98.9% (of 4,641) |
| **Left/right swaps per 100 judged transitions** | **0.00** (0 of 2,420; 0 videos) | **1.08** (26 of 2,411; 16 of 50 videos; 95% CI 0.54–1.76) |
| Videos with no hand detected at all | 0 | 0 |
| Python CPU time per frame (info only, not a criterion) | 21.5 ms | 37.9 ms |

How to read it honestly:

- **Swaps clearly favour A.** Holistic produced no label swap in any of the 50 videos. Hand+Pose
  swapped in 16 videos. With B we would need the position-continuity swap fixer from CLAUDE.md
  section 8 on every sequence. With A it would only be a safety net.
- **Detection leans towards A but is not significant.** Per video, A detects any hand more often
  in 26 videos and less often in 7 (both hands: 36 vs 12). But a video-level bootstrap 95% CI for
  the pooled difference includes zero (any hand −1.7 to +4.4 points, both hands −1.1 to +8.3).
- **Body agreement is about 99% for both**, but it can't separate them fairly. A's hands come
  from its own pose, so A agrees partly by construction.
- **A correction to the earlier draft of this ADR.** I had flagged that HandLandmarker might label
  hands as if the image were mirrored. Measured on unmirrored INCLUDE frames, B's raw labels match
  the person's own left/right 98.9% of the time (flipping them gives 1.1%), so no inversion is
  needed at 1.2.3.
- About 14% of frames have no hand in either method. These are mostly the start and end of clips,
  with hands lowered. This is Phase 1's per-video detection log's job, not a difference between
  A and B.

## Measure (b): browser fps (waiting for Mukund's run)

Page: `/dev/extractor-bench` (`make bench-page`, then open
http://localhost:3000/dev/extractor-bench in Chrome). It runs A and B on the GPU and the CPU
delegate, on live 640×480 camera frames. Each `.task` file is checked against
`spec/keypoints.json` by sha256 in the browser before use. For each configuration it measures
ms/frame (p50, p95), processed fps vs camera fps, and the share of frames with a hand. Results
are saved as a local JSON file. The page cannot send anything anywhere (ADR 006).

Decision rule, fixed before seeing the numbers. "Sustains 30 fps" means p95 ≤ 33.3 ms **and**
processed fps ≥ 95% of camera fps, with hands in view in > 80% of frames.

1. If **A on GPU** sustains 30 fps, choose **A**. It wins on swaps and matches or beats B on
   detection.
2. If A doesn't but **B on GPU** does, the tradeoff is real. Bring both numbers back to Mukund
   and don't decide alone.
3. If neither does, choose the faster one, and Phase 5 handles it with adaptive frame skipping
   on timestamps. Report it.

| Configuration | p50 ms | p95 ms | processed / camera fps | frames with hand | ≥ 30 fps? |
|---|---|---|---|---|---|
| A · Holistic · GPU | _pending_ | | | | |
| B · Hand+Pose · GPU | _pending_ | | | | |
| A · Holistic · CPU | _pending_ | | | | |
| B · Hand+Pose · CPU | _pending_ | | | | |

Device: _pending (from the saved JSON)_

## Decision

_Pending measure (b) and Mukund's approval._ On measure (a) alone, A is preferred.
