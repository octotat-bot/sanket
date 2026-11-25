# ADR 002: Holistic Landmarker vs separate Hand + Pose Landmarkers

- Status: **Decision rule approved by Mukund (2026-11-24).** Measure (a) done. Measure (b) run 1 was
  invalid (hand rate < 80%). Investigation done (2026-11-25). Waiting on Mukund's rerun (file +
  camera) and on his decision about the proposed 15 fps rule below. No extraction until then.
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

## Measure (b): browser fps

Page: `/dev/extractor-bench` (`make bench-page`, then open
http://localhost:3000/dev/extractor-bench in Chrome). Each `.task` file is checked against
`spec/keypoints.json` by sha256 in the browser before use. Results are saved as a local JSON
file. The page cannot send anything anywhere (ADR 006).

### Run 1 (2026-11-24, Mukund's laptop): INVALID

Apple M3 Pro, Chrome 157, camera 640×480 at 30 fps. Raw JSON:
`reports/bench/2026-11-24T06-55-50Z-run1-INVALID.json`.

| Configuration | p50 ms | p95 ms | processed / camera fps | frames with hand |
|---|---|---|---|---|
| A · Holistic · GPU | 133.9 | 191.8 | 7.4 / 29.7 | 57% |
| B · Hand+Pose · GPU | 26.2 | 55.3 | 27.5 / 30.0 | 9% |
| A · Holistic · CPU | 53.4 | 64.0 | 19.8 / 30.0 | 70% |
| B · Hand+Pose · CPU | 62.5 | 73.1 | 16.4 / 29.9 | 0% |

Mukund ruled the run invalid because hand rate was below 80% in every row, so the rule was not
applied. Two anomalies were investigated before a rerun.

### Investigation 1: why did Hand+Pose find almost no hands? (2026-11-25)

Every code-side cause was ruled out with an experiment, using an INCLUDE-50 test video
(`Animals/4. Bird/MVI_2988.MOV`, 83 frames; Python: Holistic 88.0%, Hand+Pose 68.7%):

| Suspect | Experiment | Result |
|---|---|---|
| Browser MediaPipe differs from Python | Same OpenCV-decoded frames (lossless PNG) and timestamps in both | **Identical** hand rates (88.0 / 68.7%). Presence agrees on 99.4% / 100% of frames, landmark distance median 0.0003 / 0.00008 (normalized). Now a test: `web/tests/e2e/extractor-parity.spec.ts` |
| Browser video decoder vs OpenCV (colour skew) | Browser-decoded frame vs OpenCV frame, per-channel linear fit | browser ≈ 1.000 × OpenCV + ≤ 0.2: **same pixels** |
| Playing `<video>` element as input | INCLUDE video played at 1× through requestVideoFrameCallback | Hand 65.3%, Holistic 85.7% |
| 640-px camera resolution | Frames downscaled to 640×360 | Hand 67.5%, Holistic 86.7% |
| `modelAssetBuffer` vs `modelAssetPath` | Same model loaded both ways | Identical (67.5%) |
| Shared fileset, run 1's config order, float timestamps | Exact bench sequence on the same frames | Hand+Pose 64% (GPU) / 70% (CPU) |
| The bench page itself | **Real page in the new file mode** on the INCLUDE video | Holistic 89.4% (Python 88.0%), Hand+Pose 67.2% (68.7%). Now a test: `web/tests/e2e/extractor-bench-file.spec.ts` |

Conclusion: the code is correct. The 9% / 0% came from the **live scene in run 1**, and the
mechanism is clear. HandLandmarker has to find hands in the *whole frame* with its palm detector.
On a full 1920×1080 INCLUDE frame the hand is about 99 px (5% of the width), and **both Python
and the browser find 0 hands on that frame, but 1 on a crop around the hand.** In video, B only
works once the palm detector catches a hand and tracking takes over. Holistic finds hand regions
from the body pose, so it never depends on the palm detector. In a waist-up webcam view, small,
motion-blurred or edge-of-frame hands can keep the palm detector from ever firing. Run 1's scene
can't be inspected from here, so the bench now draws each configuration's detected hand points
on screen (display only, never stored).

This is itself evidence for A. The app needs waist-up framing (shoulders are the normalization
anchor), and that is exactly where B's palm-detector dependency is weakest.

(Along the way, a bug in my own investigation harness was found and fixed: in headless Chromium,
`seeked` fired without presenting the new frame, so early numbers came from frame 0 repeated.
The bench page doesn't seek, and the committed tests don't depend on seeking.)

### Investigation 2: Holistic GPU slower than Holistic CPU (to be settled by the rerun)

Run 1 measured from 3 s after start in a fixed order. At about 7 fps, that is only about 20
warm-up frames, so start-up cost could have leaked in. The bench now:

- warms up for ≥ 8 s **and** ≥ 60 frames, excluded from timings;
- records init time and the first 10 frame times separately, so shader compilation is visible;
- runs configs in a **seeded random order per round** (default 2 rounds), with the seed and
  order saved in the JSON.

If Holistic GPU is still slower than CPU after that, it is a property of the 1.2.3 GPU path on
Apple Metal, not of the measurement, and the app would use the faster delegate.

### Bench changes for the rerun (Mukund's items 3 and 4)

- **Cross-origin isolated:** COOP `same-origin` + COEP `require-corp` on every route
  (`web/next.config.ts`). The e2e test asserts `crossOriginIsolated === true`.
- **INCLUDE video-file source:** the same local file for every config, rewound to its first
  frame before each, so hands are always in view. **"Both"** runs the file and then the
  camera in one go, and every result row is tagged with its source.

## How many fps does the app need? (proposal, for Mukund's decision)

Evidence: `python -m sanket.checks.fps_sensitivity` → `reports/phase1_fps_sensitivity_holistic.md`.
On the 50 `extractor_bench` videos, Holistic was run on only the frames a slower camera would
deliver, with real timestamps, and compared with the full 25 fps run. Errors are mean hand-landmark
distance in **shoulder widths**.

| fps | hand rate (25 fps run, same frames) | tracking jitter on shared frames, median / p90 | lost motion on skipped frames (linear interp.), median / p90 |
|---|---|---|---|
| 20 | 85.5% (85.9%) | 0.002 / 0.016 | 0.008 / 0.042 |
| 15 | 84.7% (85.8%) | 0.004 / 0.020 | 0.008 / 0.043 |
| 12.5 | 84.6% (85.8%) | 0.004 / 0.021 | 0.008 / 0.040 |
| 10 | 85.2% (85.9%) | 0.005 / 0.022 | 0.010 / 0.053 |

Reading:

- **Tracking does not degrade** down to 10 fps: hand rate stays within about 1 point of the
  25 fps run on the same frames.
- **Little motion is lost.** Between 20 and 12.5 fps the median interpolation error is a flat
  0.008 shoulder widths (about 3 mm for a 35–40 cm adult shoulder width, an approximation), and
  p90 is about 0.04. It starts to grow at 10 fps (p90 0.053). The classifier will also resample
  every sign to a fixed length (Phase 2 ADR), which already treats time this coarsely.
- Limits: this measures keypoint fidelity, not accuracy. A 25 fps source can't give perfectly
  even 20 fps spacing.

**Proposal:** replace "sustains 30 fps" with **"sustains 15 fps"**: p95 ≤ 66.7 ms and processed
fps ≥ 15. It keeps the hand-rate condition (> 80% in camera mode, within 0.15 of Python in file
mode) and judges each extractor on **its faster delegate** (GPU or CPU), since the app would use
whichever is faster. 15 rather than 10 leaves margin before the error starts to grow.

- The rule would be: (1) if A sustains 15 fps in both file and camera mode → A. (2) If only B
  does → bring both to Mukund. (3) If neither → the faster one, with adaptive frame skipping in
  Phase 5.
- **Confirmed in Phase 2 with accuracy:** the baseline is evaluated on INCLUDE-50 test re-extracted
  at 25 / 20 / 15 / 10 fps, and trained with frame-drop and time-stretch augmentation. If
  accuracy at 15 fps is more than 1 point below 25 fps, the floor goes back up.
- Related, not changed: CLAUDE.md section 3's end-to-end target of ≥ 25 fps is a résumé metric.
  If 15 fps is accepted here, that target should be revisited at the same time (for example
  ≥ 15 fps sustained, with 25 as the stretch goal). That is Mukund's call.

## Decision

_Pending Mukund's rerun (file + camera) and his decision on the fps proposal._ On measure (a),
and on the palm-detector finding above, A is preferred.
