# ADR 002: Holistic Landmarker vs separate Hand + Pose Landmarkers

- Status: **Proposed. Decide after the Phase 1 browser benchmark.**
- Date: 2026-11-24

## Context

We need upper-body pose plus 21 landmarks per hand, per frame, identically in Python and the
browser. Both options below are available and verified working in both runtimes at 1.2.3
(`ml/tests/test_mediapipe_smoke.py`, `web/tests/e2e/mediapipe-smoke.spec.ts`).

## Options

**A. `HolisticLandmarker`** (`holistic_landmarker.task`, 13.7 MB)

- + Hands come from the same graph as the body, so left/right should follow the pose and
  reduce the hand swaps CLAUDE.md section 8 warns about. *Expected from the design, not yet
  measured: Phase 1 checks this on crossing-hands clips.*
- + One graph and one timestamp per frame. Simpler to keep in sync.
- − It also runs the face mesh (478 landmarks) and blendshapes, which we don't need. Whether
  that work can be skipped is not verified yet. If not, it costs browser fps.
- − It follows one person only. That is fine for our use, but we need a "multiple people"
  warning by other means.

**B. `HandLandmarker` + `PoseLandmarker`** (7.8 MB + 9.4 MB)

- + No face work. Probably faster, and each part can be tuned (for example `numHands`).
- − Handedness comes from a classifier on the hand crop. It can swap labels. The legacy
  MediaPipe Hands docs say handedness assumes a mirrored (selfie) image. If the Tasks API
  keeps that convention, our **unmirrored** frames get inverted labels. *To verify in Phase 1.*
- − Hands are not tied to a body. With two people in frame, hands from different people can be
  mixed.
- − Two graphs per frame, with two latency budgets.

## Decision (proposed)

Start Phase 1 with a short browser benchmark on a mid-range laptop: fps of A vs B at 640×480
in VIDEO mode, on both CPU and GPU delegates. If A sustains ≥ 30 fps of landmark extraction
(headroom for the ≥ 25 fps end-to-end target), choose **A** for the hand/body association and
less swap handling. Otherwise choose **B** and add the position-continuity swap fix.

## Tradeoff

Choosing late costs nothing now, because nothing is extracted yet. Choosing wrong would cost
a full re-extraction of 57 GB of video, so it is worth measuring first.
