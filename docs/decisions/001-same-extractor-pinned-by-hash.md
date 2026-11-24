# ADR 001: One keypoint extractor, pinned by library version and model sha256

- Status: Accepted (Phase 0)
- Date: 2026-11-24

## Context

The model is trained on keypoints extracted offline in Python and served on keypoints extracted
live in the browser. If the two extractors differ (library version, model weights), the
inputs shift and accuracy drops silently. This is called training/serving skew.

Phase 0 found:

- `mediapipe` (PyPI) and `@mediapipe/tasks-vision` (npm) are both at **1.2.3** and are
  released together.
- Both expose `HolisticLandmarker`, `HandLandmarker`, and `PoseLandmarker`.
- The model files (`.task`) are separate downloads from `storage.googleapis.com`. Each has a
  versioned URL (`.../float16/1/...`) and a `latest` URL.
- **`latest` is not stable.** For `pose_landmarker_full`, the bytes at `latest` differ from
  version `1` (sha256 `4eaa5e…` vs `5134a3…`), with the same file size.

## Options

1. Pin only the library version and load models from `latest`.
2. Pin the library version and the model by versioned URL.
3. Pin the library version and the model by versioned URL **and sha256**. Self-host the
   exact bytes for the web app.

## Decision

Option 3. `spec/keypoints.json` records the MediaPipe version and, for each model, its
versioned URL, byte size, and sha256. `make models` downloads each model, refuses a hash
mismatch, and copies the same bytes to `web/public/models/mediapipe/`. Tests on both sides
fail if `pyproject.toml`, `web/package.json`, or the installed packages disagree with the spec.

## Tradeoff

- We must bump versions deliberately: change the spec, re-extract the keypoints, retrain.
  That is the point.
- Self-hosting adds about 14 to 31 MB of static assets (depending on which models ship). It
  removes a third-party CDN from the runtime path, which also helps the cross-origin isolation
  headers (COOP/COEP) needed later.
- The same model bytes do **not** guarantee identical numbers. Python runs the TFLite CPU
  path. The browser runs WASM (CPU) or WebGL (GPU). Phase 4 parity test 4 measures the
  actual keypoint difference on the same video.
