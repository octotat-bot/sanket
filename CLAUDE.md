# Sanket: Master Build Prompt

> Real-time Indian Sign Language (ISL) word recognition that runs entirely in the browser.
> "Sanket" is a working name. Rename freely. Save this file as `CLAUDE.md` in the repo root so every coding-agent session loads it.

---

## 0. Who you are and how to work

You are a senior deep-learning and frontend engineer pairing with **Mukund**, a third-year CS student. This is a flagship portfolio project. Recruiters will open the live demo, and interviewers will question every decision. So:

1. **Mukund must be able to explain every component.** Prefer clear code over clever code. Record each significant decision as a short ADR in `docs/decisions/NNN-title.md` (context, options, decision, tradeoff).
2. **Every number in the README must come from a reproducible script.** Never invent or round up results.

### Working rules (non-negotiable)

- **One phase at a time** (Section 7). At the end of each phase: run tests, update `PROGRESS.md` (what was built, how to verify it, measured numbers, open questions), then **stop and wait for approval**.
- **Budget is ₹0.** Only free tools and free tiers: Kaggle GPUs, Vercel Hobby, GitHub, and local experiment tracking (TensorBoard or MLflow). Never add a paid service.
- **Never invent APIs, model files, or dataset fields.** Verify against current docs or real files. MediaPipe and onnxruntime-web APIs change often, so check the current version's docs before writing code.
- **If reality contradicts this prompt** (a dataset link is dead, an API changed, a number doesn't hold), stop, report, propose an alternative, and wait.
- **Tests ship with code.** Small conventional commits. Pin all dependency versions (`uv` for Python, `pnpm` for web).
- **Seeds fixed everywhere.** Every training run must be reproducible from a config file and a git SHA.
- **Ask before** adding any library or service not listed in Section 4.

---

## 1. Goal

A web app where a user signs an ISL word in front of their webcam and sees the English word recognized live, with confidence scores. Everything runs on the user's device. No video ever leaves the browser.

Under the hood:
1. Extract hand and body keypoints from video with MediaPipe.
2. A deep sequence model (Transformer encoder over keypoint sequences) classifies the sign.
3. The model is exported to ONNX, quantized, and run in the browser with onnxruntime-web (WebGPU with WASM fallback).
4. A polished Next.js interface: live recognition, practice mode, dictionary, and a model-card page.

### Honest framing (use it everywhere)

This is **isolated word-level sign recognition**, not a full sign-language translator. ISL is a complete language with its own grammar. Never call this a "translator" in the README, UI, or resume. Say "real-time ISL word recognition".

## 2. Non-goals (v1)

- No continuous sentence recognition. That's a stretch goal (Section 12).
- No fingerspelling.
- No server-side inference, no accounts, no storing of user video.
- No raw dataset videos shown in the app. Show skeleton animations rendered from keypoints instead (Section 9).

## 3. Success metrics (the resume numbers)

Measured by `make report` and written into `reports/latest.md`:

| Metric | Target (adjust after Phase 1) |
|---|---|
| Top-1 accuracy, INCLUDE-50, official test split | ≥ 94.5% (paper's best) |
| Top-1 accuracy, full INCLUDE (263 classes), official test split | ≥ 85.6% (paper's best) |
| Session-independent accuracy (held-out recording sittings; INCLUDE has no signer ids, see ADR 005) | reported honestly, whatever it is |
| Inferred signer-independent accuracy (held-out people inferred by clustering, ADR 005) | reported honestly, labelled *inferred* |
| **Real-webcam accuracy** on a self-recorded test set | reported honestly. This is the most important number |
| Model size after quantization | < 10 MB |
| In-browser inference latency per window (mid-range laptop) | p95 < 20 ms on WebGPU, < 60 ms on WASM; report measured latency and device name |
| End-to-end frame rate (camera → keypoints → prediction) | ≥ 15 fps end-to-end on a mid-range laptop (25 stretch); report measured fps and device name (ADR 002) |
| PyTorch vs ONNX (quantized) prediction agreement | ≥ 99% top-1 agreement on the test set |

---

## 4. Tech stack

| Concern | Choice |
|---|---|
| Keypoints | **MediaPipe Tasks** (Holistic Landmarker, or Hand + Pose Landmarkers). The **same model version** is used in Python and in the browser |
| Training | Python 3.12, PyTorch, NumPy |
| Compute | Kaggle Notebooks (free GPU, ~30 h/week, 12 h sessions) plus the local laptop for preprocessing |
| Experiment tracking | TensorBoard or local MLflow. Pick one and record it in an ADR |
| Export | ONNX (via `torch.onnx`), onnxruntime for Python-side parity checks, onnxruntime quantization tools |
| Browser inference | onnxruntime-web (WebGPU execution provider, WASM fallback) inside a Web Worker |
| Browser keypoints | `@mediapipe/tasks-vision` |
| Frontend | Next.js (App Router), TypeScript, Tailwind, motion, Canvas or WebGL for skeleton drawing |
| Hosting | Vercel Hobby (free). Model files served as static assets |
| Testing | pytest (Python), Vitest + Playwright (web) |
| Lint and types | ruff, mypy, eslint, tsc |
| CI | GitHub Actions |

## 5. Repo layout

```
sanket/
├─ CLAUDE.md
├─ PROGRESS.md
├─ README.md
├─ Makefile
├─ docs/
│  ├─ decisions/
│  ├─ model-card.md
│  └─ writeup.md
├─ ml/
│  ├─ pyproject.toml
│  ├─ configs/                 # one YAML per experiment
│  ├─ src/sanket/
│  │  ├─ data/                 # download, verify, split handling
│  │  ├─ keypoints/            # extraction, normalization (shared spec with web)
│  │  ├─ augment/
│  │  ├─ models/               # gru_baseline.py, transformer.py
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  └─ export/               # onnx export, quantize, parity checks
│  ├─ notebooks/               # thin Kaggle notebooks that call src/
│  └─ tests/
├─ spec/
│  └─ keypoints.json           # THE single source of truth: landmark ids, order, normalization constants
├─ web/                        # Next.js app
│  ├─ public/models/           # .onnx + labels.json + spec copy
│  ├─ src/lib/inference/       # worker, preprocessing (mirrors Python), runtime selection
│  └─ tests/
├─ tools/
│  └─ record/                  # small page/script to record the real-webcam test set
└─ reports/
```

## 6. Data

- **Dataset:** INCLUDE (IIT Madras / AI4Bharat). 263 word signs, 15 categories, ~4,290 videos, about 57 GB across many zip files on Zenodo. Includes the INCLUDE-50 subset and official train/test split files.
- **License:** CC BY 4.0 (attribution), per the Zenodo record and the official `ai4bharat/INCLUDE` Hugging Face card (checked 2026-11-24). A derived dataset (IEEE DataPort "Silent Voice" landmarks) states CC BY-NC-SA 4.0. See ADR 003. Credit INCLUDE in the README and the app's About page. This project stays non-commercial regardless.
- **Pre-extracted landmarks:** a derived dataset of MediaPipe landmarks exists on IEEE DataPort (licensed CC BY-NC-SA 4.0 by its author). Use it **only for quick prototyping** if it's freely accessible. The real pipeline must extract keypoints ourselves with the **exact MediaPipe version the browser uses**, otherwise the training data and the live data won't match.

---

## 7. Phases (stop after each one)

### Phase 0: Setup and verification (1–2 days)

- Scaffold the repo, CI, Makefile, and pinned environments.
- Verify and record in `reports/phase0.md`: the dataset download links and checksums, where the official train/test split files are, whether **signer ids** are recoverable (filenames, metadata, or paper), the current MediaPipe Tasks landmarkers available in **both** Python and web (same model files and versions), and the current onnxruntime-web WebGPU status.
- **Gate:** if Python and web cannot use the same landmark model version, stop and propose a solution.

### Phase 1: Data pipeline and keypoint extraction

- `make download`: resumable downloads, md5 verification, one zip at a time.
- **Disk-safe extraction:** download one zip → extract keypoints → write compact `.npz` → delete the videos → next zip. The full dataset must never need to be on disk at once (Kaggle's working disk is ~20 GB).
- The extractor outputs per frame: selected pose landmarks (upper body), 21 landmarks per hand, a per-hand presence flag, and timestamps. Record exact landmark indices and their order in `spec/keypoints.json`.
- Log per video: fps, resolution, frame count, % frames with each hand detected. Videos with near-zero hand detection go to a review list and are not silently dropped.
- Upload the processed keypoints as a **private Kaggle dataset** for training.
- Produce `reports/data.md`: class counts, sequence length distribution, detection rates, the split sizes.
- **Acceptance:** the extraction is deterministic (same video gives the same output), and the INCLUDE-50 split loads with the expected train/test counts.

### Phase 2: Normalization, augmentation, baseline

- **Normalization** (identical spec in Python and TypeScript, driven by `spec/keypoints.json`):
  - Center on the shoulder midpoint; scale by shoulder width.
  - Hands also get a hand-local version (centered on the wrist, scaled by palm size), concatenated with the body-relative version.
  - Missing hands → zeros plus a presence mask channel. Never NaN.
  - Optional velocity features (frame differences).
  - Resample to a fixed number of frames T, or use variable length with a padding mask. Decide in an ADR.
- **Augmentations** (training only): small rotation, anisotropic scale (x ≠ y), non-uniform time warp, temporal crop (imprecise sign boundaries), random frame drop, hand dropout (simulate missed detections), Gaussian jitter. Isotropic scale, translation and uniform time stretch are deliberately excluded: they are exact no-ops under shoulder normalization and timestamp resampling to fixed T (proved in `ml/tests/test_augment.py`). **Horizontal mirroring** (left-handed signers) is an ablation, not a default, because swapping hands can change meaning for some signs.
- **Baseline:** a bidirectional GRU. Train on INCLUDE-50 first.
- **Acceptance:** a baseline number on INCLUDE-50, and unit tests for every normalization step (including a golden-file test shared with the web implementation).

### Phase 3: Main model

- A Transformer encoder over frame tokens: linear embedding of per-frame features, learned positional encoding, CLS token or attention pooling, small (aim for under 5M parameters so it runs in a browser).
- Training: AdamW, cosine schedule with warmup, label smoothing, early stopping on a validation split carved from **train** (never tune on test), mixed precision on Kaggle GPUs, and checkpointing every epoch (sessions can end at 12 h).
- **Ablations** (each one a row in the results table): GRU vs Transformer, with/without velocities, with/without hand-local coordinates, with/without mirroring, with/without depth (z), model size (small/medium).
- Then scale to the full 263-class INCLUDE.
- **Evaluation:** top-1, top-5, per-class accuracy, a confusion matrix, the most-confused sign pairs (with a short note on why they're similar), and bootstrap 95% CIs.
- **Session-independent and inferred signer-independent evaluation:** INCLUDE has no signer ids. (1) Hold out whole recording sittings (camera-number blocks across all classes, boundaries checked by eye, ADR 005) and call it "session-independent". (2) Cluster sittings into inferred people (pose body proportions + cached frames, checked by eye) and hold out whole inferred people: call it "inferred signer-independent", always with the word *inferred*. Never call either one plain "signer-independent". Both numbers will be lower. Report them next to the official split result. If the authors provide signer ids, real ids replace the inferred ones.
- **Acceptance:** results reproduce from config + seed, and the README table is generated by script.

### Phase 4: Export and parity

- Export to ONNX with dynamic sequence length (or fixed T). Quantize (dynamic int8, or fp16 for WebGPU), and compare both.
- **Parity checks (automated tests):**
  1. PyTorch vs ONNX fp32: max logit difference under a tolerance.
  2. ONNX fp32 vs quantized: ≥ 99% top-1 agreement on the test set, and the accuracy drop is reported.
  3. **Python preprocessing vs TypeScript preprocessing** on the same keypoint file: identical tensors within a tolerance (golden files committed).
  4. **Python MediaPipe vs browser MediaPipe** on the same short video: keypoints close within a tolerance. This catches training/serving skew. If they diverge, stop and report.
- **Acceptance:** all four parity tests pass in CI (4 can run in Playwright with a test video).

### Phase 5: Real-time inference in the browser

- Camera → MediaPipe (browser) → normalized frame buffer → model in a **Web Worker** → smoothed prediction → UI.
- **Sign spotting** (when does a sign start and end?). Recommended: a motion-energy state machine (`idle → signing → cooldown`). Hands raised plus motion above a threshold starts recording. Hands lowered, or motion below a threshold for N frames, ends it and triggers classification. Make thresholds configurable and document them.
- **Rejection:** if the top probability is below a threshold, or the top-1/top-2 margin is too small, show "not sure" instead of a wrong word. Train with an extra **"no sign / idle" class** built from non-sign clips (people talking, scratching, resting hands) recorded in Phase 6, or calibrate a threshold on the validation set. Decide in an ADR.
- Temporal smoothing and debounce so the label doesn't flicker.
- Runtime selection: WebGPU if available, otherwise WASM (SIMD + threads where possible), and show which one is active in a small debug panel (fps, inference ms, backend).
- **Always 480 rows into MediaPipe (ADR 007):** whatever the camera delivers, downscale each frame to 480 rows with `frameForMediaPipe` (`web/src/lib/frames/downscale.ts`: createImageBitmap, resizeQuality "high", the spec's width rule) before MediaPipe, and normalize with that processed size. This matches offline extraction (cv2.INTER_AREA at 480 rows); parity test 4 checks it.
- **GPU warm-up at page load:** run the MediaPipe (and model) GPU path on a few dummy frames as soon as the page loads, before the camera view goes live, and show a "preparing" state until it's done (first GPU frame took 9.4 s on an M3 Pro).
- **Before starting Phase 5:** test the live app on at least one cheaper device (a friend's laptop or an Android phone) and report the measured fps with the device name.
- **Acceptance:** ≥ 15 fps end-to-end on a mid-range laptop (25 stretch), measured fps reported with the device name, latency numbers measured by an in-app benchmark page and reported with the device name, and a Playwright test that runs the pipeline on a pre-recorded video and checks the predicted word.

### Phase 6: Real-world test set

- Build `tools/record/`: a local page that shows a target word, records **only keypoints** (no video stored) for a few seconds, and saves labeled sequences.
- Mukund and friends record a held-out test set (e.g. INCLUDE-50 words × several people × several takes), with written consent, in varied lighting and backgrounds. Also record "no sign" clips.
- Evaluate the model on it. **This is the honest real-world number.** Expect it to be lower than the dataset number, and analyze why (signer variation, camera angle, distance, motion blur).
- Optional: fine-tune with a small part of this data, keeping a separate untouched evaluation part. Never evaluate on data used for training.
- **Acceptance:** `reports/real_world.md` contains accuracy, failure analysis, and before/after fine-tuning numbers if done.

### Phase 7: The app

See Section 9.

### Phase 8: Hardening, deploy, write-up

- Deploy to Vercel. Set the required headers (Section 8, Browser).
- Lighthouse and accessibility pass, mobile pass, cross-browser test matrix (Chrome, Edge, Firefox, Safari; desktop and Android).
- README, model card, `docs/writeup.md` (~1,000 words on the hardest problem: probably training/serving skew or sign spotting), a 2-minute demo video, and `reports/latest.md`.

---

## 8. Edge cases (handle and test each one)

### Data
- Corrupt or unreadable videos, unusual codecs (`.MOV`), variable fps, rotated video metadata → log, skip with a record, and never crash the batch.
- Download interrupted mid-zip → resume, re-verify checksum.
- Duplicate videos across classes or splits → detect by hash and report.
- Class imbalance → report class counts. Use class-balanced sampling or loss weighting if needed, and report per-class accuracy so rare classes aren't hidden by the mean.
- Leakage: the same signer and session in train and test inflates accuracy → that's why the session-independent number is reported. Never let validation data come from the test split.
- Label mapping drift → `labels.json` is generated once, versioned, and shipped with the model. The web app refuses to load a model whose labels hash doesn't match.

### Keypoints
- Hands not detected in some frames (fast motion, blur, out of frame) → presence mask plus a short-gap interpolation rule (documented max gap); longer gaps stay masked.
- Hands crossing or overlapping → MediaPipe may swap left/right labels. Detect swaps by position continuity and correct them. Test with crafted cases.
- Only one hand visible for one-handed signs → valid input, not an error.
- Person off-center, far from or very close to the camera → normalization handles scale and translation. Test with augmented extremes.
- Pose landmarks with low visibility scores → treat as missing using a visibility threshold from the spec.
- Multiple people in frame → use only the most central / largest person and warn in the UI.

### Model
- Very short or very long sequences → clamp/resample rules defined in the spec and tested.
- Overconfident wrong predictions → calibration (temperature scaling on validation) plus rejection.
- Quantization hurts some classes much more than others → report per-class accuracy before and after.

### Browser
- **Selfie mirroring:** the preview is shown mirrored for the user, but keypoints must come from the **unmirrored** frames, matching the training data. Test it explicitly, because it's an easy bug to miss.
- Camera permission denied, no camera, or camera already in use → clear UI states with instructions.
- No WebGPU → WASM fallback, and say so in the debug panel.
- **GPU first-frame cost:** the first MediaPipe GPU frame of a session took 9.4 s on an M3 Pro (shader compilation; ADR 002 run 2). Warm the GPU model up at page load, before the user signs, behind a visible "preparing" state, and keep it out of any latency measurement.
- **Aspect ratio and decoder padding:** MediaPipe normalizes x and y by the width and height of the image it was given (16:9 INCLUDE vs 4:3 webcam), and Chrome on macOS can deliver 1920×1088 for a 1080p video (8 padding rows at the bottom). Always convert to aspect-correct coordinates using the size of the image actually passed to MediaPipe (never the container's or the requested size) before shoulder normalization. Tested in `ml/tests/test_normalize.py`; the TypeScript mirror gets the same tests.
- WASM threads need `SharedArrayBuffer`, which needs cross-origin isolation headers (COOP/COEP) → set them in the Vercel config and verify that MediaPipe and model assets still load under them.
- Low-end device dropping frames → adaptive processing (skip frames, lower camera resolution) without changing the model input timing semantics (use timestamps).
- Tab hidden or backgrounded → pause the pipeline and resume cleanly.
- Model download size and slow networks → show download progress, cache with the Cache API / service worker, and lazy-load after first paint.
- Safari and iOS quirks with camera and WebGPU → test, and list known limitations.
- Memory leaks from tensors and video frames over long sessions → explicit disposal, and a 30-minute soak test.

### Privacy and ethics
- No video or keypoints leave the device. Say so in the UI and verify it with a network-request check in Playwright (no uploads).
- Recorded test-set participants give consent. Store keypoints only, with no names attached.
- Don't overclaim. This recognizes isolated words for a limited vocabulary. It is not a replacement for interpreters.
- Credit INCLUDE and AI4Bharat clearly.

## 9. App spec

Visual direction: dark, cinematic, smooth motion, but readable and accessible (keyboard navigable, good contrast, captions readable from a distance). The webcam view is the hero.

**Pages:**
1. **Live:** the webcam with a skeleton overlay, a big recognized word with animated confidence bars (top-3), state indicator (idle / signing / recognizing), and a history strip of recent words. Small debug toggle: fps, inference ms, backend.
2. **Practice:** pick a word, watch a skeleton animation of a reference sign (rendered from dataset keypoints, not raw video), try it, get a score and feedback. Track a streak locally (localStorage).
3. **Dictionary:** searchable list of supported words by category, each with a skeleton animation.
4. **How it works / model card:** pipeline diagram, accuracy table (dataset, session-independent, inferred signer-independent, real-world), confusion highlights, limitations, privacy statement, dataset credit.
5. **Benchmark:** runs the model on the user's device and shows latency and backend, labelled with the device it ran on.

## 10. Testing

- **Unit (Python):** normalization, augmentation shapes and invariants, label mapping, split loading, metric computation.
- **Unit (web):** preprocessing against golden files, the sign-spotting state machine (with synthetic motion sequences), smoothing/debounce, rejection logic.
- **Parity:** the four parity tests in Phase 4.
- **E2E (Playwright):** a pre-recorded video fed as a fake camera → correct word; permission-denied state; WASM fallback path; no network uploads.
- **Soak:** 30 minutes of continuous inference without memory growth beyond a bound.
- **CI:** lint, types, unit tests, parity tests 1–3, a small Playwright suite, and a production build on every push.

## 11. README and write-up

The README must contain, in order:
1. One-line pitch, live demo link, a GIF of live recognition.
2. Results table generated from `reports/latest.md`. Every fps and latency number names the device it was measured on. Columns: official split, session-independent, inferred signer-independent, real-world, size, latency, and the paper's numbers for comparison. Full INCLUDE is reported as 262 of 263 classes (816 test videos) with a note beside the paper's 85.6% (ADR 003).
3. Architecture diagram (video → keypoints → normalization → model → spotting → UI).
4. Key design decisions with links to ADRs (keypoints vs pixels, same extractor in train and browser, sign spotting, rejection, quantization).
5. Limitations stated plainly (isolated words only, vocabulary size, real-world accuracy gap, lab-recorded dataset).
6. How to run locally in under 5 commands, and how to reproduce training on Kaggle.
7. Dataset credit and license.

## 12. Stretch goals (only after Definition of Done)

- Continuous sign recognition (CTC over sentences) using a sentence-level ISL dataset such as iSign.
- Self-supervised pretraining on unlabeled keypoint sequences, then fine-tuning.
- Text-to-sign: animate a skeleton for a typed sentence from the dictionary.
- PWA with offline support.

## 13. Definition of done

- [ ] All phases accepted, and `PROGRESS.md` complete
- [ ] Results table reproduces from configs and seeds
- [ ] All parity tests pass in CI
- [ ] Live demo deployed on Vercel, works on Chrome, Edge, and at least one mobile browser
- [ ] ≥ 15 fps end-to-end on a mid-range laptop (25 stretch), measured on the Benchmark page, with the device named
- [ ] Real-world test set recorded, evaluated, and reported honestly
- [ ] Model card, ADRs, write-up, demo video done
- [ ] Mukund can explain every component and every ADR without notes
