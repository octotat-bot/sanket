# Progress

## Phase 0: Setup and verification ✅ approved 2026-11-24

**What was built**

- Repo skeleton per CLAUDE.md section 5. Git on branch `phase-0`.
- `ml/`: uv project (Python 3.12, exact pins, `uv.lock`) with package `sanket`:
  - `sanket.spec`: reads `spec/keypoints.json` and fetches the pinned MediaPipe models
    (refuses a sha256 mismatch).
  - `sanket.data.zenodo`, `sanket.data.remote_zip`, `sanket.data.splits`,
    `sanket.data.inventory`: dataset inventory computed from metadata only (Zenodo API plus
    zip tables of contents over HTTP Range), with no 57 GB download.
  - Official split files and label maps vendored in `ml/data/splits/` (provenance + pinned hashes).
  - Manifests in `ml/data/manifests/`: md5 per zip, and every video per zip.
- `spec/keypoints.json`: extractor pins (MediaPipe 1.2.3, three model files by URL + bytes + sha256).
- `web/`: Next.js 16 app (App Router, TS, Tailwind, pnpm, exact pins) with
  `@mediapipe/tasks-vision@1.2.3` and `onnxruntime-web@1.31.1`. Placeholder home page with the
  honest framing.
- Tests: 37 offline pytest + 1 network pytest (MediaPipe smoke), 4 Vitest, 1 Playwright
  (MediaPipe smoke in Chromium).
- `Makefile` (`make help`) and GitHub Actions CI (lint, types, unit tests, both smoke tests,
  production build).
- ADRs 001–004 in `docs/decisions/` (001 and 003 accepted, 002 and 004 proposed).
- `reports/phase0.md` (findings) and `reports/phase0_inventory.md` (generated numbers).

**How to verify**

```bash
make setup        # uv sync --locked + pnpm install --frozen-lockfile
make ci           # lint, typecheck, offline tests, models + both MediaPipe smoke tests, build
make inventory    # regenerates reports/phase0_inventory.* from Zenodo (~1 min, metadata only)
```

If `pnpm` is not on PATH: `make PNPM="npx -y pnpm@12.23.0" ci`. The local corepack is too old
for pnpm 12. Installing `npm i -g pnpm@12.23.0` also works.

**Measured numbers** (from `make inventory` and the smoke tests)

- Zenodo: 44 zips, 56.75 GB, 4,284 videos, 262 classes present (263 in the label map).
- Splits: INCLUDE 3,475 train+val / 817 test. INCLUDE-50 766 / 192. 0 train/test overlap
  within each benchmark.
- 8 split videos (class "Second (Number)") missing from Zenodo. 605 `.MOV`/`.MP4` mismatches.
- Cross-benchmark leakage risk: 146 INCLUDE-50 train+val videos are in INCLUDE test.
- MediaPipe 1.2.3 Holistic/Hand/Pose run in both Python and Chromium with identical model
  bytes.

**Open questions** (details in `reports/phase0.md`, "Open questions")

1. License: Zenodo says CC BY 4.0, CLAUDE.md says CC BY-NC-SA 4.0. Confirm.
2. Signer ids are not recoverable. Proposed: session-held-out split + email the authors.
3. Approve a video decoding library (opencv-contrib-python, already transitive) for Phase 1.
4. Approve `pyyaml` for experiment configs.
5. ADR 002: decide Holistic vs Hand+Pose with a browser benchmark at the start of Phase 1.
6. ADR 004: TensorBoard.
7. Kaggle setup (Python version, API token) needs your account.

## Phase 1: Data pipeline and keypoint extraction ⏳ pipeline ready; ADR 002 (b) rerun pending

Phase 0 decisions applied: CC BY 4.0 credit (CLAUDE.md section 6, ADR 003), "session-independent"
wording everywhere, leakage and 262/263 reporting rules (ADR 003), TensorBoard (ADR 004),
`opencv-contrib-python` + `pyyaml` pinned, Kaggle token paths gitignored.

**Pre-checks (all reproducible: `make samples check-video-metadata check-extractor`)**

| Check | Result | Where |
|---|---|---|
| License sources | Zenodo + official HF card: CC BY 4.0. GitHub: MIT (code only). Derived IEEE DataPort set says CC BY-NC-SA 4.0 | ADR 003 |
| 30 .MOV (+ 8 .MP4) decode | All H.264 1920×1080 at 25 fps constant. 0 rotated, 0 VFR, 0 mishandled by OpenCV (frames and timestamps exact). **PyAV not needed** | `reports/phase1_video_metadata.md` |
| Session spot-check (10 runs) | One run = one signer/setting: 10/10. The sitting continues into other classes: 9/9. Runs within a class differ: 10/10. **Sessions span categories**, so blocks must be cut across classes | ADR 005 |
| ADR 002 (a), 50 videos | Holistic: 0 L/R swaps vs 1.08 per 100 for Hand+Pose. Detection 85.8% vs 84.4% (not significant) | `reports/phase1_extractor_compare.md`, ADR 002 |
| ADR 002 (b), browser fps | **Waiting for Mukund's run** on his laptop | `/dev/extractor-bench` |
| Privacy | `@mediapipe/tasks-vision` 1.2.3 POSTs usage metrics to Google. Blocked by `connect-src 'self'` CSP, enforced by an e2e test | ADR 006 |

**How to verify**

```bash
make test lint typecheck                         # 54 pytest, 7 vitest
make PNPM="npx -y pnpm@12.23.0" e2e              # 2 Playwright tests incl. fake-camera benchmark
make samples check-video-metadata check-extractor   # ~1.1 GB of sampled videos, then both reports
```

**Open items before full extraction**

1. Mukund runs ADR 002 (b) and sends the JSON. Then the choice is made and approved.
2. Approve ADR 006 (CSP) and the proposed block-boundary procedure in ADR 005.
3. Disk: only 16 GB free on the laptop (it was 23 GB earlier the same day). Extraction stays
   one zip at a time (largest zip 1.85 GB), but keep at least 10 GB free.

**Approvals received 2026-11-24:** ADR 002 rule, ADR 005 (+ inferred-signer split), ADR 006, Kaggle CLI.

**Pipeline (built and tested; real run not started)**

- `make extract` (`sanket.pipeline.extract`): one zip at a time. Resumable md5-verified
  download → stream frames through a fresh extractor per video → `.npz` per video + one cached
  frame → log → delete video → delete zip. Refuses to start until `extractor` is set in
  `ml/configs/extract.yaml` (ADR 002), and refuses a dirty git tree. Aborts below 10 GB free
  and deletes its partial download.
- Stored layout is in `spec/keypoints.json` 0.2.0: pose 0–24 × (x,y,z,visibility), 21 × (x,y,z)
  per hand, presence flags, t_ms. Pose names are tested against MediaPipe's enum.
- `make upload KAGGLE_DATASET=user/slug`: private dataset, CC-BY-4.0, refuses incomplete or
  media-containing folders.
- `python -m sanket.data.report` → `reports/data.md`: class counts, lengths, detection rates,
  split sizes vs missing, errors, review list, sha256 duplicates.
- Determinism (acceptance) on a real INCLUDE-50 video: both extractors bit-identical across
  runs (`ml/tests/test_extraction_real.py`, network marker, runs in CI).
- Dry run on `Adjectives_8of8.zip` (63 videos, Holistic): 63 ok / 0 errors, 24.2 ms/frame,
  peak RSS 430 MB, download ≈2.6 MB/s. Outputs deleted afterwards.
- **Projected full run: ≈6 h download + ≈1.7 h compute ≈ 8 h. Output ≈200 MB, frame cache ≈46 MB.**

**Disk investigation (2026-11-24)**

- The 7 GB drop came mostly from **macOS swap**: 14 × 1 GB swap files created between
  10:00 and 12:00 (now 21.5 GB of swap), under memory pressure from builds, Playwright and the
  1080p comparison script, plus Chrome, VS Code and Ollama. Swap isn't ours to delete. It
  shrinks as pressure drops, and a reboot clears it.
- Ours, created today: sample videos 1.1 GB and scratch 0.56 GB (both **deleted**), tool caches
  (uv 0.8, pnpm 1.1, Playwright 0.6, npm 0.4 GB; kept, needed or shared). No stray `.part` files.
- Free space after cleanup: 21–24 GB (it moves with swap).

**ADR 002 measure (b), run 1 (2026-11-24): invalid** (Mukund: hand rate < 80% everywhere).
Investigation (2026-11-25, details and evidence in ADR 002):

- Hand+Pose's 9% / 0% is **not a code bug**. On identical frames, browser and Python MediaPipe
  give identical hand rates (88.0 / 68.7%), and the real bench page in the new file mode matches
  Python (89.4 vs 88.0%, 67.2 vs 68.7%). Cause: HandLandmarker needs its palm detector to find
  small hands in the whole frame (0 hands on a full 1080p INCLUDE frame, 1 on a crop, in both
  runtimes). Holistic finds hands from the pose. Run 1's live scene defeated the palm detector.
- New tests: `extractor-parity.spec.ts` (frame-by-frame Python vs browser on an INCLUDE-50
  video) and `extractor-bench-file.spec.ts` (real page, file mode). The fixture is built by
  `make parity-fixture`; INCLUDE frames stay in the gitignored cache.
- Bench: COOP/COEP (cross-origin isolated, asserted in e2e), INCLUDE-file and "both" modes,
  warm-up ≥ 8 s and ≥ 60 frames, first-frame times recorded, seeded random order over rounds,
  on-screen hand overlay.
- fps evidence (`make check-fps`): hand tracking holds down to 10 fps, and lost motion is flat
  (median 0.008 shoulder widths) from 20 to 12.5 fps. **Proposal: 15 fps rule** instead of 30
  (ADR 002), to be confirmed with accuracy in Phase 2.

**Approved 2026-11-25:**

- ADR 002 speed rule: ≥ 15 fps floor on mid-range devices, ≥ 20 fps with p95 ≤ 50 ms on the M3 Pro,
  judged on the faster delegate. The rule changed after run 1, justified by the frame-rate ablation
  (not by any benchmark result). The bench page now shows both bars (`meetsBar`, unit-tested).
- Speed is pass/fail only. A vs B is decided on quality (ADR 002, with scopes stated: swaps
  0 vs 1.08 per 100 over 50 videos; hand rate 85.8 vs 84.4% over 50, not significant, so not
  claimed. Single-video rates are not quoted).
- CLAUDE.md section 3: "≥ 15 fps end-to-end on a mid-range laptop (25 stretch); report measured
  fps and device name". Phase 5 acceptance and the Definition of Done follow it. The README must
  name the device next to every fps number.

**Open before Phase 5:** test the live app on at least one cheaper device (a friend's laptop or
an Android phone) and report fps with the device name.

**Next:** Mukund's benchmark rerun (file + camera). If A clears the M3 Pro bar, set
`extractor: holistic` in `ml/configs/extract.yaml`, commit, and start `make extract` (≈ 8 h).
