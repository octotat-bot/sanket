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

## Phase 1: Data pipeline and keypoint extraction ⏳ pre-checks done, waiting on ADR 002 (b)

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
