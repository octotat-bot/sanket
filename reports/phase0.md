# Phase 0 report: setup and verification

Date: 2026-11-24. The dataset numbers here come from `make inventory`
(`reports/phase0_inventory.md` and `.json`). Version and hash facts come from package
registries and files that were actually downloaded, as cited.

## Gate result

**PASS.** Python and the browser can use the same landmark library version and the same model
bytes.

- `mediapipe==1.2.3` (PyPI) and `@mediapipe/tasks-vision@1.2.3` (npm) were released together.
  Both export `HolisticLandmarker`, `HandLandmarker`, and `PoseLandmarker`.
- The three pinned `.task` files (sha256 in `spec/keypoints.json`) load and run in VIDEO mode:
  - Python 3.12 on macOS arm64 (`ml/tests/test_mediapipe_smoke.py`)
  - headless Chromium on the CPU delegate (`web/tests/e2e/mediapipe-smoke.spec.ts`)
- Not yet verified: that both runtimes produce **numerically close** keypoints on the same
  video (Phase 4 parity test 4). The browser GPU delegate is also untested (headless has no
  GPU).

## 1. Dataset (INCLUDE)

| Item | Finding |
|---|---|
| Video source | Zenodo record [4010759](https://zenodo.org/records/4010759): 44 zips + README + `download_data.sh`, 56.75 GB |
| Checksums | md5 per file from the Zenodo API, saved in `ml/data/manifests/zenodo_4010759_files.json` |
| Resumable downloads | Zenodo honors HTTP Range (verified, `206 Partial Content`). Resume is feasible for `make download` |
| Per-video listing | `ml/data/manifests/zenodo_4010759_videos.tsv`, read from each zip's table of contents over Range requests, without downloading the zips |
| Official splits | **Zenodo's Google Drive link is dead (404).** `Train_Test_Split.zip` is not in the record. Found in the authors' repo [AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE) `train_test_paths/` @ `df348f4`. Vendored in `ml/data/splits/` with provenance |
| Split sizes | INCLUDE 3127 train + 348 val + 817 test. INCLUDE-50 689 + 77 + 192. Train+val equals Zenodo's stated 3475 / 766 |
| Videos in zips | 4,284 (3,679 `.MOV`, 605 `.MP4`), 262 classes |
| Formats | Mixed `.MOV` / `.MP4`. 605 split entries name the wrong extension. We match by path stem |

### Discrepancies found (each one needs handling, none blocks Phase 1)

1. **License.** CLAUDE.md says CC BY-NC-SA 4.0. **The Zenodo record says CC BY 4.0**
   (`metadata.license.id = "cc-by-4.0"`). I have not checked the paper or AI4Bharat's site.
   Proposal: credit it as CC BY 4.0 (what the distributor states) and keep our non-commercial
   posture anyway. **Mukund to confirm before CLAUDE.md is edited.**
2. **One class has no videos.** "Second (Number)" (`secondnumber`, 8 videos: 6 train, 1 val,
   1 test) is in the split files and the 263-entry label map, but in no zip. The usable full
   set is **262 classes, 816 test videos**. INCLUDE-50 is complete (958/958 videos present).
3. **Video counts.** Zenodo says 4,292 videos (the paper says 4,287). The split files list
   4,290. The zips contain 4,284, which is 4,290 minus the 8 missing ones above. Every video
   in the zips appears in a split.
4. **INCLUDE-50 and INCLUDE splits disagree.** 146 INCLUDE-50 train+val videos are INCLUDE test
   videos, and 159 INCLUDE-50 test videos are INCLUDE train+val. So you must never warm-start
   the 263-class model from an INCLUDE-50 checkpoint and then report INCLUDE test accuracy.
   This changes how Phase 3's "then scale to 263 classes" must be done
   (ADR 003, decision 5).
5. **File names are not unique.** 508 `MVI_####` stems repeat across folders (the camera
   counter wraps from 9999 to 0001). The video identity is the full path.
6. The Zenodo README says Jobs has 1 part, but the record has `Jobs_1of2` and `Jobs_2of2`.
   This is harmless: we use the API file list, not the README.

### Signer ids

**Not recoverable from the data we have.**

- Paths are `Category/NN. Class/MVI_####.MOV`, with no signer field.
- Zip entry timestamps are all 2019-05-10, which is copy time, not recording time.
- Zenodo says the signers were deaf students from St. Louis School for the Deaf, Chennai.
  Secondary sources (iSign, ACL Findings 2024, and a 2026 survey) list **7 signers**. I could
  not confirm that from the original paper, which is on ACM, and I did not access the full text.

What *is* visible: within a class, camera numbers form tight runs (Summer: 4565–4568,
4990–4992, 5420–5424, 9122–9124), and each run is probably one recording session. Options for
a held-out evaluation, for Mukund to choose in Phase 1 or 3:

- **(a)** Email the authors for signer ids. Cleanest, but uncertain to arrive.
- **(b)** Use **session-held-out** splits from camera-number runs. Recoverable today, but
  sessions ≠ signers, so we would call it "session-independent", not signer-independent.
- **(c)** Cluster signers by appearance (for example pose proportions from our own keypoints).
  Approximate, and needs manual checking.

Recommendation: (b) now, (a) in parallel. Report whichever is true, with its name.

## 2. MediaPipe Tasks (keypoints)

| Item | Python | Web |
|---|---|---|
| Package | `mediapipe==1.2.3` (2026-11-11; `py3-none` wheels for macOS arm64, Linux x86_64/aarch64, Windows) | `@mediapipe/tasks-vision@1.2.3` (`latest` tag. Nightlies are `1.3.0-rc.*`) |
| Landmarkers | Holistic, Hand, Pose, Face | Holistic, Hand, Pose, Face |
| Smoke test | passes (VIDEO mode) | passes (VIDEO mode, CPU delegate, Chromium headless) |

Model files (pinned in `spec/keypoints.json`, fetched and verified by `make models`):

| Model | Versioned URL | Bytes | sha256 (prefix) | Note |
|---|---|---|---|---|
| holistic_landmarker | `.../holistic_landmarker/float16/1/` | 13,683,609 | `e2dab611…` | `latest` = v1 |
| hand_landmarker | `.../hand_landmarker/float16/1/` | 7,819,105 | `fbc2a300…` | `latest` = v1 |
| pose_landmarker_full | `.../pose_landmarker_full/float16/1/` | 9,398,198 | `5134a3aa…` | **`latest` ≠ v1** (`4eaa5eb7…`, same size) |

Holistic vs Hand+Pose is still open. See ADR 002 (proposal: decide with a browser fps
benchmark at the start of Phase 1).

Side note: `mediapipe` pulls in `opencv-contrib-python` 5.1 and `matplotlib` as transitive
dependencies. Phase 1 needs video decoding, so see open question 3.

## 3. onnxruntime-web and WebGPU

- `onnxruntime-web@1.31.1` (npm `latest`). Its package exports `./webgpu`, `./wasm`, `./all`,
  and `./jspi`, plus `ort-wasm-simd-threaded{,.jsep,.jspi,.asyncify}.wasm`. So the WebGPU
  execution provider and a SIMD+threads WASM fallback are both in the release.
- Python side: `onnxruntime==1.32.0`, `onnx==1.23.2`. Pinned for the Phase 4 parity checks.
- Browser support: per [web.dev](https://web.dev/blog/webgpu-supported-major-browsers), WebGPU now
  ships in Chrome, Edge, Firefox, and Safari. Platform coverage differs (OS, Android, iOS), so
  this gets verified on real devices in Phase 8. The ORT docs
  ([WebGPU EP](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html))
  point to webgpu.io/status for live availability.
- WASM multi-threading needs `SharedArrayBuffer`, which needs COOP/COEP headers. Not set yet
  (Phase 5 and 8). Because we self-host the MediaPipe models (ADR 001), there is one fewer
  cross-origin asset to break under COEP.
- **Not verified in Phase 0:** actually running an ONNX model in the browser. There is no
  model yet, and headless CI has no WebGPU. That happens in Phases 4 and 5.

## 4. Toolchain (all pinned)

| Tool | Version |
|---|---|
| Python | 3.12.13 (uv-managed, `requires-python ==3.12.*`), `uv.lock` committed |
| torch | 2.15.0, in a separate `train` dependency group so CI doesn't pull CUDA wheels |
| Node | 22 · pnpm 12.23.0 (`packageManager` field) · Next.js 16.4.6 · React 19.3.0 · TypeScript 5.9.3 · Tailwind 4.3.3 |
| Tests | pytest 9.2.0 · Vitest 5.1.1 · Playwright 1.65.1 |
| Lint/types | ruff 0.17.6 · mypy 2.4.1 (strict) · ESLint 9.39.5 · tsc |
| CI | GitHub Actions (`.github/workflows/ci.yml`), actions pinned to release tags |

## Open questions for Mukund

1. **License** (discrepancy 1): OK to credit INCLUDE as CC BY 4.0 per Zenodo and update
   CLAUDE.md section 6?
2. **Held-out evaluation:** option (b) session-held-out now, plus (a) emailing the
   authors?
3. **Video decoding library (Phase 1)** is not listed in section 4. Options: `opencv-python`
   (already pulled in transitively as `opencv-contrib-python` by mediapipe, so it adds no
   download) or `PyAV` (better handling of rotation metadata and variable fps in `.MOV`). I
   recommend declaring `opencv-contrib-python` explicitly at the version mediapipe already
   pins, and adding PyAV only if rotation or variable-fps handling turns out to need it.
   **Needs your approval.**
4. **YAML configs** (section 5 says one YAML per experiment) need `pyyaml==6.0.4`. **Approve?**
5. **ADR 002** (Holistic vs Hand+Pose): OK to settle it with the browser benchmark as the
   first step of Phase 1?
6. **ADR 004** (TensorBoard over MLflow): approve?
7. **Kaggle:** not verified in Phase 0 (needs your account). Phase 1 needs: the Kaggle image's
   Python version (our package pins 3.12, but uv can install 3.12 there), and a Kaggle API
   token on your machine to upload the private keypoint dataset.
