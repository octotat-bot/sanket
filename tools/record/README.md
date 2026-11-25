# Recording the real-webcam test set (Phase 6)

The recorder is a page inside the web app, **`/dev/record`**, so it uses exactly the same code path
as the live app: camera frame → 480-row downscale (`frameForMediaPipe`, ADR 007) → MediaPipe
Holistic (GPU, CPU fallback) → the same per-frame keypoint layout as offline extraction. A separate
tool would drift from the app; this one can't.

It records **keypoints only**. No images, video or audio are stored, and nothing leaves the device
(the page's Content-Security-Policy blocks all requests to other origins, ADR 006). The participant
saves one JSON file per session and hands it over.

## Running a session

```bash
make models                      # once: pinned MediaPipe models (sha256-checked)
make PNPM="npx -y pnpm@12.23.0" record-page
# open http://localhost:3000/dev/record in Chrome
```

1. **Paper consent first.** Print [CONSENT.md](CONSENT.md) and have the participant sign it. The paper
   form has their name but **not** their participant code, so recordings can't be linked to names.
2. On the page, the participant reads the on-screen consent and ticks "I agree". The camera is not
   started before that. The page creates a random participant code (for example `P-K7Q2MX`). Tell
   them they can note it privately if they might want their data deleted later.
3. Setup: sit so the camera sees the head, shoulders, elbows and both hands (shoulders are the
   body reference; hips aren't needed). Wait for "ready" (the first GPU frame can take several seconds).
4. Session: every INCLUDE-50 word × N takes plus "no sign" clips, in random order. Per clip: Record
   (Space) → 3-2-1 → fixed-length recording → quality check → Accept (Enter) or Retake (R). Skip (S)
   for words the participant doesn't know.
5. **Save session file** (any time, and at the end): `sanket-rec-<participant>-<session>.json`.

"No sign" clips are everyday movements that are not signing (resting hands, scratching the face,
talking, typing...). They train and test the "not sure / no sign" behaviour (Phase 5 rejection).

## Importing the files

```bash
make import-recordings FILES="~/Downloads/sanket-rec-*.json"
```

`python -m sanket.data.recorded` validates every file against `spec/recording_v1.json` (the same rules
as the browser), refuses files with image, video or name fields, and writes one `.npz` per clip to
`data/realworld/<session_id>/` in the same layout as INCLUDE extraction. Session files and converted
clips are gitignored and never committed.

## Open before collecting at scale

- **Learning the signs.** Participants who don't know ISL need a reference for each INCLUDE-50 word.
  The app must not show raw dataset videos (CLAUDE.md section 2). Options: the practice page's skeleton
  animation (Phase 7, needs the extracted keypoints), or a public ISL dictionary. Decide before recording.
- Keep a separate, untouched evaluation part if any of this data is later used for fine-tuning
  (CLAUDE.md Phase 6).
