"""Fixture for the Python-vs-browser MediaPipe parity test (web/tests/e2e/extractor-parity.spec.ts).

    uv run python -m sanket.checks.browser_parity_fixture

Takes the first INCLUDE-50 *test* video, decodes it once with OpenCV (the training decoder),
and writes into ml/.cache/parity/ (gitignored: INCLUDE frames are never committed):

* frames/NNNN.png   lossless frames, so Python and the browser see identical pixels
* stamps.json       per-frame timestamps (ms) from the decoder
* frames480/NNNN.png the same frames downscaled to 480 rows exactly as offline extraction does
                    (cv2.INTER_AREA, ADR 007)
* expected.json     per extractor, native and `_480`: hand rate and per-frame hand landmarks,
                    computed in Python on those same frames

This is an early, narrow form of Phase 4 parity test 4 (same video → close keypoints).
"""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sanket.checks.sample import MANIFEST, VIDEO_CACHE, fetch
from sanket.data.sessions import load_manifest
from sanket.data.splits import load_split, video_key
from sanket.keypoints.extractor import EXTRACTORS, Sequence
from sanket.keypoints.video import iter_frames, open_video
from sanket.paths import CACHE_DIR
from sanket.pipeline.extract import resize_to_height
from sanket.spec import fetch_model, load_spec, model_pins

OUT = CACHE_DIR / "parity"
EXTRACT_HEIGHT = 480  # ADR 007


def pick_video() -> Path:
    videos = load_manifest(MANIFEST)
    by_key = {video_key(v.path): v for v in videos}
    first = next(
        by_key[video_key(p)] for p in load_split("include50", "test") if video_key(p) in by_key
    )
    fetch([first.path], videos)
    return VIDEO_CACHE / first.path


def main() -> None:
    video = pick_video()
    frames_dir = OUT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap, _ = open_video(video)
    stamps: list[int] = []
    for i, (rgb, t) in enumerate(iter_frames(cap)):
        cv2.imwrite(str(frames_dir / f"{i:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        stamps.append(t)
    cap.release()
    (OUT / "stamps.json").write_text(json.dumps(stamps))

    models = {p.name: fetch_model(p, CACHE_DIR / "models") for p in model_pins(load_spec())}
    expected: dict[str, Any] = {
        "video": str(video.relative_to(VIDEO_CACHE)),
        "n_frames": len(stamps),
    }
    # Native frames, and the same frames downscaled exactly as offline extraction does it
    # (ADR 007: 480 rows with cv2.INTER_AREA via the pipeline's own resize_to_height).
    frames480 = OUT / "frames480"
    frames480.mkdir(exist_ok=True)
    for i in range(len(stamps)):
        bgr = cv2.imread(str(frames_dir / f"{i:04d}.png"))
        if bgr is None:
            raise RuntimeError(f"could not read frame {i}")
        cv2.imwrite(str(frames480 / f"{i:04d}.png"), resize_to_height(bgr, EXTRACT_HEIGHT))

    for suffix, height in (("", None), ("_480", EXTRACT_HEIGHT)):
        for name, cls in EXTRACTORS.items():
            seq = Sequence()
            with cls(models) as ex:
                for i, t in enumerate(stamps):
                    bgr = cv2.imread(str(frames_dir / f"{i:04d}.png"))
                    if bgr is None:
                        raise RuntimeError(f"could not read frame {i}")
                    rgb = resize_to_height(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), height)
                    ex.add(rgb, t, seq)
            a = seq.arrays()
            any_hand = a["left_present"] | a["right_present"]
            key = name + suffix
            expected[key] = {
                "hand_rate": float(any_hand.mean()),
                "left_present": a["left_present"].tolist(),
                "right_present": a["right_present"].tolist(),
                "left": np.round(a["left"][:, :, :2], 5).tolist(),
                "right": np.round(a["right"][:, :, :2], 5).tolist(),
            }
            print(f"{key}: hand rate {expected[key]['hand_rate']:.3f} over {len(stamps)} frames")
    (OUT / "expected.json").write_text(json.dumps(expected))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
