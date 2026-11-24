"""Contact sheets for the session spot-check (ADR 005).

    uv run python -m sanket.checks.session_frames

For each sampled class: one row per camera-number run, showing the middle frame of the run's
first take, its last take, and the take that follows it in another class (if any). Rows are
labelled with the run id, camera numbers and role. Sheets go to ml/.cache/spotcheck/ (gitignored).

The INCLUDE card asks users to respect the signers' privacy and dignity, so these frames are
for local verification only: they are never committed, published, or shown in the app.
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from sanket.checks.sample import SAMPLES_FILE, VIDEO_CACHE
from sanket.paths import CACHE_DIR

OUT_DIR = CACHE_DIR / "spotcheck"
TILE_W = 360


def middle_frame(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n // 2, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {path}")
    h, w = frame.shape[:2]
    return cv2.resize(frame, (TILE_W, round(h * TILE_W / w)))


def labelled(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((28, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w, 3), np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    return out


def main() -> None:
    samples = json.loads(SAMPLES_FILE.read_text())
    by_class: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for s in samples["session_spotcheck"]:
        by_class[s["run"][:2]][s["run"]].append(s)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cls, runs in sorted(by_class.items()):
        rows = []
        for run_id, items in sorted(runs.items()):
            tiles = []
            for it in items:
                name = Path(it["path"])
                tag = f"{run_id} {it['role']}: {name.parent.name[:18]} / {name.stem}"
                tiles.append(labelled(middle_frame(VIDEO_CACHE / it["path"]), tag))
            rows.append(tiles)
        th = max(t.shape[0] for r in rows for t in r)
        sheet = np.vstack(
            [
                np.hstack(
                    [pad_to(t, th, TILE_W) for t in r]
                    + [np.zeros((th, TILE_W, 3), np.uint8)] * (3 - len(r))
                )
                for r in rows
            ]
        )
        out = OUT_DIR / f"{cls}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(out)


if __name__ == "__main__":
    main()
