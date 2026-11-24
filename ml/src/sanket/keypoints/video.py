"""Stream decoded frames from a video file, one at a time (never the whole video in memory).

OpenCV with the FFmpeg backend forced, so macOS and Linux/Kaggle decode the same way.
The Phase 1 pre-check (reports/phase1_video_metadata.md) found OpenCV's frames and
timestamps identical to ffprobe on INCLUDE's H.264 files.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int  # as reported by the container; decoding may yield fewer


class VideoError(RuntimeError):
    pass


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise VideoError(f"cannot open {path}")
    info = VideoInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return cap, info


def iter_frames(cap: cv2.VideoCapture) -> Iterator[tuple[np.ndarray, int]]:
    """Yield (RGB frame, t_ms) with strictly increasing integer timestamps."""
    last = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            return
        t = round(cap.get(cv2.CAP_PROP_POS_MSEC))
        if t <= last:  # MediaPipe VIDEO mode requires strictly increasing timestamps
            t = last + 1
        last = t
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t
