"""Per-frame keypoint extraction with the pinned MediaPipe models (spec/keypoints.json).

Two implementations of the same interface, because ADR 002 chooses between them:
  * HolisticExtractor   (holistic_landmarker.task)
  * HandPoseExtractor   (hand_landmarker.task + pose_landmarker_full.task)

Both return the stored layout from spec "stored": pose landmarks 0-24 with x,y,z,visibility,
21 x,y,z points per hand, presence flags, and the signer's own left/right. A fresh extractor
is created per video, so MediaPipe's tracking state never carries over between videos
(needed for deterministic, order-independent output).
"""

from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from sanket.spec import load_spec

_STORED = load_spec()["stored"]
POSE_INDICES: list[int] = _STORED["pose"]["indices"]
N_POSE = len(POSE_INDICES)
N_HAND: int = _STORED["hands"]["landmarks"]


@dataclass
class Sequence:
    """Keypoints for one video, appended frame by frame."""

    t_ms: list[int] = field(default_factory=list)
    pose: list[np.ndarray] = field(default_factory=list)  # (N_POSE, 4)
    left: list[np.ndarray] = field(default_factory=list)  # (N_HAND, 3)
    right: list[np.ndarray] = field(default_factory=list)
    pose_present: list[bool] = field(default_factory=list)
    left_present: list[bool] = field(default_factory=list)
    right_present: list[bool] = field(default_factory=list)
    dropped_duplicate_hands: int = 0  # Hand+Pose only: two hands with the same label

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self.t_ms)
        return {
            "t_ms": np.asarray(self.t_ms, np.int64),
            "pose": np.asarray(self.pose, np.float32).reshape(n, N_POSE, 4),
            "left": np.asarray(self.left, np.float32).reshape(n, N_HAND, 3),
            "right": np.asarray(self.right, np.float32).reshape(n, N_HAND, 3),
            "pose_present": np.asarray(self.pose_present, bool),
            "left_present": np.asarray(self.left_present, bool),
            "right_present": np.asarray(self.right_present, bool),
        }


def _pose_array(landmarks: list[Any]) -> tuple[np.ndarray, bool]:
    if not landmarks:
        return np.zeros((N_POSE, 4), np.float32), False
    rows = []
    for i in POSE_INDICES:
        lm = landmarks[i]
        vis = lm.visibility if lm.visibility is not None else 0.0
        rows.append((lm.x, lm.y, lm.z, vis))
    return np.asarray(rows, np.float32), True


def _hand_array(landmarks: list[Any]) -> tuple[np.ndarray, bool]:
    if not landmarks:
        return np.zeros((N_HAND, 3), np.float32), False
    return np.asarray([(lm.x, lm.y, lm.z) for lm in landmarks], np.float32), True


def _flat(landmarks: Any) -> list[Any]:
    """Holistic returns one person: accept [lm, ...] or [[lm, ...]]."""
    if landmarks and isinstance(landmarks[0], list):
        return list(landmarks[0])
    return list(landmarks or [])


class Extractor(Protocol):
    name: str

    def __enter__(self) -> Self: ...
    def __exit__(
        self, t: type[BaseException] | None, e: BaseException | None, tb: TracebackType | None
    ) -> None: ...
    def add(self, rgb: np.ndarray, t_ms: int, seq: Sequence) -> None: ...


class HolisticExtractor:
    name = "holistic"

    def __init__(self, models: dict[str, Path]) -> None:
        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models["holistic_landmarker"])),
            running_mode=vision.RunningMode.VIDEO,
        )
        self._lm = vision.HolisticLandmarker.create_from_options(opts)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._lm.close()

    def add(self, rgb: np.ndarray, t_ms: int, seq: Sequence) -> None:
        r = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
        pose, pp = _pose_array(_flat(r.pose_landmarks))
        left, lp = _hand_array(_flat(r.left_hand_landmarks))
        right, rp = _hand_array(_flat(r.right_hand_landmarks))
        seq.t_ms.append(t_ms)
        seq.pose.append(pose)
        seq.left.append(left)
        seq.right.append(right)
        seq.pose_present.append(pp)
        seq.left_present.append(lp)
        seq.right_present.append(rp)


class HandPoseExtractor:
    name = "hand_pose"

    def __init__(self, models: dict[str, Path]) -> None:
        self._hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(models["hand_landmarker"])),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(models["pose_landmarker_full"])),
                running_mode=vision.RunningMode.VIDEO,
            )
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._hand.close()
        self._pose.close()

    def add(self, rgb: np.ndarray, t_ms: int, seq: Sequence) -> None:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hr = self._hand.detect_for_video(img, t_ms)
        pr = self._pose.detect_for_video(img, t_ms)
        hands: dict[str, list[Any]] = {}
        ranked = sorted(
            zip(hr.hand_landmarks, hr.handedness, strict=True), key=lambda h: -h[1][0].score
        )
        for lms, cats in ranked:
            side = cats[0].category_name.lower()  # the signer's own side (ADR 002, measure a)
            if side in hands:
                seq.dropped_duplicate_hands += 1
            else:
                hands[side] = lms
        pose, pp = _pose_array(pr.pose_landmarks[0] if pr.pose_landmarks else [])
        left, lp = _hand_array(hands.get("left", []))
        right, rp = _hand_array(hands.get("right", []))
        seq.t_ms.append(t_ms)
        seq.pose.append(pose)
        seq.left.append(left)
        seq.right.append(right)
        seq.pose_present.append(pp)
        seq.left_present.append(lp)
        seq.right_present.append(rp)


EXTRACTORS: dict[str, type[HolisticExtractor] | type[HandPoseExtractor]] = {
    "holistic": HolisticExtractor,
    "hand_pose": HandPoseExtractor,
}
