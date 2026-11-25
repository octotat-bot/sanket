"""Synthetic sign clips in the stored .npz layout, for testing the training pipeline end to end.

Each class is a distinct right-hand trajectory relative to the body (class k moves the hand along
direction 2πk / n_classes, then back). Clips vary in frame size (16:9, padded 1080p, 4:3), body
placement and scale, signing speed and length, and have random missing frames, so they flow through
the real gap-fill → features → resampling → augmentation path. No dataset data is involved.
"""

import numpy as np

FRAME_SIZES = [(1920, 1080), (1920, 1088), (854, 480), (640, 480)]


def synthetic_clip(label: int, n_classes: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = int(rng.integers(30, 80))
    w, h = FRAME_SIZES[int(rng.integers(len(FRAME_SIZES)))]
    a = w / h
    scale = rng.uniform(0.15, 0.3)  # shoulder width in frame-height units
    cx, cy = rng.uniform(0.4, 0.6) * a, rng.uniform(0.35, 0.5)
    body = rng.uniform(-1, 1, size=(25, 2)) * 0.5
    body[11], body[12] = (0.5, 0.0), (-0.5, 0.0)
    hand_shape = rng.uniform(-0.12, 0.12, size=(21, 2))
    hand_shape[9] = hand_shape[0] + (0.0, -0.15)
    angle = 2 * np.pi * label / n_classes
    direction = np.array([np.cos(angle), np.sin(angle)])
    phase = np.linspace(0, 1, n) ** rng.uniform(0.7, 1.4)  # signing speed varies within the sign
    reach = np.sin(np.pi * phase)[:, None] * direction * 0.8  # out and back, shoulder widths

    def to_norm(units: np.ndarray) -> np.ndarray:
        px = units * scale + np.array([cx, cy])
        return np.column_stack([px[:, 0] / a, px[:, 1]])

    pose = np.zeros((n, 25, 4))
    left = np.zeros((n, 21, 3))
    right = np.zeros((n, 21, 3))
    for i in range(n):
        pose[i, :, :2] = to_norm(body + rng.normal(0, 0.01, size=(25, 2)))
        pose[i, :, 3] = 1.0
        right[i, :, :2] = to_norm(
            hand_shape + np.array([-0.6, 0.6]) + reach[i] + rng.normal(0, 0.01, (21, 2))
        )
        left[i, :, :2] = to_norm(hand_shape + np.array([0.6, 1.2]) + rng.normal(0, 0.01, (21, 2)))
    present = {k: rng.random(n) > 0.05 for k in ("pose", "left", "right")}
    for key in ("pose", "left", "right"):
        arr = {"pose": pose, "left": left, "right": right}[key]
        arr[~present[key]] = 0.0
    t = np.cumsum(rng.uniform(30, 50, size=n)) - 30
    return {
        "t_ms": t,
        "frame_size": np.array([w, h]),
        "pose": pose,
        "left": left,
        "right": right,
        "pose_present": present["pose"],
        "left_present": present["left"],
        "right_present": present["right"],
    }


def synthetic_dataset(
    n_classes: int, per_class: int, seed: int
) -> tuple[list[dict[str, np.ndarray]], list[int]]:
    rng = np.random.default_rng(seed)
    labels = [k for k in range(n_classes) for _ in range(per_class)]
    return [synthetic_clip(k, n_classes, rng) for k in labels], labels
