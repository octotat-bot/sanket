"""The same body must give the same features whatever the frame's aspect ratio or padding.

A synthetic pose and hand are placed in *pixels*, then "observed" the way MediaPipe reports
them for different frames: divided by that frame's own width and height.
"""

import numpy as np
import pytest

from sanket.keypoints.normalize import L_SHOULDER, R_SHOULDER, aspect_correct, body_relative

# One fixed body (unit: shoulder width), reused for every frame so only the camera differs.
_rng = np.random.default_rng(0)
BODY_POSE = _rng.uniform(-1, 1, size=(25, 3))
BODY_POSE[L_SHOULDER, :2] = (0.5, 0.0)  # signer's left shoulder appears on the image right
BODY_POSE[R_SHOULDER, :2] = (-0.5, 0.0)
BODY_HAND = _rng.uniform(-0.3, 0.3, size=(21, 3)) + np.array([0.2, 0.8, 0.0])


def _to_px(a: np.ndarray, scale: float, origin: tuple[float, float]) -> np.ndarray:
    return np.column_stack(
        [a[:, 0] * scale + origin[0], a[:, 1] * scale + origin[1], a[:, 2] * scale]
    )


def synthetic_body(scale: float, origin: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """The same pose (25 points) and hand (21 points) in pixels, placed and scaled as given."""
    return _to_px(BODY_POSE, scale, origin), _to_px(BODY_HAND, scale, origin)


def observe(px: np.ndarray, width: int, height: int) -> np.ndarray:
    """What MediaPipe returns: x / W, y / H, and z on x's scale (z / W)."""
    return np.column_stack([px[:, 0] / width, px[:, 1] / height, px[:, 2] / width])


# INCLUDE 1080p (16:9), Chrome on macOS (same pixels + 8 padding rows), a 4:3 webcam, and the
# 854x480 benchmark copy.
FRAMES = [(1920, 1080), (1920, 1088), (640, 480), (854, 480)]


def test_same_body_same_features_across_aspect_ratios_and_padding() -> None:
    feats = []
    for w, h in FRAMES:
        # Same person, positioned and sized differently in each frame (as a different camera would).
        pose_px, hand_px = synthetic_body(scale=0.12 * w, origin=(0.45 * w, 0.35 * h))
        pose, hand = observe(pose_px, w, h), observe(hand_px, w, h)
        feats.append(body_relative(hand, pose, w, h))
    for f in feats[1:]:
        np.testing.assert_allclose(f, feats[0], atol=1e-9)


def test_padding_rows_do_not_change_features() -> None:
    """Content unmoved, 8 extra rows at the bottom: Chrome's 1920x1088 for a 1080p video."""
    pose_px, hand_px = synthetic_body(scale=200, origin=(900, 400))
    a = body_relative(observe(hand_px, 1920, 1080), observe(pose_px, 1920, 1080), 1920, 1080)
    b = body_relative(observe(hand_px, 1920, 1088), observe(pose_px, 1920, 1088), 1920, 1088)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_using_the_wrong_frame_size_is_detected() -> None:
    """The bug class to avoid: container size (1080) applied to a 1088-row image."""
    pose_px, hand_px = synthetic_body(scale=200, origin=(900, 400))
    right = body_relative(observe(hand_px, 1920, 1080), observe(pose_px, 1920, 1080), 1920, 1080)
    wrong = body_relative(observe(hand_px, 1920, 1088), observe(pose_px, 1920, 1088), 1920, 1080)
    assert np.abs(right - wrong).max() > 1e-3


def test_without_aspect_correction_features_differ() -> None:
    """Guard that the main test is meaningful: raw normalized coords are NOT aspect-invariant."""
    raw = []
    for w, h in [(1920, 1080), (640, 480)]:
        pose_px, hand_px = synthetic_body(scale=0.12 * w, origin=(0.45 * w, 0.35 * h))
        pose, hand = observe(pose_px, w, h), observe(hand_px, w, h)
        mid = (pose[L_SHOULDER, :2] + pose[R_SHOULDER, :2]) / 2
        sw = np.linalg.norm(pose[L_SHOULDER, :2] - pose[R_SHOULDER, :2])
        raw.append((hand[:, :2] - mid) / sw)
    assert np.abs(raw[0] - raw[1]).max() > 1e-2


def test_aspect_correct_scales_x_and_z_only() -> None:
    p = np.array([[0.5, 0.5, 0.1, 0.9]])
    np.testing.assert_allclose(
        aspect_correct(p, 1920, 1080), [[0.5 * 16 / 9, 0.5, 0.1 * 16 / 9, 0.9]]
    )
    with pytest.raises(ValueError):
        aspect_correct(p, 0, 1080)


def test_degenerate_shoulders_raise() -> None:
    pose = np.zeros((25, 3))
    with pytest.raises(ValueError, match="shoulder"):
        body_relative(np.zeros((21, 3)), pose, 1920, 1080)
