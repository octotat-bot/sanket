"""Features v1 (spec normalization.features_v1): crafted cases for every rule."""

import numpy as np
import pytest

from sanket.keypoints.features import (
    FeatureSpec,
    clip_features,
    clip_to_model_input,
    fill_short_gaps,
    frame_features,
    resample,
    velocities,
)
from sanket.spec import load_spec

FS = FeatureSpec.from_spec()
P = FS.parts
PRES = slice(FS.n_coord, FS.n_coord + 3)


def body(
    width: int, height: int, scale: float = 0.12, ox: float = 0.45, oy: float = 0.35
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One fixed body in pixels, observed in a frame of the given size (as MediaPipe reports it)."""
    rng = np.random.default_rng(0)
    pose_u = rng.uniform(-1, 1, size=(25, 2))
    pose_u[11], pose_u[12] = (0.5, 0.0), (-0.5, 0.0)
    left_u = rng.uniform(-0.2, 0.2, size=(21, 2)) + np.array([0.7, 0.9])
    left_u[9] = left_u[0] + (0.0, -0.15)  # palm size 0.15 shoulder widths
    right_u = rng.uniform(-0.2, 0.2, size=(21, 2)) + np.array([-0.7, 0.9])
    right_u[9] = right_u[0] + (0.0, -0.15)
    s, o = scale * width, np.array([ox * width, oy * height])

    def obs(u: np.ndarray) -> np.ndarray:
        px = u * s + o
        return np.column_stack([px[:, 0] / width, px[:, 1] / height, np.zeros(len(u))])

    pose = np.column_stack([obs(pose_u), np.ones(25)])  # x, y, z, visibility
    return pose, obs(left_u), obs(right_u)


def clip(width: int, height: int, n: int = 5, **kw: float) -> dict[str, np.ndarray]:
    pose, left, right = body(width, height, **kw)
    return {
        "t_ms": np.arange(n) * 40,
        "frame_size": np.array([width, height]),
        "pose": np.repeat(pose[None], n, 0),
        "left": np.repeat(left[None], n, 0),
        "right": np.repeat(right[None], n, 0),
        "pose_present": np.ones(n, bool),
        "left_present": np.ones(n, bool),
        "right_present": np.ones(n, bool),
    }


def test_feature_count_matches_spec() -> None:
    assert FS.n_features == load_spec()["normalization"]["features_v1"]["n_features"] == 193


@pytest.mark.parametrize("size", [(1920, 1088), (640, 480), (854, 480)])
def test_same_body_same_model_input_across_frames_and_placements(size: tuple[int, int]) -> None:
    ref = clip_to_model_input(clip(1920, 1080), FS, use_velocity=True)
    other = clip_to_model_input(clip(*size, scale=0.2, ox=0.3, oy=0.4), FS, use_velocity=True)
    np.testing.assert_allclose(other, ref, atol=1e-5)


def test_missing_hand_is_zeros_with_flag_zero_and_never_nan() -> None:
    pose, left, right = body(1920, 1080)
    f = frame_features(pose, left, right * np.nan, (True, True, False), 1920, 1080, FS)
    assert not np.isnan(f).any()
    assert f[PRES].tolist() == [1.0, 1.0, 0.0]
    assert not f[P["right"]].any() and not f[P["right_local"]].any()
    assert f[P["left"]].any() and f[P["left_local"]].any()


def test_low_visibility_shoulders_remove_the_body_frame_but_keep_hand_local() -> None:
    pose, left, right = body(1920, 1080)
    pose[11, 3] = 0.2
    f = frame_features(pose, left, right, (True, True, True), 1920, 1080, FS)
    assert f[PRES].tolist() == [0.0, 1.0, 1.0]
    assert not f[P["pose"]].any() and not f[P["left"]].any()  # no body frame
    assert f[P["left_local"]].any()


def test_low_visibility_pose_point_is_zeroed() -> None:
    pose, left, right = body(1920, 1080)
    pose[0, 3] = 0.1  # nose
    f = frame_features(pose, left, right, (True, True, True), 1920, 1080, FS)
    nose = FS.pose_idx.index(0)
    assert f[P["pose"]][2 * nose : 2 * nose + 2].tolist() == [0.0, 0.0]


def test_hand_local_is_wrist_centred_and_palm_scaled() -> None:
    pose, left, right = body(1920, 1080)
    f = frame_features(pose, left, right, (True, True, True), 1920, 1080, FS)
    local = f[P["left_local"]].reshape(21, 2)
    np.testing.assert_allclose(local[0], [0, 0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(local[9]), 1.0, atol=1e-12)


def test_short_gaps_filled_long_gaps_and_edges_kept() -> None:
    c = clip(1920, 1080, n=12)
    c["left"][:, :, 0] = np.linspace(0.1, 0.9, 12)[:, None]  # moving hand
    c["left_present"][[0, 3, 4, 5, 7, 8, 9, 10]] = False  # edge 0, gap 3 (3-5), gap 4 (7-10)
    out = fill_short_gaps(c, FS.max_gap)
    assert out["left_present"].tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
    ]
    np.testing.assert_allclose(out["left"][4, 0, 0], c["left"][4, 0, 0])  # linear motion recovered


def test_resample_length_linear_motion_and_nearest_for_missing() -> None:
    c = clip(1920, 1080, n=5)
    feats = clip_features(c, FS)
    feats[:, 0] = np.arange(5, dtype=float)  # pose x feature moves linearly in time
    out = resample(feats, c["t_ms"], FS, T=9)
    assert out.shape == (9, FS.n_features)
    np.testing.assert_allclose(out[:, 0], np.linspace(0, 4, 9))
    feats[2, FS.n_coord + 1] = 0.0  # left hand missing at frame 2
    feats[2, P["left_local"]] = 0.0
    out = resample(feats, c["t_ms"], FS, T=9)
    # Output 5 sits at frame 2.5, between frame 2 (left missing) and frame 3. At w = 0.5 the
    # "nearest" rule picks the later frame (3), which has the hand, so the output has it.
    assert out[5, FS.n_coord + 1] == 1.0


def test_single_frame_and_empty_clips() -> None:
    c = clip(640, 480, n=1)
    assert clip_to_model_input(c, FS, T=7).shape == (7, 193)
    with pytest.raises(ValueError):
        resample(np.zeros((0, FS.n_features)), np.zeros(0), FS)


def test_velocities_zero_where_part_missing_and_at_start() -> None:
    seq = np.zeros((4, FS.n_features))
    seq[:, FS.n_coord :] = 1.0
    seq[:, P["left_local"]] = np.arange(4)[:, None]
    seq[2, FS.n_coord + 1] = 0.0
    v = velocities(seq, FS)
    assert v.shape == (4, FS.n_coord)
    assert not v[0].any()
    assert (v[1, P["left_local"]] == 1).all()
    assert not v[2, P["left_local"]].any() and not v[3, P["left_local"]].any()
