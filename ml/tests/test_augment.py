import numpy as np
import pytest

from sanket.augment.transforms import (
    AugmentConfig,
    augment,
    frame_drop,
    hand_dropout,
    jitter,
    mirror,
    rotate,
    scale,
    scale_xy,
    temporal_crop,
    time_stretch,
    time_warp,
    translate,
)
from sanket.keypoints.features import FeatureSpec, clip_to_model_input
from sanket.keypoints.golden import build_clips

FS = FeatureSpec.from_spec()
KEYS = ("t_ms", "pose", "left", "right", "pose_present", "left_present", "right_present")


def base() -> dict[str, np.ndarray]:
    return build_clips()[0]  # 10 frames, 1920x1080, with gaps and low-visibility points


def features(c: dict[str, np.ndarray]) -> np.ndarray:
    return clip_to_model_input(c, FS, use_velocity=True, T=16, dtype=np.float64)


def well_formed(c: dict[str, np.ndarray]) -> None:
    n = len(c["t_ms"])
    assert all(len(c[k]) == n for k in KEYS)
    assert c["pose"].shape[1:] == (25, 4) and c["left"].shape[1:] == (21, 3)
    assert (np.diff(c["t_ms"]) > 0).all()
    for key, flag in (
        ("pose", "pose_present"),
        ("left", "left_present"),
        ("right", "right_present"),
    ):
        assert not np.isnan(c[key]).any()
        assert not c[key][~c[flag].astype(bool)].any()  # missing parts stay zeros


# ---- exact no-ops under our normalization (and why the replacements exist) ----


@pytest.mark.parametrize(
    "aug",
    [lambda c: scale(c, 1.3), lambda c: translate(c, 0.07, -0.05), lambda c: time_stretch(c, 1.4)],
)
def test_isotropic_scale_translation_and_uniform_stretch_do_not_change_the_model_input(aug) -> None:  # type: ignore[no-untyped-def]
    c = base()
    np.testing.assert_allclose(features(aug(c)), features(c), atol=1e-6)


@pytest.mark.parametrize(
    "aug",
    [
        lambda c: scale_xy(c, 1.1, 0.9),
        lambda c: rotate(c, 8.0),
        lambda c: time_warp(c, np.random.default_rng(1), 0.5),
        lambda c: temporal_crop(c, np.random.default_rng(3), 0.3),
    ],
)
def test_effective_augmentations_change_the_model_input(aug) -> None:  # type: ignore[no-untyped-def]
    c = base()
    out = aug(c)
    well_formed(out)
    assert np.abs(features(out) - features(c)).max() > 1e-3


# ---- geometry ----


def test_rotation_preserves_shoulder_width_and_round_trips() -> None:
    c = base()
    r = rotate(c, 12.0)
    a = 1920 / 1080
    ok = c["pose_present"]

    def sw(x: dict[str, np.ndarray]) -> np.ndarray:
        d = x["pose"][ok, 11, :2] - x["pose"][ok, 12, :2]
        out: np.ndarray = np.hypot(d[:, 0] * a, d[:, 1])
        return out

    np.testing.assert_allclose(sw(r), sw(c), atol=1e-9)
    back = rotate(r, -12.0)
    np.testing.assert_allclose(back["left"], c["left"], atol=1e-9)


def test_mirror_swaps_hands_flips_x_and_is_an_involution() -> None:
    c = base()
    m = mirror(c)
    well_formed(m)
    assert (m["left_present"] == c["right_present"]).all()
    ok = c["right_present"]
    np.testing.assert_allclose(m["left"][ok, :, 0], 1.0 - c["right"][ok, :, 0])
    np.testing.assert_allclose(
        m["pose"][:, 11, 0][c["pose_present"]], 1.0 - c["pose"][:, 12, 0][c["pose_present"]]
    )
    for k in KEYS:
        np.testing.assert_allclose(mirror(m)[k], c[k])


def test_mirror_is_off_by_default() -> None:
    assert AugmentConfig().mirror is False


# ---- time and missing data ----


def test_time_warp_keeps_endpoints_and_strictly_increasing_timestamps() -> None:
    c = base()
    w = time_warp(c, np.random.default_rng(0), 0.6)
    assert w["t_ms"][0] == c["t_ms"][0] and w["t_ms"][-1] == c["t_ms"][-1]
    assert (np.diff(w["t_ms"]) > 0).all()


def test_frame_drop_keeps_first_and_last() -> None:
    c = base()
    d = frame_drop(c, np.random.default_rng(0), 0.9)
    well_formed(d)
    assert d["t_ms"][0] == c["t_ms"][0] and d["t_ms"][-1] == c["t_ms"][-1]
    assert len(d["t_ms"]) < len(c["t_ms"])


def test_hand_dropout_marks_missing_and_zeros_landmarks() -> None:
    c = base()
    d = hand_dropout(c, np.random.default_rng(0), p=0.5)
    well_formed(d)
    assert d["left_present"].sum() < c["left_present"].sum()


def test_jitter_zero_is_identity_and_never_touches_missing_parts() -> None:
    c = base()
    for k in KEYS:
        np.testing.assert_allclose(jitter(c, np.random.default_rng(0), 0.0)[k], c[k])
    j = jitter(c, np.random.default_rng(0), 0.05)
    well_formed(j)
    assert not np.allclose(j["left"][c["left_present"]], c["left"][c["left_present"]])


# ---- the composed pipeline ----


def test_augment_is_deterministic_for_a_seed_and_well_formed() -> None:
    c = base()
    cfg = AugmentConfig(
        p_rotate=1, p_scale_xy=1, p_time_warp=1, p_crop=1, p_frame_drop=1, p_hand_dropout=1
    )
    a = augment(c, np.random.default_rng(7), cfg)
    b = augment(c, np.random.default_rng(7), cfg)
    well_formed(a)
    for k in KEYS:
        np.testing.assert_array_equal(a[k], b[k])
    assert np.isfinite(features(a)).all()


def test_augment_never_modifies_its_input() -> None:
    c = base()
    before = {k: v.copy() for k, v in c.items()}
    augment(c, np.random.default_rng(0), AugmentConfig(mirror=True))
    for k in c:
        np.testing.assert_array_equal(c[k], before[k])
