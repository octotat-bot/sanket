"""Training-time augmentations on raw clips (stored .npz layout), applied *before* features.

Every transform takes a clip dict (and a numpy Generator where random) and returns a new clip;
inputs are never modified. Time transforms return float millisecond timestamps (no rounding).
Geometry is done in aspect-correct units (x * W/H, like normalization) around the clip's mean
shoulder midpoint, then converted back to MediaPipe's normalized coordinates.

Two facts that matter for choosing augmentations (tested in tests/test_augment.py):
* isotropic `scale` and `translate` are exact no-ops for the model input, because features are
  shoulder-centred and shoulder-scaled. `scale_xy` (different x and y factors) is not.
* uniform `time_stretch` is an exact no-op, because resampling maps the first..last timestamp onto
  T frames. `time_warp` (speed varying within the sign) and `temporal_crop` are not.

`mirror` swaps the signer's hands and flips x. It is an ablation, never a default (CLAUDE.md
section 7: swapping hands can change meaning).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

Clip = dict[str, np.ndarray]
L_SH, R_SH = 11, 12
# MediaPipe pose left/right pairs (indices 0-24): eyes, ears, mouth, shoulders, arms, hands, hips.
POSE_MIRROR_PAIRS = [
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
]


def _copy(clip: Clip) -> Clip:
    return {k: np.array(v, copy=True) for k, v in clip.items()}


def _aspect(clip: Clip) -> float:
    w, h = (float(v) for v in clip["frame_size"])
    return w / h


def _center(clip: Clip) -> np.ndarray:
    """Mean shoulder midpoint (aspect-correct) over frames with a pose; frame centre if none."""
    a = _aspect(clip)
    ok = clip["pose_present"].astype(bool)
    if not ok.any():
        return np.array([0.5 * a, 0.5])
    mid = (clip["pose"][ok, L_SH, :2] + clip["pose"][ok, R_SH, :2]) / 2
    return np.array([mid[:, 0].mean() * a, mid[:, 1].mean()])


def _map_xy(clip: Clip, fn: Any) -> Clip:
    """Apply fn to (…, 2) aspect-correct xy of pose and hands, only where the part is present."""
    out = _copy(clip)
    a = _aspect(clip)
    for key, flag in (
        ("pose", "pose_present"),
        ("left", "left_present"),
        ("right", "right_present"),
    ):
        ok = out[flag].astype(bool)
        if not ok.any():
            continue
        xy = out[key][ok, :, :2].astype(np.float64)
        xy[..., 0] *= a
        xy = fn(xy)
        xy[..., 0] /= a
        out[key][ok, :, :2] = xy
    return out


def rotate(clip: Clip, degrees: float) -> Clip:
    c = _center(clip)
    th = np.deg2rad(degrees)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return _map_xy(clip, lambda xy: (xy - c) @ rot.T + c)


def scale(clip: Clip, factor: float) -> Clip:
    """Isotropic scale about the body centre. A no-op for features (CLAUDE.md lists it)."""
    return scale_xy(clip, factor, factor)


def scale_xy(clip: Clip, sx: float, sy: float) -> Clip:
    """Anisotropic scale (body proportions, lens and aspect differences): changes features."""
    c = _center(clip)
    return _map_xy(clip, lambda xy: (xy - c) * np.array([sx, sy]) + c)


def translate(clip: Clip, dx: float, dy: float) -> Clip:
    """Shift in frame-height units. A no-op for features (kept because CLAUDE.md lists it)."""
    return _map_xy(clip, lambda xy: xy + np.array([dx, dy]))


def jitter(clip: Clip, rng: np.random.Generator, sigma: float) -> Clip:
    """Gaussian noise of `sigma` shoulder widths on present landmarks (x, y)."""
    a = _aspect(clip)
    ok = clip["pose_present"].astype(bool)
    if ok.any():
        d = clip["pose"][ok, L_SH, :2] - clip["pose"][ok, R_SH, :2]
        sw = float(np.median(np.hypot(d[:, 0] * a, d[:, 1])))
    else:
        sw = 0.25  # no pose in the clip: a typical shoulder width in frame-height units
    return _map_xy(clip, lambda xy: xy + rng.normal(0.0, sigma * sw, size=xy.shape))


def time_stretch(clip: Clip, factor: float) -> Clip:
    """Uniform speed change. A no-op for features (resampling to T removes it)."""
    out = _copy(clip)
    t = out["t_ms"].astype(np.float64)
    out["t_ms"] = t[0] + (t - t[0]) * factor  # float ms: rounding would make it not quite a no-op
    return out


def time_warp(clip: Clip, rng: np.random.Generator, strength: float) -> Clip:
    """Non-uniform speed: a random monotonic piecewise-linear warp of the timestamps."""
    out = _copy(clip)
    t = out["t_ms"].astype(np.float64)
    if len(t) < 3:
        return out
    knots = np.linspace(0.0, 1.0, 5)
    speeds = np.exp(rng.uniform(-strength, strength, size=4))  # per-segment speed factors
    warped_knots = np.concatenate([[0.0], np.cumsum(speeds) / speeds.sum()])
    u = (t - t[0]) / (t[-1] - t[0])
    # Float ms (features accept them). Positive speeds make the warp strictly increasing.
    out["t_ms"] = t[0] + np.interp(u, knots, warped_knots) * (t[-1] - t[0])
    return out


def _take(clip: Clip, idx: np.ndarray) -> Clip:
    out = _copy(clip)
    for k in ("t_ms", "pose", "left", "right", "pose_present", "left_present", "right_present"):
        out[k] = out[k][idx]
    return out


def temporal_crop(clip: Clip, rng: np.random.Generator, max_frac: float) -> Clip:
    """Cut up to max_frac of frames off each end (imprecise sign boundaries); keeps ≥ 2."""
    n = len(clip["t_ms"])
    k = int(max_frac * n)
    a = int(rng.integers(0, k + 1)) if k else 0
    b = n - (int(rng.integers(0, k + 1)) if k else 0)
    if b - a < 2:
        return _copy(clip)
    return _take(clip, np.arange(a, b))


def frame_drop(clip: Clip, rng: np.random.Generator, p: float) -> Clip:
    """Drop each interior frame with probability p (first and last are kept)."""
    n = len(clip["t_ms"])
    keep = rng.random(n) >= p
    keep[0] = keep[-1] = True
    return _take(clip, np.flatnonzero(keep))


def hand_dropout(clip: Clip, rng: np.random.Generator, p: float, max_run: int = 5) -> Clip:
    """Missed detections: per hand, with prob p start a run of 1..max_run missing frames."""
    out = _copy(clip)
    n = len(out["t_ms"])
    for key, flag in (("left", "left_present"), ("right", "right_present")):
        i = 0
        while i < n:
            if rng.random() < p:
                run = int(rng.integers(1, max_run + 1))
                out[flag][i : i + run] = False
                out[key][i : i + run] = 0.0
                i += run
            else:
                i += 1
    return out


def mirror(clip: Clip) -> Clip:
    """Swap the signer's hands and flip x (ablation only)."""
    out = _copy(clip)
    for key, flag in (
        ("pose", "pose_present"),
        ("left", "left_present"),
        ("right", "right_present"),
    ):
        ok = out[flag].astype(bool)
        out[key][ok, :, 0] = 1.0 - out[key][ok, :, 0]  # flip only present parts; missing stay zeros
    for i, j in POSE_MIRROR_PAIRS:
        out["pose"][:, [i, j]] = out["pose"][:, [j, i]]
    out["left"], out["right"] = out["right"].copy(), out["left"].copy()
    out["left_present"], out["right_present"] = (
        out["right_present"].copy(),
        out["left_present"].copy(),
    )
    return out


@dataclass(frozen=True)
class AugmentConfig:
    """Probabilities and ranges. Defaults are a starting point, tuned on INCLUDE val."""

    p_rotate: float = 0.5
    max_degrees: float = 10.0
    p_scale_xy: float = 0.5
    max_scale_xy: float = 0.1  # sx, sy in [1 - m, 1 + m], independently
    p_time_warp: float = 0.5
    warp_strength: float = 0.4
    p_crop: float = 0.3
    max_crop_frac: float = 0.1
    p_frame_drop: float = 0.3
    frame_drop_p: float = 0.1
    p_hand_dropout: float = 0.3
    hand_dropout_p: float = 0.05
    jitter_sigma: float = 0.01  # shoulder widths, always applied when > 0
    mirror: bool = False  # ablation only
    extra: dict[str, Any] = field(default_factory=dict)


def augment(clip: Clip, rng: np.random.Generator, cfg: AugmentConfig) -> Clip:
    out = clip
    if rng.random() < cfg.p_crop:
        out = temporal_crop(out, rng, cfg.max_crop_frac)
    if rng.random() < cfg.p_frame_drop:
        out = frame_drop(out, rng, cfg.frame_drop_p)
    if rng.random() < cfg.p_time_warp:
        out = time_warp(out, rng, cfg.warp_strength)
    if rng.random() < cfg.p_rotate:
        out = rotate(out, float(rng.uniform(-cfg.max_degrees, cfg.max_degrees)))
    if rng.random() < cfg.p_scale_xy:
        m = cfg.max_scale_xy
        out = scale_xy(out, float(rng.uniform(1 - m, 1 + m)), float(rng.uniform(1 - m, 1 + m)))
    if rng.random() < cfg.p_hand_dropout:
        out = hand_dropout(out, rng, cfg.hand_dropout_p)
    if cfg.jitter_sigma > 0:
        out = jitter(out, rng, cfg.jitter_sigma)
    if cfg.mirror and rng.random() < 0.5:
        out = mirror(out)
    return out if out is not clip else _copy(clip)
