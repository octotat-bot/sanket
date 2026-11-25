"""Normalization, step 1: aspect-correct coordinates, then body-relative (Phase 2 builds on this).

MediaPipe normalizes x by the width and y by the height **of the image it was given**, and z
"uses roughly the same scale as x" (tasks-vision NormalizedLandmark docs). So a 16:9 frame and a
4:3 frame squash x differently, and the same body produces different numbers. Before any
geometry we convert to *aspect-correct* units, where one unit = the frame height in both axes:

    x' = x * W / H        y' = y        z' = z * W / H

W and H must be the size of the image MediaPipe actually processed. That is not the container's
claim: Chrome on macOS hands MediaPipe 1920x1088 for 1920x1080 H.264 (8 padding rows at the
bottom, content unmoved; ADR 002). With the true size, padding only changes the unit by a
uniform factor, and step 2 removes it:

    body-relative = (p' - shoulder_midpoint') / shoulder_width'

Shoulders are pose landmarks 11 (the signer's left) and 12 (right), at the same positions in
the stored pose array (spec "stored.pose.indices" keeps 0-24 in order).

The TypeScript mirror and its golden-file test come in Phase 2, from the same spec section.
"""

import numpy as np

L_SHOULDER, R_SHOULDER = 11, 12


def aspect_correct(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """(..., 2 or 3+) normalized MediaPipe coords → frame-height units.

    Extra columns (e.g. visibility) pass through unchanged.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")
    out = np.array(points, dtype=np.float64, copy=True)
    out[..., 0] *= width / height
    if out.shape[-1] >= 3:
        out[..., 2] *= width / height
    return out


def body_relative(
    points: np.ndarray, pose: np.ndarray, width: int, height: int, min_shoulder: float = 1e-6
) -> np.ndarray:
    """Center on the shoulder midpoint and scale by shoulder width, in aspect-correct space.

    `points`: (..., D) normalized coords to transform; `pose`: (N_POSE, D) normalized pose of
    the same frame. Only x, y (and z if present) are transformed; other columns pass through.
    Raises if the shoulders coincide (callers treat such frames as missing pose).
    """
    p = aspect_correct(points, width, height)
    s = aspect_correct(pose, width, height)
    d = min(p.shape[-1], 3)
    mid = (s[L_SHOULDER, :d] + s[R_SHOULDER, :d]) / 2
    sw = float(np.linalg.norm(s[L_SHOULDER, :2] - s[R_SHOULDER, :2]))
    if sw < min_shoulder:
        raise ValueError("shoulder width ~0: pose unusable for normalization")
    p[..., :d] = (p[..., :d] - mid) / sw
    return p
