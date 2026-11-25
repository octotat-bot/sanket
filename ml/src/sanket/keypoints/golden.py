"""Golden file shared by the Python and TypeScript feature implementations.

    uv run python -m sanket.keypoints.golden        # (re)writes spec/golden/features_v1.json

Synthetic clips only (no dataset data), built to hit every rule of features_v1: short and long
gaps, edge gaps, low-visibility points and shoulders, missing pose, uneven timestamps, padded
1920x1088, a 4:3 frame and a 1-frame clip. Inputs are rounded to 6 decimals *before* computing
the expected outputs, so both languages start from exactly the same numbers.

`ml/tests/test_golden.py` fails if this file drifts from the Python implementation.
`web/tests/unit/features.test.ts` fails if TypeScript disagrees with it (tolerance 1e-9).
"""

import json
from typing import Any

import numpy as np

from sanket.keypoints.features import FeatureSpec, clip_to_model_input
from sanket.paths import REPO_ROOT

GOLDEN = REPO_ROOT / "spec" / "golden" / "features_v1.json"
T_GOLDEN = 12


def _clip(rng: np.random.Generator, n: int, width: int, height: int) -> dict[str, np.ndarray]:
    base_pose = rng.uniform(0.3, 0.7, size=(25, 2))
    base_pose[11], base_pose[12] = (0.58, 0.40), (0.42, 0.40)
    pose = np.zeros((n, 25, 4))
    left = np.zeros((n, 21, 3))
    right = np.zeros((n, 21, 3))
    for i in range(n):
        drift = 0.01 * i
        pose[i, :, :2] = base_pose + rng.normal(0, 0.003, size=(25, 2)) + drift
        pose[i, :, 2] = rng.normal(0, 0.05, size=25)
        pose[i, :, 3] = rng.uniform(0.6, 1.0, size=25)
        for hand, cx in ((left, 0.62), (right, 0.38)):
            hand[i, :, :2] = np.array([cx, 0.55]) + rng.normal(0, 0.03, size=(21, 2)) + 2 * drift
            hand[i, :, 2] = rng.normal(0, 0.02, size=21)
    t = np.cumsum(rng.integers(30, 50, size=n)) - 30
    return {
        "t_ms": t.astype(np.int64),
        "frame_size": np.array([width, height]),
        "pose": pose,
        "left": left,
        "right": right,
        "pose_present": np.ones(n, bool),
        "left_present": np.ones(n, bool),
        "right_present": np.ones(n, bool),
    }


def build_clips() -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    a = _clip(rng, 10, 1920, 1080)
    a["left_present"][[2, 3]] = False  # short gap → filled
    a["right_present"][[5, 6, 7, 8, 9]] = False  # trailing gap → stays missing
    a["pose"][4, 0, 3] = 0.1  # low-visibility nose
    a["pose"][7, 11, 3] = 0.2  # low-visibility left shoulder → no body frame
    b = _clip(rng, 6, 1920, 1088)  # Chrome-padded frame, uneven timestamps
    b["pose_present"][1] = False
    b["left_present"][[0, 1, 2, 3, 4]] = False  # long leading gap (5 > max_gap)
    c = _clip(rng, 1, 640, 480)  # single frame, 4:3, right hand only
    c["left_present"][0] = False
    clips = [a, b, c]
    for cl in clips:  # round inputs first: both languages then see identical numbers
        for k in ("pose", "left", "right"):
            cl[k] = np.round(cl[k], 6)
        for k, flag in (
            ("pose", "pose_present"),
            ("left", "left_present"),
            ("right", "right_present"),
        ):
            cl[k][~cl[flag]] = 0.0
    return clips


def build() -> dict[str, Any]:
    fs = FeatureSpec.from_spec()
    out: dict[str, Any] = {"T": T_GOLDEN, "n_features": fs.n_features, "clips": []}
    for cl in build_clips():
        out["clips"].append(
            {
                "input": {k: v.tolist() for k, v in cl.items()},
                "expected": clip_to_model_input(cl, fs, False, T_GOLDEN, np.float64).tolist(),
                "expected_velocity": clip_to_model_input(
                    cl, fs, True, T_GOLDEN, np.float64
                ).tolist(),
            }
        )
    return out


def main() -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(build()) + "\n")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    main()
