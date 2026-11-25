"""spec/golden/features_v1.json must match the Python implementation exactly (no silent drift).
The TypeScript side checks the same file in web/tests/unit/features.test.ts."""

import json

import numpy as np

from sanket.keypoints.golden import GOLDEN, build


def test_golden_file_matches_python_implementation() -> None:
    committed = json.loads(GOLDEN.read_text())
    fresh = build()
    assert committed["T"] == fresh["T"] and committed["n_features"] == fresh["n_features"]
    assert len(committed["clips"]) == len(fresh["clips"]) == 4
    for c, f in zip(committed["clips"], fresh["clips"], strict=True):
        for key in ("expected", "expected_velocity"):
            np.testing.assert_allclose(np.array(c[key]), np.array(f[key]), atol=1e-12)


def test_golden_outputs_are_finite_and_exercise_missing_parts() -> None:
    g = json.loads(GOLDEN.read_text())
    flags = np.concatenate([np.array(c["expected"])[:, -3:] for c in g["clips"]])
    assert np.isfinite(flags).all()
    assert (flags == 0).any() and (flags == 1).any()  # both present and missing parts covered


def test_golden_15fps_clip_exercises_the_millisecond_gap_rule() -> None:
    from sanket.keypoints.features import FeatureSpec, fill_short_gaps
    from sanket.keypoints.golden import build_clips

    d = build_clips()[3]
    out = fill_short_gaps(d, FeatureSpec.from_spec().max_gap_ms)
    assert bool(out["right_present"][2])  # 1 frame missing at 15 fps (67 ms): filled
    assert not out["right_present"][[4, 5]].any()  # 2 frames (133 ms > 120 ms): not filled
