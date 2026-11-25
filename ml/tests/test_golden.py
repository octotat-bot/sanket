"""spec/golden/features_v1.json must match the Python implementation exactly (no silent drift).
The TypeScript side checks the same file in web/tests/unit/features.test.ts."""

import json

import numpy as np

from sanket.keypoints.golden import GOLDEN, build


def test_golden_file_matches_python_implementation() -> None:
    committed = json.loads(GOLDEN.read_text())
    fresh = build()
    assert committed["T"] == fresh["T"] and committed["n_features"] == fresh["n_features"]
    assert len(committed["clips"]) == len(fresh["clips"]) == 3
    for c, f in zip(committed["clips"], fresh["clips"], strict=True):
        for key in ("expected", "expected_velocity"):
            np.testing.assert_allclose(np.array(c[key]), np.array(f[key]), atol=1e-12)


def test_golden_outputs_are_finite_and_exercise_missing_parts() -> None:
    g = json.loads(GOLDEN.read_text())
    flags = np.concatenate([np.array(c["expected"])[:, -3:] for c in g["clips"]])
    assert np.isfinite(flags).all()
    assert (flags == 0).any() and (flags == 1).any()  # both present and missing parts covered
