"""Phase 0 gate (Python side): the pinned models load and run in VIDEO mode.

Needs the model files (~31 MB, fetched and sha256-checked into ml/.cache/models).
The browser side of the same check is web/tests/e2e/mediapipe-smoke.spec.ts.
"""

import numpy as np
import pytest

from sanket.paths import CACHE_DIR
from sanket.spec import fetch_model, load_spec, model_pins

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def model_paths() -> dict[str, str]:
    return {p.name: str(fetch_model(p, CACHE_DIR / "models")) for p in model_pins(load_spec())}


def test_holistic_hand_and_pose_landmarkers_run(model_paths: dict[str, str]) -> None:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    blank = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), np.uint8))
    cases = [
        ("holistic_landmarker", vision.HolisticLandmarker, vision.HolisticLandmarkerOptions),
        ("hand_landmarker", vision.HandLandmarker, vision.HandLandmarkerOptions),
        ("pose_landmarker_full", vision.PoseLandmarker, vision.PoseLandmarkerOptions),
    ]
    for name, cls, options in cases:
        opts = options(
            base_options=BaseOptions(model_asset_path=model_paths[name]),
            running_mode=vision.RunningMode.VIDEO,
        )
        with cls.create_from_options(opts) as landmarker:
            for i in range(3):  # VIDEO mode needs strictly increasing timestamps
                result = landmarker.detect_for_video(blank, i * 33)
        # A blank frame has no person: the call must succeed and return empty landmarks.
        landmarks = getattr(result, "pose_landmarks", None) or getattr(
            result, "hand_landmarks", None
        )
        assert not landmarks, name
