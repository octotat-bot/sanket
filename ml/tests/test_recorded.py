import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sanket.data.recorded import convert, load_spec, validate_session
from sanket.keypoints.features import FeatureSpec, clip_to_model_input


def session(n_frames: int = 12) -> dict[str, Any]:
    pose = [[0.5, 0.5, 0.0, 0.9]] * 25
    hand = [[0.4, 0.6, 0.0]] * 21
    clip = {
        "clip_id": "C-ABCDEF",
        "kind": "sign",
        "label": "hello",
        "label_id": 22,
        "take": 1,
        "prompt": "Hello",
        "started_at": "2026-11-25T10:00:00.000Z",
        "frame_size": [640, 480],
        "camera_size": [640, 480],
        "t_ms": [i * 33.333 for i in range(n_frames)],
        "pose": [pose] * n_frames,
        "left": [hand] * n_frames,
        "right": [[[0.0, 0.0, 0.0]] * 21] * n_frames,
        "pose_present": [True] * n_frames,
        "left_present": [True] * n_frames,
        "right_present": [False] * n_frames,
    }
    return {
        "format": "sanket-recording-v1",
        "session_id": "S-ABC234",
        "participant_id": "P-ABC234",
        "consent": {
            "version": "1",
            "accepted_at": "2026-11-25T10:00:00.000Z",
            "text_sha256": "a" * 64,
        },
        "created_at": "2026-11-25T10:00:00.000Z",
        "app": {"commit": None, "user_agent": "test"},
        "spec_version": "0.7.0",
        "mediapipe_version": "1.2.3",
        "model_sha256": {"holistic_landmarker": "b" * 64},
        "delegate": "GPU",
        "extract_height": 480,
        "clips": [clip],
    }


def test_valid_session_converts_to_the_stored_layout(tmp_path: Path) -> None:
    f = tmp_path / "s.json"
    f.write_text(json.dumps(session()))
    [npz] = convert(f, tmp_path / "out")
    d = dict(np.load(npz))
    assert d["pose"].shape == (12, 25, 4) and d["left"].shape == (12, 21, 3)
    assert d["frame_size"].tolist() == [640, 480] and int(d["label_id"]) == 22
    meta = json.loads(str(d["meta"]))
    assert meta["participant_id"] == "P-ABC234" and "name" not in meta
    # Flows through the same feature code as INCLUDE keypoints.
    x = clip_to_model_input(d, FeatureSpec.from_spec(), T=16)
    assert x.shape == (16, FeatureSpec.from_spec().n_features) and np.isfinite(x).all()


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda s: s.update(participant_id="Mukund Mangla"), "participant_id"),
        (lambda s: s.update(image="data:image/png;base64,AAAA"), "forbidden key image"),
        (lambda s: s["clips"][0].update(t_ms=[0, 5, 3] + [10] * 9), "t_ms not increasing"),
        (lambda s: s["clips"][0].update(pose=[[[0.5] * 4] * 24] * 12), "pose shape"),
        (lambda s: s.pop("consent"), "missing consent"),
    ],
)
def test_invalid_sessions_are_rejected(mutate: Any, error: str, tmp_path: Path) -> None:
    s = session()
    mutate(s)
    errs = validate_session(s, load_spec())
    assert any(error in e for e in errs), errs
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(s))
    with pytest.raises(ValueError):
        convert(f, tmp_path / "out")
