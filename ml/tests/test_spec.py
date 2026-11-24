"""The keypoint extractor must be identical in Python and the browser (Phase 0 gate)."""

import importlib.metadata
import json
import re
import tomllib
from pathlib import Path

import pytest

from sanket import net
from sanket.paths import REPO_ROOT, WEB_DIR
from sanket.spec import ModelPin, fetch_model, load_spec, model_pins


def test_spec_models_are_pinned_by_version_and_sha256() -> None:
    pins = model_pins(load_spec())
    assert {p.name for p in pins} >= {
        "holistic_landmarker",
        "hand_landmarker",
        "pose_landmarker_full",
    }
    for p in pins:
        assert re.fullmatch(r"[0-9a-f]{64}", p.sha256), p.name
        assert p.bytes > 0
        # "latest" is mutable: pose_landmarker_full/latest already differs from /1/.
        assert "/latest/" not in p.url, p.name
        assert p.url.endswith(f"/{p.name}.task")


def test_python_mediapipe_pin_matches_spec_and_installed_version() -> None:
    version = load_spec()["extractor"]["mediapipe_version"]
    pyproject = tomllib.loads((REPO_ROOT / "ml" / "pyproject.toml").read_text())
    assert f"mediapipe=={version}" in pyproject["project"]["dependencies"]
    assert importlib.metadata.version("mediapipe") == version


def test_web_mediapipe_pin_matches_spec() -> None:
    version = load_spec()["extractor"]["mediapipe_version"]
    pkg = json.loads((WEB_DIR / "package.json").read_text())
    assert pkg["dependencies"]["@mediapipe/tasks-vision"] == version  # exact, no ^ or ~


def test_fetch_model_rejects_wrong_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(net, "get_bytes", lambda url, timeout=0: b"not the model")
    pin = ModelPin("m", "https://example.invalid/m.task", "0" * 64, 13)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch_model(pin, tmp_path)
    assert not (tmp_path / "m.task").exists()


def test_fetch_model_skips_download_when_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"cached model bytes"
    import hashlib

    (tmp_path / "m.task").write_bytes(data)
    pin = ModelPin(
        "m", "https://example.invalid/m.task", hashlib.sha256(data).hexdigest(), len(data)
    )

    def boom(url: str, timeout: float = 0) -> bytes:
        raise AssertionError("should not download")

    monkeypatch.setattr(net, "get_bytes", boom)
    assert fetch_model(pin, tmp_path).read_bytes() == data
