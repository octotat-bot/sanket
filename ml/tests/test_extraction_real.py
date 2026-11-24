"""Phase 1 acceptance: the real extractors are deterministic (same video → identical arrays).

Fetches one INCLUDE-50 test video (~10-20 MB) out of its Zenodo zip with Range requests.
"""

from pathlib import Path

import numpy as np
import pytest

from sanket.checks.sample import MANIFEST, VIDEO_CACHE, fetch
from sanket.data.sessions import load_manifest
from sanket.data.splits import load_split, video_key
from sanket.keypoints.extractor import EXTRACTORS
from sanket.paths import CACHE_DIR
from sanket.pipeline.extract import extract_video
from sanket.spec import fetch_model, load_spec, model_pins

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def video() -> Path:
    videos = load_manifest(MANIFEST)
    by_key = {video_key(v.path): v for v in videos}
    first = next(
        by_key[video_key(p)] for p in load_split("include50", "test") if video_key(p) in by_key
    )
    fetch([first.path], videos)
    return VIDEO_CACHE / first.path


@pytest.fixture(scope="module")
def models() -> dict[str, Path]:
    return {p.name: fetch_model(p, CACHE_DIR / "models") for p in model_pins(load_spec())}


@pytest.mark.parametrize("name", sorted(EXTRACTORS))
def test_real_extractor_is_deterministic_and_never_nan(
    name: str, video: Path, models: dict[str, Path]
) -> None:
    cls = EXTRACTORS[name]
    a, sa = extract_video(video, lambda: cls(models), None)
    b, _ = extract_video(video, lambda: cls(models), None)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"{name}: {k} differs between runs"
    assert sa["n_frames"] > 10
    assert sa["rate_pose"] > 0.9 and sa["rate_any_hand"] > 0.3
    for k in ("pose", "left", "right"):
        assert not np.isnan(a[k]).any()
    # Visibility must be real numbers where pose is present (Phase 2 thresholds on it).
    vis = a["pose"][a["pose_present"]][:, :, 3]
    assert vis.max() > 0.5, f"{name}: pose visibility looks unset"
