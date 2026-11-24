import json
import zipfile
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np
import pytest

from sanket.data.disk import DiskGuard
from sanket.keypoints.extractor import N_HAND, N_POSE, Sequence
from sanket.pipeline.extract import Config, process_zip
from sanket.spec import load_spec


class FakeExtractor:
    """Deterministic stand-in: 'detects' a left hand on even frames only."""

    name = "fake"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def add(self, rgb: np.ndarray, t_ms: int, seq: Sequence) -> None:
        i = len(seq.t_ms)
        seq.t_ms.append(t_ms)
        seq.pose.append(np.full((N_POSE, 4), 0.5, np.float32))
        seq.left.append(np.full((N_HAND, 3), float(rgb.mean()) / 255, np.float32))
        seq.right.append(np.zeros((N_HAND, 3), np.float32))
        seq.pose_present.append(True)
        seq.left_present.append(i % 2 == 0)
        seq.right_present.append(False)


class NoHands(FakeExtractor):
    def add(self, rgb: np.ndarray, t_ms: int, seq: Sequence) -> None:
        super().add(rgb, t_ms, seq)
        seq.left_present[-1] = False


def make_video(path: Path, n: int = 10) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i * 20, np.uint8))
    w.release()


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Config]:
    vid = tmp_path / "v.mp4"
    make_video(vid)
    zp = tmp_path / "A_1of1.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.write(vid, "Animals/1. Dog/MVI_0001.MP4")
        zf.writestr("Animals/2. Cat/MVI_0002.MOV", b"not a video at all")
        zf.writestr("Animals/README.txt", b"ignored")
    cfg = Config(
        extractor="holistic",
        raw_dir=tmp_path / "raw",
        out_dir=tmp_path / "out",
        frame_cache_dir=tmp_path / "frames",
        min_free_gb=0,
        review_below_hand_rate=0.05,
    )
    cfg.raw_dir.mkdir()
    cfg.out_dir.mkdir()
    return zp, cfg


def run(zp: Path, cfg: Config, ex: type = FakeExtractor) -> dict[str, int]:
    return process_zip(
        zp,
        zp.name,
        cfg,
        ex,
        DiskGuard(cfg.raw_dir, min_free=0),
        {"extractor": "fake"},
        lambda m: None,
    )


def test_good_video_saved_corrupt_video_logged_and_batch_continues(
    setup: tuple[Path, Config],
) -> None:
    zp, cfg = setup
    assert run(zp, cfg) == {"ok": 1, "error": 1, "skipped": 0, "review": 0}

    npz = np.load(cfg.out_dir / "Animals/1. Dog/MVI_0001.npz")
    assert npz["pose"].shape == (10, N_POSE, 4) and npz["left"].shape == (10, N_HAND, 3)
    assert npz["t_ms"].tolist() == [i * 40 for i in range(10)]  # 25 fps, strictly increasing
    assert npz["left_present"].tolist() == [i % 2 == 0 for i in range(10)]
    assert not np.isnan(npz["pose"]).any()
    meta = json.loads(str(npz["meta"]))
    assert meta["label"] == "dog" and meta["extractor"] == "fake" and len(meta["sha256"]) == 64

    log = [json.loads(line) for line in (cfg.out_dir / "_log.jsonl").read_text().splitlines()]
    by = {r["path"]: r for r in log}
    assert by["Animals/1. Dog/MVI_0001.MP4"]["status"] == "ok"
    assert by["Animals/1. Dog/MVI_0001.MP4"]["rate_any_hand"] == 0.5
    assert by["Animals/2. Cat/MVI_0002.MOV"]["status"] == "error"
    assert (cfg.frame_cache_dir / "Animals/1. Dog/MVI_0001.jpg").exists()  # type: ignore[operator]
    assert list(cfg.raw_dir.iterdir()) == []  # temp videos removed


def test_rerun_skips_finished_videos(setup: tuple[Path, Config]) -> None:
    zp, cfg = setup
    run(zp, cfg)
    assert run(zp, cfg)["skipped"] == 1


def test_videos_without_hands_are_saved_and_listed_for_review(setup: tuple[Path, Config]) -> None:
    zp, cfg = setup
    assert run(zp, cfg, NoHands)["review"] == 1
    review = (cfg.out_dir / "_review.jsonl").read_text()
    assert "MVI_0001" in review
    assert (cfg.out_dir / "Animals/1. Dog/MVI_0001.npz").exists()


def test_extraction_is_deterministic(setup: tuple[Path, Config], tmp_path: Path) -> None:
    zp, cfg = setup
    run(zp, cfg)
    cfg2 = Config(**{**cfg.__dict__, "out_dir": tmp_path / "out2"})
    cfg2.out_dir.mkdir()
    run(zp, cfg2)
    a = np.load(cfg.out_dir / "Animals/1. Dog/MVI_0001.npz")
    b = np.load(cfg2.out_dir / "Animals/1. Dog/MVI_0001.npz")
    for k in ("t_ms", "pose", "left", "right", "left_present"):
        assert np.array_equal(a[k], b[k]), k


def test_config_refuses_to_run_without_an_extractor_choice(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "extractor: null\nraw_dir: r\nout_dir: o\nmin_free_gb: 10\nreview_below_hand_rate: 0.05\n"
    )
    with pytest.raises(SystemExit, match="ADR 002"):
        Config.load(p)


def test_spec_pose_names_match_mediapipe_enum() -> None:
    from mediapipe.tasks.python.vision import PoseLandmark

    pose = load_spec()["stored"]["pose"]
    assert pose["names"] == [PoseLandmark(i).name.lower() for i in pose["indices"]]


def test_upload_refuses_incomplete_extraction_or_media(tmp_path: Path) -> None:
    from sanket.pipeline.upload import N_ZIPS, check_complete, metadata

    with pytest.raises(SystemExit, match="incomplete"):
        check_complete(tmp_path)
    (tmp_path / "_done_zips.txt").write_text("\n".join(f"z{i}.zip" for i in range(N_ZIPS)))
    check_complete(tmp_path)
    (tmp_path / "a.jpg").write_bytes(b"x")
    with pytest.raises(SystemExit, match="refusing"):
        check_complete(tmp_path)
    m = metadata("me/ds", "holistic")
    assert m["licenses"] == [{"name": "CC-BY-4.0"}] and "INCLUDE" in str(m["description"])


def test_split_index_joins_by_key_and_reports_missing(tmp_path: Path) -> None:
    from sanket.data.keypoint_index import load_split_index
    from sanket.data.splits import load_split

    entries = load_split("include50", "test")
    # Save keypoints for all but one entry, with the extension flipped (.MOV <-> .MP4 drift).
    for e in entries[1:]:
        p = tmp_path / Path(e).with_suffix(".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    ix = load_split_index("include50", "test", tmp_path)
    assert len(ix.items) == len(entries) - 1 and ix.missing == [entries[0]]
    assert all(0 <= it.label_id < 50 for it in ix.items)


def test_data_report_counts_duplicates_errors_and_review(tmp_path: Path) -> None:
    from sanket.data.report import build

    base = {
        "status": "ok",
        "fps": 25.0,
        "width": 1920,
        "height": 1080,
        "n_frames": 60,
        "rate_pose": 1.0,
        "rate_left": 0.5,
        "rate_right": 0.5,
        "rate_any_hand": 0.8,
        "rate_both_hands": 0.2,
    }
    recs: list[dict[str, Any]] = [
        {**base, "path": "A/1. Dog/MVI_1.MOV", "label": "dog", "sha256": "h1"},
        {**base, "path": "A/2. Cat/MVI_2.MOV", "label": "cat", "sha256": "h1"},  # duplicate
        {
            **base,
            "path": "A/2. Cat/MVI_3.MOV",
            "label": "cat",
            "sha256": "h3",
            "rate_any_hand": 0.0,
        },
        {"status": "error", "path": "A/2. Cat/MVI_4.MOV", "label": "cat", "error": "bad"},
    ]
    md, s = build(tmp_path, recs)
    assert s["videos_ok"] == 3 and s["videos_error"] == 1
    assert list(s["duplicates"].values()) == [["A/1. Dog/MVI_1.MOV", "A/2. Cat/MVI_2.MOV"]]
    assert s["review"] == ["A/2. Cat/MVI_3.MOV"]
    assert "bad" in md


def test_run_refuses_a_dirty_working_tree(
    setup: tuple[Path, Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sanket.pipeline import extract

    _, cfg = setup
    monkeypatch.setattr(extract, "_git_sha", lambda: "abc123-dirty")
    with pytest.raises(SystemExit, match="uncommitted"):
        extract.run(cfg, only_zips=[])
