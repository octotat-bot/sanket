"""The 'does OpenCV mishandle this file?' rule used to decide on PyAV (Phase 1 decision 3)."""

from typing import Any

from sanket.checks.video_metadata import compare


def ff(rotation: int = 0, n: int = 3, interval_ms: float = 40.0) -> dict[str, Any]:
    size = [1080, 1920] if abs(rotation) % 180 == 90 else [1920, 1080]
    return {
        "display_size": size,
        "n_frames": n,
        "pts_s": [i * interval_ms / 1000 for i in range(n)],
        "frame_interval_ms": {"median": interval_ms},
    }


def cv(size: list[int], times: list[float]) -> dict[str, Any]:
    return {"opened": True, "decoded_size": size, "decoded_frames": len(times), "pos_msec": times}


def test_correct_decode_passes() -> None:
    assert not compare(ff(), cv([1920, 1080], [0, 40, 80]))["mishandled"]


def test_ignored_rotation_is_flagged() -> None:
    r = compare(ff(rotation=-90), cv([1920, 1080], [0, 40, 80]))
    assert r["mishandled"] and "orientation" in r["reasons"][0]


def test_dropped_frames_are_flagged() -> None:
    r = compare(ff(n=3), cv([1920, 1080], [0, 40]))
    assert any("frames" in x for x in r["reasons"])


def test_timestamps_off_by_more_than_half_a_frame_are_flagged() -> None:
    assert compare(ff(), cv([1920, 1080], [0, 40, 101]))["mishandled"]  # 21 ms > 20 ms
    assert not compare(ff(), cv([1920, 1080], [0, 40, 99]))["mishandled"]  # 19 ms


def test_unopenable_file_is_flagged() -> None:
    assert compare(ff(), {"opened": False})["mishandled"]
