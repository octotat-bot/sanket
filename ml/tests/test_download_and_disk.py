import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from sanket.data import disk
from sanket.data.disk import GB, DiskGuard, LowDiskSpace
from sanket.data.download import ChecksumMismatch, download
from sanket.data.zenodo import RemoteFile

PAYLOAD = bytes(range(256)) * 100  # 25,600 bytes
REMOTE = RemoteFile("A_1of1.zip", len(PAYLOAD), hashlib.md5(PAYLOAD).hexdigest())


def plenty(tmp_path: Path) -> DiskGuard:
    return DiskGuard(tmp_path, min_free=0)


def streamer(
    data: bytes, fail_after: int | None = None, chunk: int = 4096
) -> tuple[Callable[[str, int], Iterator[bytes]], list[int]]:
    calls: list[int] = []

    def stream(url: str, start: int) -> Iterator[bytes]:
        calls.append(start)
        sent = 0
        for i in range(start, len(data), chunk):
            if fail_after is not None and len(calls) == 1 and sent >= fail_after:
                raise ConnectionError("dropped")
            piece = data[i : i + chunk]
            sent += len(piece)
            yield piece

    return stream, calls


def test_download_resumes_with_range_after_a_dropped_connection(tmp_path: Path) -> None:
    stream, calls = streamer(PAYLOAD, fail_after=8192)
    out = download(
        REMOTE, tmp_path, plenty(tmp_path), stream=stream, sleep=lambda s: None, log=lambda m: None
    )
    assert out.read_bytes() == PAYLOAD
    assert calls == [0, 8192]  # second request resumed from the partial size
    assert not (tmp_path / "A_1of1.zip.part").exists()


def test_existing_partial_file_is_resumed(tmp_path: Path) -> None:
    (tmp_path / "A_1of1.zip.part").write_bytes(PAYLOAD[:10000])
    stream, calls = streamer(PAYLOAD)
    download(REMOTE, tmp_path, plenty(tmp_path), stream=stream)
    assert calls == [10000]


def test_complete_verified_file_is_not_downloaded_again(tmp_path: Path) -> None:
    (tmp_path / "A_1of1.zip").write_bytes(PAYLOAD)
    stream, calls = streamer(PAYLOAD)
    download(REMOTE, tmp_path, plenty(tmp_path), stream=stream)
    assert calls == []


def test_checksum_mismatch_restarts_once_then_raises(tmp_path: Path) -> None:
    bad = bytes(len(PAYLOAD))
    stream, calls = streamer(bad)
    with pytest.raises(ChecksumMismatch):
        download(REMOTE, tmp_path, plenty(tmp_path), stream=stream, log=lambda m: None)
    assert calls == [0, 0]
    assert not (tmp_path / "A_1of1.zip.part").exists()


def test_low_disk_aborts_and_removes_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "A_1of1.zip.part").write_bytes(PAYLOAD[:100])
    monkeypatch.setattr(disk, "free_bytes", lambda p: 9 * GB)
    stream, calls = streamer(PAYLOAD)
    with pytest.raises(LowDiskSpace):
        download(REMOTE, tmp_path, DiskGuard(tmp_path, min_free=10 * GB), stream=stream)
    assert calls == []
    assert not (tmp_path / "A_1of1.zip.part").exists()


def test_guard_counts_planned_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(disk, "free_bytes", lambda p: 12 * GB)
    g = DiskGuard(tmp_path, min_free=10 * GB)
    assert g.check(about_to_write=1 * GB) == 12 * GB
    with pytest.raises(LowDiskSpace):
        g.check(about_to_write=3 * GB)


def test_many_failures_with_progress_still_complete(tmp_path: Path) -> None:
    """The failure that killed the first full run: intermittent errors during a long download."""
    calls: list[int] = []

    def flaky(url: str, start: int) -> Iterator[bytes]:
        calls.append(start)
        yield PAYLOAD[start : start + 1000]  # a little progress every attempt...
        if start + 1000 < len(PAYLOAD):
            raise TimeoutError("The read operation timed out")  # ...then a drop

    out = download(
        REMOTE, tmp_path, plenty(tmp_path), stream=flaky, sleep=lambda s: None, log=lambda m: None
    )
    assert out.read_bytes() == PAYLOAD
    assert len(calls) > 20  # far more than the old limit of 5 retries


def test_gives_up_only_after_max_stall_without_progress(tmp_path: Path) -> None:
    t = [0.0]

    def dead(url: str, start: int) -> Iterator[bytes]:
        raise TimeoutError("down")
        yield b""  # pragma: no cover

    waits: list[float] = []

    def sleep(s: float) -> None:
        waits.append(s)
        t[0] += s

    with pytest.raises(TimeoutError):
        download(
            REMOTE,
            tmp_path,
            plenty(tmp_path),
            stream=dead,
            sleep=sleep,
            log=lambda m: None,
            clock=lambda: t[0],
            max_stall_s=600,
        )
    assert 600 < t[0] <= 600 + 120  # waited out the stall budget, not a fixed retry count
    assert max(waits) == 120  # backoff capped
