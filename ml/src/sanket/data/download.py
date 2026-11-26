"""Resumable, md5-verified download of one Zenodo zip at a time.

* Bytes go to `<name>.part`. After an interruption, the next call resumes with an HTTP
  Range request from the `.part` file's current size.
* When complete, the md5 is checked against Zenodo's published checksum. On mismatch the
  `.part` is deleted and the download restarts from zero once. A second mismatch raises.
* The disk guard runs before starting and every chunk. If it trips, the partial file is
  deleted (giving the space back matters more than resuming) and LowDiskSpace propagates.
"""

import hashlib
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

from sanket import net
from sanket.data.disk import DiskGuard, LowDiskSpace
from sanket.data.zenodo import RemoteFile, file_url

CHUNK = 8 * 1024 * 1024

# (url, start_offset) -> iterator of byte chunks from start_offset to the end
Streamer = Callable[[str, int], Iterator[bytes]]


class ChecksumMismatch(RuntimeError):
    pass


def http_stream(url: str, start: int) -> Iterator[bytes]:
    headers = {"User-Agent": net.USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120, context=net._ssl_context()) as resp:
        if start and resp.status != 206:
            raise RuntimeError(f"server ignored Range (HTTP {resp.status}); refusing to append")
        while chunk := resp.read(CHUNK):
            yield chunk


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(
    remote: RemoteFile,
    dest_dir: Path,
    guard: DiskGuard,
    stream: Streamer = http_stream,
    max_stall_s: float = 3600.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> Path:
    """Survives outages (Wi-Fi drops, the Mac sleeping): retries with backoff (2, 4, ... capped at
    120 s) for as long as bytes keep arriving, and gives up only after `max_stall_s` seconds with
    no progress at all. The failure count resets whenever a retry made progress."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / remote.key
    if final.exists() and final.stat().st_size == remote.size and md5_of(final) == remote.md5:
        return final
    part = dest_dir / (remote.key + ".part")
    url = file_url(remote.key)

    for _full_pass in range(2):  # a second full pass only after a checksum mismatch
        failures = 0
        last_progress_t = clock()
        last_size = part.stat().st_size if part.exists() else 0
        while True:
            have = part.stat().st_size if part.exists() else 0
            if have > last_size:  # progress since the last failure: reset the stall clock
                last_size, last_progress_t, failures = have, clock(), 0
            if have > remote.size:
                part.unlink()  # corrupt / foreign partial: start over
                have = 0
            if have == remote.size:
                break
            try:
                guard.check(about_to_write=remote.size - have)
                with part.open("ab") as f:
                    for chunk in stream(url, have):
                        f.write(chunk)
                        guard.check()
            except LowDiskSpace:
                part.unlink(missing_ok=True)
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                now_size = part.stat().st_size if part.exists() else 0
                if now_size > last_size:
                    last_size, last_progress_t, failures = now_size, clock(), 0
                stalled = clock() - last_progress_t
                if stalled > max_stall_s:
                    log(f"{remote.key}: no progress for {stalled:.0f}s; giving up")
                    raise
                failures += 1
                wait = min(2**failures, 120)
                log(f"{remote.key}: {e!r}; resuming in {wait}s (no progress for {stalled:.0f}s)")
                sleep(wait)
        got = md5_of(part)
        if got == remote.md5:
            part.replace(final)
            return final
        log(f"{remote.key}: md5 {got} != {remote.md5}; deleting and restarting")
        part.unlink()
    raise ChecksumMismatch(f"{remote.key}: md5 mismatch twice")
