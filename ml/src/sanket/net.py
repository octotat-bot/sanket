"""Minimal HTTP helpers on the standard library (no extra dependency).

Kept tiny on purpose: GET a whole small resource, GET a byte range, and HEAD for size.
"""

import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

DEFAULT_TIMEOUT_S = 60
USER_AGENT = "sanket-data-tools/0.1"

# Some Python builds (e.g. python.org macOS installers) ship without a CA bundle.
# Fall back to the OS bundle when it exists; otherwise use Python's defaults.
_SYSTEM_CA = Path("/etc/ssl/cert.pem")


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if _SYSTEM_CA.exists():
        ctx.load_verify_locations(cafile=str(_SYSTEM_CA))
    return ctx


def _open(req: urllib.request.Request, timeout: float) -> bytes:
    req.add_header("User-Agent", USER_AGENT)
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        data: bytes = resp.read()
        return data


def get_bytes(url: str, timeout: float = DEFAULT_TIMEOUT_S) -> bytes:
    return _open(urllib.request.Request(url), timeout)


def get_range(
    url: str, start: int, end_inclusive: int, timeout: float = DEFAULT_TIMEOUT_S
) -> bytes:
    """GET bytes [start, end_inclusive]. Raises if the server ignores the Range header."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
    req.add_header("User-Agent", USER_AGENT)
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        if resp.status != 206:
            raise RuntimeError(f"server ignored Range request (HTTP {resp.status}) for {url}")
        data: bytes = resp.read()
        return data


def content_length(url: str, timeout: float = DEFAULT_TIMEOUT_S) -> int:
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", USER_AGENT)
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return int(resp.headers["Content-Length"])


RETRYABLE: tuple[type[BaseException], ...] = (urllib.error.URLError, TimeoutError, ConnectionError)


def with_retries[T](
    fn: Callable[[], T],
    attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> T:
    """Call fn(), retrying transient network errors (e.g. a TLS handshake timeout from Zenodo)
    with exponential backoff: 2, 4, 8, 16 s. The last error is re-raised."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except RETRYABLE as e:
            if attempt == attempts:
                raise
            wait = float(2**attempt)
            log(f"network error ({e!r}); retry {attempt}/{attempts - 1} in {wait:.0f}s")
            sleep(wait)
    raise AssertionError("unreachable")
