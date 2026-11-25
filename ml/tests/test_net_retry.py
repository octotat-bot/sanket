import urllib.error

import pytest

from sanket.net import with_retries


def test_retries_transient_errors_then_succeeds() -> None:
    calls: list[int] = []
    waits: list[float] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError("_ssl.c:993: The handshake operation timed out")
        return "ok"

    assert with_retries(flaky, sleep=waits.append, log=lambda m: None) == "ok"
    assert len(calls) == 3 and waits == [2.0, 4.0]


def test_gives_up_after_the_last_attempt() -> None:
    def down() -> None:
        raise TimeoutError("down")

    with pytest.raises(TimeoutError):
        with_retries(down, attempts=3, sleep=lambda s: None, log=lambda m: None)


def test_non_network_errors_are_not_retried() -> None:
    calls: list[int] = []

    def bug() -> None:
        calls.append(1)
        raise ValueError("bug")

    with pytest.raises(ValueError):
        with_retries(bug, sleep=lambda s: None, log=lambda m: None)
    assert calls == [1]
