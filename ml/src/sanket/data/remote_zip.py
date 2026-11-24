"""List the members of a remote zip without downloading it.

A zip's table of contents (the "central directory") sits at the end of the file. Python's
`zipfile` only needs `seek` + `read`, so we give it a file-like object whose reads become
HTTP Range requests. Listing a 1.4 GB zip this way transfers well under 1 MB.
"""

import io
import zipfile
from collections.abc import Callable
from dataclasses import dataclass

from sanket import net

# (start, end_inclusive) -> bytes
RangeFetcher = Callable[[int, int], bytes]


class RangeReader(io.RawIOBase):
    """Read-only, seekable file object backed by a byte-range fetch function."""

    def __init__(self, size: int, fetch: RangeFetcher) -> None:
        self._size = size
        self._fetch = fetch
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        return self._pos

    def readinto(self, buffer: "bytearray | memoryview") -> int:  # type: ignore[override]
        n = min(len(buffer), self._size - self._pos)
        if n <= 0:
            return 0
        data = self._fetch(self._pos, self._pos + n - 1)
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


@dataclass(frozen=True)
class ZipMember:
    name: str
    size: int  # uncompressed bytes


def list_members(size: int, fetch: RangeFetcher) -> list[ZipMember]:
    reader = io.BufferedReader(RangeReader(size, fetch), buffer_size=1 << 16)
    with zipfile.ZipFile(reader) as zf:
        return [ZipMember(i.filename, i.file_size) for i in zf.infolist() if not i.is_dir()]


def list_remote_zip(url: str) -> list[ZipMember]:
    return list_members(net.content_length(url), lambda a, b: net.get_range(url, a, b))
