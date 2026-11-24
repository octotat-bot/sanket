import io
import os
import zipfile

from sanket.data.remote_zip import RangeReader, list_members


def _make_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("Seasons/", b"")
        zf.writestr("Seasons/61. Summer/MVI_4565.MOV", os.urandom(2_000_000))
        zf.writestr("Seasons/61. Summer/MVI_9122.MP4", b"x" * 10)
    return buf.getvalue()


def test_list_members_reads_only_the_table_of_contents() -> None:
    blob = _make_zip()
    fetched: list[int] = []

    def fetch(a: int, b: int) -> bytes:
        fetched.append(b - a + 1)
        return blob[a : b + 1]

    members = list_members(len(blob), fetch)
    assert [(m.name, m.size) for m in members] == [
        ("Seasons/61. Summer/MVI_4565.MOV", 2_000_000),
        ("Seasons/61. Summer/MVI_9122.MP4", 10),
    ]
    # Directories are skipped, and we never pull the 2 MB payload.
    assert sum(fetched) < 200_000


def test_range_reader_seek_and_read() -> None:
    data = bytes(range(100))
    r = io.BufferedReader(RangeReader(len(data), lambda a, b: data[a : b + 1]))
    r.seek(-10, io.SEEK_END)
    assert r.read() == data[90:]
    r.seek(5)
    assert r.read(3) == data[5:8]
