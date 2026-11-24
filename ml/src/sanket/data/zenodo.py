"""The Zenodo record that hosts the INCLUDE videos (https://zenodo.org/records/4010759)."""

import json
from dataclasses import dataclass
from typing import Any

from sanket import net

RECORD_ID = 4010759
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"


@dataclass(frozen=True)
class RemoteFile:
    key: str  # file name inside the record, e.g. "Animals_1of2.zip"
    size: int  # bytes
    md5: str  # hex digest published by Zenodo


def file_url(key: str) -> str:
    return f"https://zenodo.org/records/{RECORD_ID}/files/{key}"


def parse_record(record: dict[str, Any]) -> list[RemoteFile]:
    """Extract (key, size, md5) for every file. Zenodo writes checksums as 'md5:<hex>'."""
    out = []
    for f in record["files"]:
        algo, _, digest = f["checksum"].partition(":")
        if algo != "md5" or len(digest) != 32:
            raise ValueError(f"unexpected checksum format for {f['key']}: {f['checksum']!r}")
        out.append(RemoteFile(key=f["key"], size=int(f["size"]), md5=digest))
    return sorted(out, key=lambda r: r.key)


def fetch_record() -> dict[str, Any]:
    record: dict[str, Any] = json.loads(net.get_bytes(API_URL))
    return record
