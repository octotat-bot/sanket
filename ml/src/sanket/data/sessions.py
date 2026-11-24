"""Camera-number runs: the raw material for session-independent splits.

INCLUDE file names are camera counters (MVI_0001 .. MVI_9999). Within one class, consecutive
takes have consecutive numbers, so a "run" (numbers at most `max_gap` apart) is probably one
recording session of one signer. Whether that is true is checked visually in Phase 1
(docs/decisions/005-session-independent-evaluation.md). This module only finds the runs.

We call held-out groups "sessions", never "signers": INCLUDE has no signer ids.
"""

import csv
import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_NUMBER = re.compile(r"MVI_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Video:
    path: str  # relative path inside the zips, e.g. "Seasons/61. Summer/MVI_4565.MOV"
    zip: str  # Zenodo zip that contains it
    bytes: int

    @property
    def category(self) -> str:
        return self.path.split("/")[0]

    @property
    def class_folder(self) -> str:
        return self.path.split("/")[1]

    @property
    def ext(self) -> str:
        return Path(self.path).suffix.upper()

    @property
    def number(self) -> int | None:
        """Camera counter, or None if the name doesn't follow the MVI_#### pattern."""
        m = _NUMBER.search(Path(self.path).name)
        return int(m.group(1)) if m else None

    @property
    def is_renamed_copy(self) -> bool:
        """Names like 'MVI_0050_ (1).MOV' or 'MVI_0057_.MOV': copies, not plain camera names."""
        return re.fullmatch(r"MVI_\d{4}", Path(self.path).stem, re.IGNORECASE) is None


def load_manifest(path: Path) -> list[Video]:
    with path.open() as f:
        return [
            Video(path=r["path"], zip=r["zip"], bytes=int(r["bytes"]))
            for r in csv.DictReader(f, delimiter="\t")
        ]


def class_runs(videos: list[Video], max_gap: int = 3) -> list[list[Video]]:
    """Split each (class, extension) into runs of camera numbers at most `max_gap` apart.

    The extension is part of the key because .MOV and .MP4 come from different counters.
    A repeated number (gap 0) starts a new run: it means another counter cycle, not the
    next take. Renamed copies and unnumbered files become runs of their own, so they can
    never silently join a session.
    """
    groups: dict[tuple[str, str], list[Video]] = defaultdict(list)
    singletons: list[list[Video]] = []
    for v in videos:
        if v.number is None or v.is_renamed_copy:
            singletons.append([v])
        else:
            groups[(v.class_folder, v.ext)].append(v)

    runs: list[list[Video]] = []
    for key in sorted(groups):
        vs = sorted(groups[key], key=lambda v: (v.number, v.path))
        current = [vs[0]]
        for prev, nxt in itertools.pairwise(vs):
            gap = (nxt.number or 0) - (prev.number or 0)
            if 0 < gap <= max_gap:
                current.append(nxt)
            else:
                runs.append(current)
                current = [nxt]
        runs.append(current)
    return runs + sorted(singletons, key=lambda r: r[0].path)
