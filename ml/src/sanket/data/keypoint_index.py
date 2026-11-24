"""Map official split entries to extracted .npz files (Phase 1 acceptance: splits load).

The join is by `video_key` (extension-free, lower-case path), so the 605 split entries that
say .MOV where the zip had .MP4 still find their keypoints. Entries with no .npz are returned
separately and never silently dropped (for full INCLUDE, the 8 "Second (Number)" videos).
"""

from dataclasses import dataclass
from pathlib import Path

from sanket.data.splits import Dataset, Split, label_of, load_label_map, load_split, video_key


@dataclass(frozen=True)
class Item:
    path: Path  # .npz
    label: str
    label_id: int


@dataclass(frozen=True)
class SplitIndex:
    items: list[Item]
    missing: list[str]  # split entries without keypoints


def index_npz(out_dir: Path) -> dict[str, Path]:
    return {video_key(str(p.relative_to(out_dir))): p for p in out_dir.rglob("*.npz")}


def load_split_index(
    dataset: Dataset, split: Split, out_dir: Path, npz_by_key: dict[str, Path] | None = None
) -> SplitIndex:
    by_key = npz_by_key if npz_by_key is not None else index_npz(out_dir)
    label_map = load_label_map(dataset)
    items, missing = [], []
    for entry in load_split(dataset, split):
        p = by_key.get(video_key(entry))
        if p is None:
            missing.append(entry)
        else:
            lab = label_of(entry)
            items.append(Item(p, lab, label_map[lab]))
    return SplitIndex(items, missing)
