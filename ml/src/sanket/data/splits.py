"""Load the official INCLUDE / INCLUDE-50 split files and match them to videos.

Rules (see ml/data/splits/PROVENANCE.md for why each exists):

* A video's identity is its full relative path without extension, lower-cased
  (`video_key`). File names alone are NOT unique: the camera counter (MVI_0001..9999)
  wraps, so the same MVI number appears in different classes.
* The label is derived like the authors' code does: keep only the letters of the class
  folder name and lower-case them ("81. Second" -> "second",
  "Second (Number)" -> "secondnumber", "Cell phone" -> "cellphone").
* The authors' `train` + `val` files make up the official training set.
"""

import json
from pathlib import Path
from typing import Literal

from sanket.paths import SPLITS_DIR

Dataset = Literal["include", "include50"]
Split = Literal["train", "val", "test"]
DATASETS: tuple[Dataset, ...] = ("include", "include50")
SPLITS: tuple[Split, ...] = ("train", "val", "test")


def read_split_file(path: Path) -> list[str]:
    """One relative video path per line; tolerates CRLF, blank lines, no trailing newline."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_split(dataset: Dataset, split: Split, splits_dir: Path = SPLITS_DIR) -> list[str]:
    return read_split_file(splits_dir / f"{dataset}_{split}.txt")


def load_label_map(dataset: Dataset, splits_dir: Path = SPLITS_DIR) -> dict[str, int]:
    label_map: dict[str, int] = json.loads((splits_dir / f"label_map_{dataset}.json").read_text())
    return label_map


def video_key(rel_path: str) -> str:
    """Extension-free, lower-cased, forward-slash path: the join key between splits and zips."""
    p = rel_path.strip().replace("\\", "/").lower()
    stem, dot, ext = p.rpartition(".")
    if dot and "/" not in ext:
        return stem
    return p


def class_folder(rel_path: str) -> str:
    """'Category/Class/[Extra/]file' -> 'Class'."""
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) < 3:
        raise ValueError(f"expected Category/Class/file, got {rel_path!r}")
    return parts[1]


def label_of(rel_path: str) -> str:
    return "".join(ch for ch in class_folder(rel_path) if ch.isalpha()).lower()
