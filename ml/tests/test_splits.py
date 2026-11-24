from pathlib import Path

import pytest

from sanket.data.splits import (
    DATASETS,
    SPLITS,
    class_folder,
    label_of,
    load_label_map,
    load_split,
    read_split_file,
    video_key,
)
from sanket.paths import SPLITS_DIR
from sanket.spec import sha256_of

# The vendored files must stay byte-identical to AI4Bharat/INCLUDE@df348f4.
PINNED_SHA256 = {
    "include50_test.txt": "e8bd54ec642e2eea52f82730b4418b82268a4a34aa2df17e8bf66d54f2e1882e",
    "include50_train.txt": "ba36abcf06c5cfd2f7dc66babe8c0d46d407ebba50700ff2041b590dae2609f9",
    "include50_val.txt": "60ed8603c22a5dbc06570cd3d9d466e561d071bf402dde1f5668e061e1ba34d0",
    "include_test.txt": "48a6219b6b78c68e4831f03171ac377fcf980430bd7ea557106e111fe59d0e65",
    "include_train.txt": "1d3f0f21c10f8a6b4bf984aea5354f6841b53cc2b9a11a63fd459c1f527d8ea5",
    "include_val.txt": "75556b02dc9833e5c5a12a4e43294778eaa97f23fd72ba2114d26596ad694ea1",
    "label_map_include.json": "067a8a51369a31c19f7441844d5064ed57f2f47bf4155beda8fcceeec1d8df9a",
    "label_map_include50.json": "54fcdc3d0cb8e1d059bb41031f71cf64992d18587301f8058bed57f45a80905f",
}


@pytest.mark.parametrize("name", sorted(PINNED_SHA256))
def test_vendored_split_files_unchanged(name: str) -> None:
    assert sha256_of(SPLITS_DIR / name) == PINNED_SHA256[name]


@pytest.mark.parametrize(
    ("dataset", "train", "val", "test"),
    [("include50", 689, 77, 192), ("include", 3127, 348, 817)],
)
def test_split_counts(dataset: str, train: int, val: int, test: int) -> None:
    counts = {s: len(load_split(dataset, s)) for s in SPLITS}  # type: ignore[arg-type]
    assert counts == {"train": train, "val": val, "test": test}


def test_official_train_counts_match_zenodo_description() -> None:
    # Zenodo: "train.csv (3475) ... test.csv (817)"; "INCLUDE50 has 766 train videos and 192 test".
    assert len(load_split("include", "train")) + len(load_split("include", "val")) == 3475
    assert len(load_split("include50", "train")) + len(load_split("include50", "val")) == 766


@pytest.mark.parametrize("dataset", DATASETS)
def test_no_train_test_overlap(dataset: str) -> None:
    train = {video_key(p) for s in ("train", "val") for p in load_split(dataset, s)}  # type: ignore[arg-type]
    test = {video_key(p) for p in load_split(dataset, "test")}  # type: ignore[arg-type]
    assert not train & test


@pytest.mark.parametrize(("dataset", "n_classes"), [("include50", 50), ("include", 263)])
def test_every_split_label_is_in_label_map(dataset: str, n_classes: int) -> None:
    label_map = load_label_map(dataset)  # type: ignore[arg-type]
    assert len(label_map) == n_classes
    assert sorted(label_map.values()) == list(range(n_classes))
    labels = {label_of(p) for s in SPLITS for p in load_split(dataset, s)}  # type: ignore[arg-type]
    assert labels == set(label_map)


def test_read_split_file_tolerates_crlf_blank_lines_and_no_trailing_newline(tmp_path: Path) -> None:
    f = tmp_path / "s.txt"
    f.write_bytes(b"A/1. x/MVI_1.MOV\r\n\r\nB/2. y/MVI_2.MP4")
    assert read_split_file(f) == ["A/1. x/MVI_1.MOV", "B/2. y/MVI_2.MP4"]


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ("Animals/1. Dog/MVI_8560.MOV", "animals/1. dog/mvi_8560"),
        ("Animals/1. Dog/MVI_8560.MP4", "animals/1. dog/mvi_8560"),  # .MOV/.MP4 drift
        ("Places/19. House/Extra/MVI_3439.MOV", "places/19. house/extra/mvi_3439"),
        ("A\\1. b\\MVI_1.MOV", "a/1. b/mvi_1"),
        ("Dir.v2/1. a/noext", "dir.v2/1. a/noext"),
    ],
)
def test_video_key(path: str, key: str) -> None:
    assert video_key(path) == key


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("Days_and_Time/81. Second/MVI_5056.MOV", "second"),
        ("Days_and_Time/Second (Number)/MVI_4650.MOV", "secondnumber"),
        ("Electronics/54. Cell phone/MVI_9100.MOV", "cellphone"),
        ("Greetings/51. Good Morning/MVI_0099.MOV", "goodmorning"),
        ("Places/19. House/Extra/MVI_3439.MOV", "house"),
    ],
)
def test_label_of(path: str, label: str) -> None:
    assert label_of(path) == label


def test_class_folder_rejects_short_paths() -> None:
    with pytest.raises(ValueError):
        class_folder("MVI_1.MOV")
