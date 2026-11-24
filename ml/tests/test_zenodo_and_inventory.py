import pytest

from sanket.data.inventory import compute_inventory, render_markdown
from sanket.data.zenodo import RemoteFile, parse_record


def test_parse_record_sorts_and_strips_md5_prefix() -> None:
    record = {
        "files": [
            {"key": "b.zip", "size": 2, "checksum": "md5:" + "b" * 32},
            {"key": "a.zip", "size": 1, "checksum": "md5:" + "a" * 32},
        ]
    }
    assert parse_record(record) == [
        RemoteFile("a.zip", 1, "a" * 32),
        RemoteFile("b.zip", 2, "b" * 32),
    ]


def test_parse_record_rejects_non_md5() -> None:
    with pytest.raises(ValueError):
        parse_record({"files": [{"key": "a.zip", "size": 1, "checksum": "sha1:abc"}]})


def test_compute_inventory_on_synthetic_data() -> None:
    files = [RemoteFile("A_1of1.zip", 100, "0" * 32), RemoteFile("README.md", 5, "1" * 32)]
    videos = [
        ("A_1of1.zip", "A/1. Dog/MVI_0001.MP4", 10),  # listed as .MOV in split -> drift
        ("A_1of1.zip", "A/2. Cat/MVI_0001.MOV", 10),  # same file name, different class
        ("A_1of1.zip", "A/2. Cat/MVI_0002.MOV", 10),  # in no split
    ]
    splits = {
        ("include", "train"): ["A/1. Dog/MVI_0001.MOV"],
        ("include", "val"): [],
        ("include", "test"): [
            "A/2. Cat/MVI_0001.MOV",
            "A/Bird (X)/MVI_0003.MOV",
        ],  # last one missing
        ("include50", "train"): [
            "A/2. Cat/MVI_0001.MOV"
        ],  # in include test -> cross-benchmark overlap
        ("include50", "val"): [],
        ("include50", "test"): ["A/1. Dog/MVI_0001.MOV"],
    }
    label_maps = {"include": {"dog": 0, "cat": 1, "birdx": 2}, "include50": {"dog": 0, "cat": 1}}

    inv = compute_inventory(files, videos, splits, label_maps)  # type: ignore[arg-type]

    assert inv["zenodo"]["n_zips"] == 1
    assert inv["zenodo"]["non_zip_files"] == ["README.md"]
    assert inv["videos_in_zips"]["n"] == 3
    assert inv["videos_in_zips"]["n_reused_file_names"] == 1
    assert inv["videos_in_zips"]["not_in_any_split"] == ["A/2. Cat/MVI_0002.MOV"]
    inc = inv["splits"]["include"]
    assert inc["official_train"] == 1 and inc["official_test"] == 2
    assert inc["missing_from_zips"]["test"] == ["A/Bird (X)/MVI_0003.MOV"]
    assert inc["classes_without_videos"] == ["birdx"]
    assert inc["n_extension_drift"] == 1
    assert inc["train_test_overlap"] == 0
    assert inv["cross_benchmark"] == {
        "include50_train_in_include_test": 1,
        "include50_test_in_include_train": 1,
    }
    assert "birdx" in render_markdown(inv)
