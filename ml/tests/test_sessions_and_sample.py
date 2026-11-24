from sanket.checks.sample import MANIFEST, select
from sanket.data.sessions import Video, class_runs, load_manifest


def v(path: str) -> Video:
    return Video(path=path, zip="z.zip", bytes=1)


def paths(runs: list[list[Video]]) -> list[list[str]]:
    return [[x.path.split("/")[-1] for x in r] for r in runs]


def test_runs_split_on_gaps_larger_than_max_gap() -> None:
    vids = [v(f"S/61. Summer/MVI_{n:04d}.MOV") for n in (4565, 4566, 4568, 4990, 4991)]
    assert paths(class_runs(vids)) == [
        ["MVI_4565.MOV", "MVI_4566.MOV", "MVI_4568.MOV"],
        ["MVI_4990.MOV", "MVI_4991.MOV"],
    ]


def test_runs_never_mix_classes_or_cameras() -> None:
    vids = [
        v("S/61. Summer/MVI_4568.MOV"),
        v("S/62. Spring/MVI_4569.MOV"),  # next number, other class
        v("S/61. Summer/MVI_4569.MP4"),  # next number, other camera
    ]
    assert len(class_runs(vids)) == 3


def test_repeated_number_starts_a_new_run_and_renamed_copies_stand_alone() -> None:
    vids = [
        v("G/52. Good afternoon/MVI_0050.MOV"),
        v("G/52. Good afternoon/MVI_0050_ (1).MOV"),
        v("G/52. Good afternoon/MVI_0051.MOV"),
        v("G/52. Good afternoon/MVI_0051.MOV"),  # same number again (counter cycle)
    ]
    runs = paths(class_runs(vids))
    assert ["MVI_0050_ (1).MOV"] in runs
    assert ["MVI_0050.MOV", "MVI_0051.MOV"] in runs
    assert ["MVI_0051.MOV"] in runs  # the repeated 0051 is its own run


def test_video_properties() -> None:
    x = v("Greetings/53. Good evening/MVI_005_4 (1).MOV")
    assert (x.category, x.class_folder, x.ext, x.number) == (
        "Greetings",
        "53. Good evening",
        ".MOV",
        5,
    )
    assert x.is_renamed_copy
    assert not v("A/1. b/MVI_0001.mp4").is_renamed_copy


def test_sample_selection_is_deterministic_and_well_formed() -> None:
    videos = load_manifest(MANIFEST)
    a, b = select(videos), select(videos)
    assert a == b
    assert len(a["extractor_bench"]) == 50 == len(set(a["extractor_bench"]))
    movs = [p for p in a["video_metadata"] if p.upper().endswith(".MOV")]
    assert len(movs) == 30
    assert len({s["run"] for s in a["session_spotcheck"]}) == 10
    known = {x.path for x in videos}
    assert all(s["path"] in known for s in a["session_spotcheck"])
