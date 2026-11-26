from sanket.checks.sample import MANIFEST
from sanket.data.reference_clips import select
from sanket.data.sessions import load_manifest
from sanket.data.splits import label_of, load_label_map, load_split, video_key


def test_one_train_clip_per_include50_word_never_val_or_test() -> None:
    chosen = select(load_manifest(MANIFEST))
    assert set(chosen) == set(load_label_map("include50"))
    train = {video_key(p) for p in load_split("include50", "train")}
    held_out = {video_key(p) for s in ("val", "test") for p in load_split("include50", s)}
    for label, v in chosen.items():
        assert video_key(v.path) in train and video_key(v.path) not in held_out
        assert label_of(v.path) == label and not v.is_renamed_copy
    assert select(load_manifest(MANIFEST)) == chosen  # deterministic
