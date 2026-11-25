import json

from sanket.data.splits import load_label_map
from sanket.data.vocab import build
from sanket.paths import REPO_ROOT


def test_committed_vocab_matches_label_map_and_generator() -> None:
    committed = json.loads((REPO_ROOT / "spec" / "vocab_include50.json").read_text())["words"]
    assert committed == build("include50")
    assert {w["label"]: w["id"] for w in committed} == load_label_map("include50")
    assert all(w["display"] and w["category"] for w in committed)
