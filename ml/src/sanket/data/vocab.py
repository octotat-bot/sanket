"""Vocabulary file for the apps: label ids with human-readable names and categories.

    uv run python -m sanket.data.vocab        # writes spec/vocab_include50.json

Derived from the vendored official split files (class folder names, e.g. "51. Good Morning") and
the label map, so ids match training exactly. Used by the recording tool now, and by the app later.
"""

import json
import re
from typing import Any

from sanket.data.splits import SPLITS, Dataset, class_folder, label_of, load_label_map, load_split
from sanket.paths import REPO_ROOT


def build(dataset: Dataset) -> list[dict[str, Any]]:
    label_map = load_label_map(dataset)
    info: dict[str, tuple[str, str]] = {}
    for split in SPLITS:
        for path in load_split(dataset, split):
            name = re.sub(r"^\d+\.\s*", "", class_folder(path)).strip()
            info.setdefault(label_of(path), (name[:1].upper() + name[1:], path.split("/")[0]))
    return [
        {"id": i, "label": lab, "display": info[lab][0], "category": info[lab][1].replace("_", " ")}
        for lab, i in sorted(label_map.items(), key=lambda kv: kv[1])
    ]


def main() -> None:
    out = REPO_ROOT / "spec" / "vocab_include50.json"
    out.write_text(
        json.dumps({"dataset": "include50", "words": build("include50")}, indent=1) + "\n"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
