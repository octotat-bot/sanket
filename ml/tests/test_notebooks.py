"""The Kaggle notebook stays thin, output-free and pointed at real configs."""

import json
import re

from sanket.paths import REPO_ROOT

NB = REPO_ROOT / "ml" / "notebooks" / "kaggle_include50_gru.ipynb"


def cells() -> list[dict]:  # type: ignore[type-arg]
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4
    return nb["cells"]  # type: ignore[no-any-return]


def test_notebook_has_no_outputs() -> None:
    for c in cells():
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_notebook_calls_src_and_uses_an_existing_config() -> None:
    src = "".join("".join(c["source"]) for c in cells())
    assert "sanket.train" in src and "sanket.evaluate" in src
    assert "--group train" in src and "uv sync --locked" in src
    config = re.search(r'CONFIG = "([^"]+)"', src)
    assert config and (REPO_ROOT / "ml" / config.group(1)).exists()
    assert "GIT_SHA" in src  # every run traces to a commit


def test_notebook_stays_thin() -> None:
    for c in cells():
        if c["cell_type"] == "code":
            body = [ln for ln in c["source"] if ln.strip() and not ln.lstrip().startswith("#")]
            assert len(body) <= 15, "move logic into ml/src/sanket instead"
            assert not any(re.match(r"\s*(def|class) ", ln) for ln in body)
