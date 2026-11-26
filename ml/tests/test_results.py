import json
from pathlib import Path

import yaml

from sanket.results import BEGIN, END, bootstrap_ci, load_runs, table


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible() -> None:
    correct = [1] * 80 + [0] * 20
    lo, hi = bootstrap_ci(correct)
    assert lo < 0.8 < hi
    assert lo > 0.7 and hi < 0.9
    assert bootstrap_ci(correct) == (lo, hi)


def test_table_is_built_only_from_result_files(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "gru_x"
    run.mkdir(parents=True)
    ev = {"n": 4, "top1": 0.75, "top5": 1.0, "correct": [1, 1, 1, 0]}
    (run / "eval_test.json").write_text(json.dumps(ev))
    (run / "eval_val.json").write_text(json.dumps({**ev, "top1": 0.5}))
    (run / "config.yaml").write_text(yaml.safe_dump({"train": {"model": {"arch": "gru"}}}))
    (run / "git_sha.txt").write_text("abc1234def\n")
    (run / "environment.json").write_text(json.dumps({"hardware": "Apple M3 Pro CPU"}))
    runs = load_runs(tmp_path / "runs")
    md = table(runs, {})
    assert "75.0%" in md and "50.0%" in md and "n = 4" in md and "94.5%" in md
    assert "Apple M3 Pro CPU" in md
    assert BEGIN not in md and END not in md
