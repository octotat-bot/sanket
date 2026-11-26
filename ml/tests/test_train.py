"""GRU baseline + training loop on synthetic sign clips (needs the `train` group: torch).

Run in the training env:  UV_PROJECT_ENVIRONMENT=.venv-train uv run pytest tests/test_train.py
Skipped automatically where torch isn't installed (the extraction env, the plain CI job).
"""

import itertools
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sanket.data.synthetic import synthetic_dataset  # noqa: E402
from sanket.models.gru_baseline import GRUClassifier, n_parameters  # noqa: E402
from sanket.train import TrainConfig, lr_at, train  # noqa: E402

N_CLASSES = 4


@pytest.fixture(autouse=True)
def few_threads() -> None:
    torch.set_num_threads(2)  # keep CPU light (the extraction may be running)


def items(seed: int, per_class: int) -> list:  # type: ignore[type-arg]
    clips, labels = synthetic_dataset(N_CLASSES, per_class, seed)
    return [((lambda c=c: c), y) for c, y in zip(clips, labels, strict=True)]


def tiny(**kw: object) -> TrainConfig:
    base: dict[str, object] = dict(
        seed=0,
        epochs=12,
        batch_size=16,
        lr=3e-3,
        warmup_epochs=1,
        patience=50,
        T=16,
        model={"hidden": 32, "layers": 1, "dropout": 0.1},
        device="cpu",
        augment=True,
    )
    base.update(kw)
    return TrainConfig(**base)  # type: ignore[arg-type]


def test_model_shapes_and_size() -> None:
    m = GRUClassifier(193, 50)
    assert m(torch.zeros(3, 64, 193)).shape == (3, 50)
    assert n_parameters(m) < 5_000_000  # browser budget (CLAUDE.md)


def test_lr_schedule_warmup_then_cosine() -> None:
    lrs = [lr_at(s, 100, 10, 1.0) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(a >= b for a, b in itertools.pairwise(lrs[10:]))  # non-increasing after warmup
    assert lrs[-1] < 0.01


def test_learns_synthetic_signs(tmp_path: Path) -> None:
    r = train(tiny(), items(0, 24), items(1, 8), N_CLASSES, tmp_path / "run", log=lambda m: None)
    assert r["best_val_acc"] >= 0.9, r["history"]
    for f in ("best.pt", "last.pt", "config.yaml", "git_sha.txt", "history.json"):
        assert (tmp_path / "run" / f).exists(), f
    assert any((tmp_path / "run" / "tb").iterdir())  # TensorBoard events


def test_same_seed_same_result(tmp_path: Path) -> None:
    a = train(
        tiny(epochs=3), items(0, 8), items(1, 4), N_CLASSES, tmp_path / "a", log=lambda m: None
    )
    b = train(
        tiny(epochs=3), items(0, 8), items(1, 4), N_CLASSES, tmp_path / "b", log=lambda m: None
    )
    assert a["history"] == b["history"]


class SessionDied(Exception):
    pass


def test_resume_equals_uninterrupted(tmp_path: Path) -> None:
    cfg = tiny(epochs=4)
    train(cfg, items(0, 8), items(1, 4), N_CLASSES, tmp_path / "straight", log=lambda m: None)

    def die_after_epoch_2(msg: str) -> None:
        if msg.startswith("epoch 2:"):  # logged right after epoch 2's checkpoint was written
            raise SessionDied

    with pytest.raises(SessionDied):
        train(cfg, items(0, 8), items(1, 4), N_CLASSES, tmp_path / "resumed", log=die_after_epoch_2)
    logs: list[str] = []
    r = train(cfg, items(0, 8), items(1, 4), N_CLASSES, tmp_path / "resumed", log=logs.append)
    assert any("resumed from" in m and "epoch 2" in m for m in logs)
    assert [h["epoch"] for h in r["history"]] == [1, 2, 3, 4]

    a = torch.load(tmp_path / "straight" / "last.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)["model"]
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_early_stopping(tmp_path: Path) -> None:
    r = train(
        tiny(epochs=20, lr=0.0, patience=2),
        items(0, 4),
        items(1, 2),
        N_CLASSES,
        tmp_path / "e",
        log=lambda m: None,
    )
    assert r["epochs_run"] == 3  # epoch 1 sets the best; 2 and 3 don't improve → stop


def test_val_features_are_never_augmented(tmp_path: Path) -> None:
    # Same val set with augmentation on and off must give the same epoch-0 val metrics for an
    # untrained model (lr=0): only training samples are augmented.
    a = train(
        tiny(epochs=1, lr=0.0),
        items(0, 4),
        items(1, 4),
        N_CLASSES,
        tmp_path / "x",
        log=lambda m: None,
    )
    b = train(
        tiny(epochs=1, lr=0.0, augment=False),
        items(0, 4),
        items(1, 4),
        N_CLASSES,
        tmp_path / "y",
        log=lambda m: None,
    )
    assert a["history"][0]["val_acc"] == b["history"][0]["val_acc"]
    assert np.isclose(a["history"][0]["val_loss"], b["history"][0]["val_loss"])


def test_topk_metrics_by_hand() -> None:
    from sanket.evaluate import topk_metrics

    logits = np.array([[0.9, 0.1, 0.0], [0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    m = topk_metrics(logits, np.array([0, 1, 1]), 3)
    assert m["top1"] == pytest.approx(1 / 3) and m["top5"] == 1.0  # top-5 capped at n_classes
    assert m["per_class"][1] == {"n": 2, "acc": 0.0}


def test_evaluate_run_uses_best_checkpoint_and_never_augments(tmp_path: Path) -> None:
    from sanket.evaluate import evaluate_run

    r = train(tiny(), items(0, 24), items(1, 8), N_CLASSES, tmp_path / "run", log=lambda m: None)
    test_items = items(2, 8)
    a = evaluate_run(tmp_path / "run", test_items, N_CLASSES)
    b = evaluate_run(tmp_path / "run", test_items, N_CLASSES)
    assert a["top1"] == b["top1"]  # deterministic: no augmentation at evaluation
    assert a["n"] == 32 and a["top1"] >= 0.85 and a["top5"] >= a["top1"]
    assert a["best_val"]["acc"] == pytest.approx(r["best_val_acc"])


def test_transformer_shapes_size_and_fixed_T() -> None:
    from sanket.models import build_model

    m = build_model(193, 50, 64, {"arch": "transformer"})
    assert m(torch.zeros(3, 64, 193)).shape == (3, 50)
    assert n_parameters(m) < 5_000_000  # browser budget (CLAUDE.md)
    with pytest.raises(ValueError, match="T=64"):
        m(torch.zeros(1, 32, 193))
    with pytest.raises(ValueError, match="arch"):
        build_model(193, 50, 64, {"arch": "lstm"})


def test_transformer_learns_synthetic_signs(tmp_path: Path) -> None:
    cfg = tiny(
        epochs=30,
        lr=1e-3,
        model={"arch": "transformer", "d_model": 32, "heads": 2, "layers": 2, "dropout": 0.1},
    )
    out = train(cfg, items(0, 24), items(1, 8), N_CLASSES, tmp_path / "run", log=lambda _: None)
    assert out["best_val_acc"] >= 0.9, out
