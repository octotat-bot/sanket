"""ONNX export + parity on a tiny synthetic-trained model (training env: torch + onnxruntime)."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sanket.data.synthetic import synthetic_dataset  # noqa: E402
from sanket.evaluate import load_best  # noqa: E402
from sanket.export.onnx_export import export_onnx, labels_sha256, parity, quantize_int8  # noqa: E402
from sanket.keypoints.features import FeatureSpec  # noqa: E402
from sanket.train import TrainConfig, featurize, train  # noqa: E402


@pytest.fixture(autouse=True)
def few_threads() -> None:
    torch.set_num_threads(2)


@pytest.mark.parametrize(
    "model",
    [
        {"arch": "gru", "hidden": 32, "layers": 1, "dropout": 0.1},
        {"arch": "transformer", "d_model": 32, "heads": 2, "layers": 2, "dropout": 0.1},
    ],
)
def test_export_matches_torch_and_int8_agrees(tmp_path: Path, model: dict[str, object]) -> None:
    clips, labels = synthetic_dataset(4, 16, 0)
    items = [((lambda c=c: c), y) for c, y in zip(clips, labels, strict=True)]
    cfg = TrainConfig(
        seed=0, epochs=6, batch_size=16, lr=3e-3, warmup_epochs=1, T=16, model=model, device="cpu"
    )
    train(cfg, items, items[:16], 4, tmp_path / "run", log=lambda _: None)
    m, cfg2, _ = load_best(tmp_path / "run", 4, device="cpu")
    fs = FeatureSpec.from_spec()
    x = np.stack([featurize(c, fs, cfg2, None) for c in clips]).astype(np.float32)
    fp32, int8 = tmp_path / "m.fp32.onnx", tmp_path / "m.int8.onnx"
    export_onnx(m, cfg2.T, x.shape[-1], fp32)
    quantize_int8(fp32, int8)
    r = parity(m, fp32, int8, x, np.array(labels))
    assert r["torch_vs_fp32_max_abs_logit_diff"] < 1e-4
    assert r["torch_vs_fp32_top1_agreement"] == 1.0
    assert r["fp32_vs_int8_top1_agreement"] >= 0.95  # tiny synthetic model; the real gate is ≥ 99%
    assert r["size_bytes"]["int8"] < r["size_bytes"]["fp32"]
    json.dumps(r)  # report is JSON-serialisable


def test_labels_hash_is_order_sensitive() -> None:
    assert labels_sha256(["a", "b"]) != labels_sha256(["b", "a"])
    # The web app hashes JSON.stringify(labels) (no spaces): same bytes as this.
    assert (
        labels_sha256(["hello", "dog"])
        == __import__("hashlib").sha256(b'["hello","dog"]').hexdigest()
    )


def test_temperature_fit_recovers_a_known_temperature() -> None:
    from sanket.export.onnx_export import fit_temperature

    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 10)) * 2
    p = np.exp(true_logits) / np.exp(true_logits).sum(1, keepdims=True)
    y = np.array([rng.choice(10, p=row) for row in p])
    # A model that is 3x overconfident: its logits are 3x the calibrated ones.
    assert abs(fit_temperature(true_logits * 3, y) - 3) < 0.25
    assert abs(fit_temperature(true_logits, y) - 1) < 0.1


def test_calibration_refuses_a_degenerate_fit_on_a_perfect_val_set() -> None:
    from sanket.export.onnx_export import calibrate

    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 200)
    logits = rng.normal(size=(200, 5))
    logits[np.arange(200), y] += 10  # every val prediction correct
    assert (logits.argmax(1) == y).all()
    c = calibrate(logits, y)
    assert c["temperature"] == 1.0 and not c["fitted"] and c["val_errors"] == 0
