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
