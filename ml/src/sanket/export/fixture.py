"""A tiny UNTRAINED model with the real INCLUDE-50 labels and input contract, for browser e2e
tests of the Live page (web/tests/e2e/live.spec.ts). Its predictions are meaningless; it only
proves the pipeline loads and runs: worker, onnxruntime-web runtime files under COEP, labels-hash
check, feature shapes. Never served by the app.

    UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync python -m sanket.export.fixture
"""

import hashlib
import json

import torch

from sanket.data.splits import load_label_map
from sanket.export.onnx_export import export_onnx, labels_sha256, quantize_int8
from sanket.keypoints.features import FeatureSpec
from sanket.models import build_model
from sanket.paths import REPO_ROOT
from sanket.spec import load_spec

OUT = REPO_ROOT / "web" / "tests" / "fixtures" / "model"


def main() -> None:
    torch.manual_seed(0)
    labels = [k for k, _ in sorted(load_label_map("include50").items(), key=lambda kv: kv[1])]
    fs = FeatureSpec.from_spec()
    T = fs.T
    tiny = {"arch": "transformer", "d_model": 16, "heads": 2, "layers": 1}
    model = build_model(fs.n_features, len(labels), T, tiny)
    fp32, int8 = OUT / "include50.fp32.onnx", OUT / "include50.int8.onnx"
    export_onnx(model, T, fs.n_features, fp32)
    quantize_int8(fp32, int8)
    meta = {
        "labels": labels,
        "labels_sha256": labels_sha256(labels),
        "dataset": "include50",
        "input": {"name": "x", "shape": ["batch", T, fs.n_features], "dtype": "float32"},
        "output": {"name": "logits", "shape": ["batch", len(labels)]},
        "T": T,
        "use_velocity": False,
        "spec_version": load_spec()["spec_version"],
        "arch": "untrained-test-fixture",
        "files": {"fp32": fp32.name, "int8": int8.name},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (fp32, int8)},
    }
    (OUT / "include50.labels.json").write_text(json.dumps(meta, indent=1) + "\n")
    print({p.name: p.stat().st_size for p in OUT.iterdir()})


if __name__ == "__main__":
    main()
