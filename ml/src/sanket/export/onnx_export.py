"""Phase 4: export a trained run to ONNX (fp32 + dynamic int8) and check parity.

    python -m sanket.export.onnx_export --config configs/gru_include50.yaml \
        --run-dir runs/gru_include50 --name include50

Writes into `web/public/models/`:
  <name>.fp32.onnx, <name>.int8.onnx   input "x" (batch, T, F) float32 → output "logits" (batch, C)
  <name>.labels.json                    labels by id, their sha256, and the model's input contract

and `reports/export_<name>.json` with the parity numbers, measured on the official test split:
  1. PyTorch vs ONNX fp32: max |logit difference|.
  2. ONNX fp32 vs int8: top-1 agreement (target ≥ 99%), accuracy of each, per-class drop.

Export uses the TorchScript-based exporter (`dynamo=False`): it needs no extra dependency
(onnxscript) and both models export as plain ops (opset 17). Batch is dynamic; T is fixed by the
config (inputs are always resampled to T frames, ADR 008).
"""

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic

from sanket.paths import REPO_ROOT

OPSET = 17
WEB_MODELS = REPO_ROOT / "web" / "public" / "models"


def labels_sha256(labels: Sequence[str]) -> str:
    """Hash of the labels in id order, as compact JSON. The web app recomputes it before loading."""
    return hashlib.sha256(json.dumps(list(labels), separators=(",", ":")).encode()).hexdigest()


def export_onnx(model: torch.nn.Module, T: int, n_features: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, T, n_features),),
        str(path),
        dynamo=False,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
    )


def quantize_int8(fp32: Path, int8: Path) -> None:
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature minimising the negative log-likelihood (grid over 0.25..8, log-spaced, then a
    finer grid around the best): deterministic, no optimiser state. Fit on val, never test."""

    def nll(t: float) -> float:
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(labels)), labels].mean())

    grid = np.exp(np.linspace(np.log(GRID[0]), np.log(GRID[1]), 61))
    best = float(min(grid, key=nll))
    fine = np.linspace(best / 1.06, best * 1.06, 41)
    return round(float(min(fine, key=nll)), 4)


GRID = (0.25, 8.0)


def calibrate(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Fitted temperature, or 1.0 with the reason when a fit would be meaningless: with no val
    errors the likelihood keeps improving as T → 0, so the optimum sits at the grid's lower edge
    and would make every live prediction look certain (rejection would never fire)."""
    errors = int((logits.argmax(1) != labels).sum())
    t = fit_temperature(logits, labels)
    if errors == 0 or not (GRID[0] * 1.06 < t < GRID[1] / 1.06):
        return {
            "temperature": 1.0,
            "fitted": False,
            "reason": f"degenerate fit (val errors: {errors}, optimum {t} at the grid edge)",
            "val_errors": errors,
        }
    return {"temperature": t, "fitted": True, "val_errors": errors}


def run_onnx(path: Path, x: np.ndarray, batch: int = 64) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate(
        [sess.run(["logits"], {"x": x[i : i + batch]})[0] for i in range(0, len(x), batch)]
    )


@torch.no_grad()
def parity(
    model: torch.nn.Module, fp32: Path, int8: Path, x: np.ndarray, y: np.ndarray
) -> dict[str, Any]:
    model = model.cpu().eval()
    ref = np.concatenate([model(torch.tensor(x[i : i + 64])).numpy() for i in range(0, len(x), 64)])
    a, b = run_onnx(fp32, x), run_onnx(int8, x)
    pa, pb = a.argmax(1), b.argmax(1)
    per_class = {}
    for c in np.unique(y):
        m = y == c
        per_class[int(c)] = {
            "n": int(m.sum()),
            "fp32_acc": float((pa[m] == c).mean()),
            "int8_acc": float((pb[m] == c).mean()),
        }
    return {
        "n": len(x),
        "torch_vs_fp32_max_abs_logit_diff": float(np.abs(ref - a).max()),
        "torch_vs_fp32_top1_agreement": float((ref.argmax(1) == pa).mean()),
        "fp32_vs_int8_top1_agreement": float((pa == pb).mean()),
        "torch_acc": float((ref.argmax(1) == y).mean()),
        "fp32_acc": float((pa == y).mean()),
        "int8_acc": float((pb == y).mean()),
        "per_class": per_class,
        "size_bytes": {"fp32": os.path.getsize(fp32), "int8": os.path.getsize(int8)},
    }


def main() -> None:
    from sanket.data.keypoint_index import load_split_index
    from sanket.data.splits import load_label_map
    from sanket.evaluate import load_best
    from sanket.keypoints.features import FeatureSpec
    from sanket.spec import load_spec
    from sanket.train import featurize

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--name", default="include50")
    ap.add_argument("--out", type=Path, default=WEB_MODELS)
    args = ap.parse_args()

    data = yaml.safe_load(args.config.read_text())["data"]
    label_map = load_label_map(data["dataset"])
    labels = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    model, cfg, ck = load_best(args.run_dir, len(labels), device="cpu")
    fs = FeatureSpec.from_spec()
    n_features = fs.n_features + (fs.n_coord if cfg.use_velocity else 0)

    fp32 = args.out / f"{args.name}.fp32.onnx"
    int8 = args.out / f"{args.name}.int8.onnx"
    export_onnx(model, cfg.T, n_features, fp32)
    quantize_int8(fp32, int8)

    kp_dir = (REPO_ROOT / data["keypoints_dir"]).resolve()
    ix = load_split_index(data["dataset"], "test", kp_dir)
    x = np.stack([featurize(dict(np.load(it.path)), fs, cfg, None) for it in ix.items])
    y = np.array([it.label_id for it in ix.items])
    report = parity(model, fp32, int8, x.astype(np.float32), y)
    report["per_class"] = {labels[c]: v for c, v in report["per_class"].items()}

    # Calibration on val (ADR 009): temperature for the browser's softmax; thresholds as set there.
    vix = load_split_index(data["dataset"], "val", kp_dir)
    xv = np.stack([featurize(dict(np.load(it.path)), fs, cfg, None) for it in vix.items])
    yv = np.array([it.label_id for it in vix.items])
    with torch.no_grad():
        val_logits = model(torch.tensor(xv.astype(np.float32))).numpy()
    cal = calibrate(val_logits, yv)
    temperature = cal["temperature"]
    report["calibration"] = {**cal, "fit_on": "val", "n": len(yv)}

    git_sha = (args.run_dir / "git_sha.txt").read_text().strip()
    spec = load_spec()
    meta = {
        "labels": labels,
        "labels_sha256": labels_sha256(labels),
        "dataset": data["dataset"],
        "input": {"name": "x", "shape": ["batch", cfg.T, n_features], "dtype": "float32"},
        "output": {"name": "logits", "shape": ["batch", len(labels)]},
        "T": cfg.T,
        "use_velocity": cfg.use_velocity,
        "spec_version": spec["spec_version"],
        "arch": cfg.model.get("arch", "gru"),
        "train_git_sha": git_sha,
        "best_epoch": int(ck["epoch"]),
        "files": {"fp32": fp32.name, "int8": int8.name},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (fp32, int8)},
        "calibration": {"temperature": temperature, "min_prob": 0.45, "min_margin": 0.15},
    }
    (args.out / f"{args.name}.labels.json").write_text(json.dumps(meta, indent=1) + "\n")
    report.update(
        name=args.name, run_dir=args.run_dir.name, train_git_sha=git_sha, missing=ix.missing
    )
    out = REPO_ROOT / "reports" / f"export_{args.name}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=1))


if __name__ == "__main__":
    main()
