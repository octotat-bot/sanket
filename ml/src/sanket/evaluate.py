"""Evaluate a trained run on a split (Phase 2 baseline number; Phase 3 adds CIs, confusion matrix).

    python -m sanket.evaluate --config configs/gru_include50.yaml \
        --run-dir runs/gru_include50 --split test

Loads `best.pt` (the epoch with the best *validation* accuracy; the test split never influenced
training or model selection, ADR 003), and writes `eval_<split>.json` into the run dir:
top-1, top-5, per-class accuracy, counts, and the split entries that had no keypoints (listed,
never silently dropped).
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from sanket.keypoints.features import FeatureSpec
from sanket.models import build_model
from sanket.paths import REPO_ROOT
from sanket.train import ClipSource, TrainConfig, featurize, pick_device


def topk_metrics(logits: np.ndarray, labels: np.ndarray, n_classes: int) -> dict[str, Any]:
    order = np.argsort(-logits, axis=1)
    top1 = order[:, 0] == labels
    top5 = (order[:, : min(5, n_classes)] == labels[:, None]).any(axis=1)
    per_class = {
        int(c): {"n": int((labels == c).sum()), "acc": float(top1[labels == c].mean())}
        for c in range(n_classes)
        if (labels == c).any()
    }
    return {
        "n": len(labels),
        "top1": float(top1.mean()),
        "top5": float(top5.mean()),
        "per_class": per_class,
    }


def load_best(
    run_dir: Path, n_classes: int, device: str | None = None
) -> tuple[torch.nn.Module, TrainConfig, dict[str, Any]]:
    """The run's best-validation checkpoint as an eval-mode model, its config and the checkpoint."""
    cfg = TrainConfig(**yaml.safe_load((run_dir / "config.yaml").read_text())["train"])
    dev = pick_device(device or cfg.device)
    fs = FeatureSpec.from_spec()
    n_features = fs.n_features + (fs.n_coord if cfg.use_velocity else 0)
    model = build_model(n_features, n_classes, cfg.T, cfg.model).to(dev)
    ck = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


@torch.no_grad()
def evaluate_run(
    run_dir: Path, items: Sequence[tuple[ClipSource, int]], n_classes: int, batch: int = 64
) -> dict[str, Any]:
    model, cfg, ck = load_best(run_dir, n_classes)
    device = next(model.parameters()).device
    fs = FeatureSpec.from_spec()
    x = np.stack([featurize(src(), fs, cfg, None) for src, _ in items])  # never augmented
    logits = np.concatenate(
        [
            model(torch.tensor(x[i : i + batch], device=device)).cpu().numpy()
            for i in range(0, len(x), batch)
        ]
    )
    out = topk_metrics(logits, np.array([lab for _, lab in items]), n_classes)
    out["best_epoch"] = int(ck["epoch"])
    out["best_val"] = ck["val"]
    return out


def main() -> None:
    from functools import partial

    from sanket.data.keypoint_index import load_split_index
    from sanket.data.splits import Split, load_label_map

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()
    data = yaml.safe_load(args.config.read_text())["data"]
    kp_dir = (REPO_ROOT / data["keypoints_dir"]).resolve()
    label_map = load_label_map(data["dataset"])
    split: Split = args.split
    ix = load_split_index(data["dataset"], split, kp_dir)

    def load_npz(path: Path) -> dict[str, np.ndarray]:
        return dict(np.load(path))

    items = [(partial(load_npz, it.path), it.label_id) for it in ix.items]
    result = evaluate_run(args.run_dir, items, len(label_map))
    result["dataset"], result["split"], result["missing"] = data["dataset"], split, ix.missing
    names = {v: k for k, v in label_map.items()}
    result["per_class"] = {names[c]: v for c, v in result["per_class"].items()}
    (args.run_dir / f"eval_{split}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: result[k] for k in ("dataset", "split", "n", "top1", "top5", "best_epoch")})
    )
    if ix.missing:
        print(f"{len(ix.missing)} {split} entries had no keypoints (listed in eval_{split}.json)")


if __name__ == "__main__":
    main()
