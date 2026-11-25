"""Training loop (Phase 2 baseline; reused for the Phase 3 Transformer).

    python -m sanket.train --config configs/gru_include50.yaml [--run-dir runs/NAME]

* AdamW, linear warmup then cosine decay (per step), label smoothing.
* Early stopping on the **validation split carved from train** (the authors' val files; never
  the test split, ADR 003). The test split is only touched by the evaluation script.
* A checkpoint every epoch (`last.pt`) plus `best.pt`. Kaggle sessions end at 12 h, so `--run-dir`
  of an existing run resumes from `last.pt`.
* Reproducible: every random draw is seeded from (seed, epoch, sample index), so a resumed run
  produces the same weights as an uninterrupted one (tested). Mixed precision only on CUDA.
* TensorBoard event files, `config.yaml`, `git_sha.txt` and `history.json` in the run directory.
"""

import argparse
import json
import math
import random
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sanket.augment.transforms import AugmentConfig, augment
from sanket.keypoints.features import FeatureSpec, clip_to_model_input
from sanket.models.gru_baseline import GRUClassifier, n_parameters
from sanket.paths import REPO_ROOT

Clip = dict[str, np.ndarray]
ClipSource = Callable[[], Clip]  # loads one clip lazily (an .npz on disk, or an in-memory clip)


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 3.0
    label_smoothing: float = 0.1
    patience: int = 10  # epochs without val-accuracy improvement before stopping
    T: int = 64
    use_velocity: bool = False
    augment: bool = True
    augment_cfg: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(
        default_factory=lambda: {"hidden": 128, "layers": 2, "dropout": 0.3}
    )
    device: str = (
        "auto"  # auto = cuda if available else cpu ("mps" only if asked: not deterministic)
    )
    amp: bool = True  # only used on CUDA

    @staticmethod
    def from_yaml(path: Path) -> "TrainConfig":
        d = yaml.safe_load(path.read_text())["train"]
        return TrainConfig(**d)


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup to base_lr, then cosine decay to 0 at total_steps."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def featurize(
    clip: Clip, fs: FeatureSpec, cfg: TrainConfig, rng: np.random.Generator | None
) -> np.ndarray:
    if rng is not None and cfg.augment:
        clip = augment(clip, rng, AugmentConfig(**cfg.augment_cfg))
    return clip_to_model_input(clip, fs, use_velocity=cfg.use_velocity, T=cfg.T)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct = 0.0, 0
    for i in range(0, len(x), batch):
        logits = model(x[i : i + batch])
        total_loss += float(loss_fn(logits, y[i : i + batch])) * len(logits)
        correct += int((logits.argmax(1) == y[i : i + batch]).sum())
    return {"loss": total_loss / max(len(x), 1), "acc": correct / max(len(x), 1)}


def train(
    cfg: TrainConfig,
    train_items: Sequence[tuple[ClipSource, int]],
    val_items: Sequence[tuple[ClipSource, int]],
    n_classes: int,
    run_dir: Path,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    fs = FeatureSpec.from_spec()

    # Validation features: computed once, never augmented.
    xv = torch.tensor(
        np.stack([featurize(src(), fs, cfg, None) for src, _ in val_items]), device=device
    )
    yv = torch.tensor([lab for _, lab in val_items], device=device)

    n_features = xv.shape[-1]
    model = GRUClassifier(n_features, n_classes, **cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = math.ceil(len(train_items) / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    state: dict[str, Any] = {"epoch": 0, "best_acc": -1.0, "bad_epochs": 0, "history": []}
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        state = ck["state"]
        log(f"resumed from {last} at epoch {state['epoch']}")
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": asdict(cfg)}, sort_keys=True))
        (run_dir / "git_sha.txt").write_text(_git_sha() + "\n")
    log(f"model: {n_parameters(model):,} parameters, {n_features} features, device {device}")

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    for epoch in range(int(state["epoch"]), cfg.epochs):
        model.train()
        torch.manual_seed(cfg.seed * 1_000_003 + epoch)  # dropout draws: exact on resume
        order = np.random.default_rng([cfg.seed, epoch]).permutation(len(train_items))
        total, seen = 0.0, 0
        for b in range(steps_per_epoch):
            idx = order[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            xb = torch.tensor(
                np.stack(
                    [
                        featurize(
                            train_items[i][0](),
                            fs,
                            cfg,
                            np.random.default_rng([cfg.seed, epoch, int(i)]),
                        )
                        for i in idx
                    ]
                ),
                device=device,
            )
            yb = torch.tensor([train_items[i][1] for i in idx], device=device)
            step = epoch * steps_per_epoch + b
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup_steps, cfg.lr)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(idx)
            seen += len(idx)

        val = evaluate(model, xv, yv, cfg.batch_size)
        row = {
            "epoch": epoch + 1,
            "train_loss": total / seen,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "lr": opt.param_groups[0]["lr"],
        }
        state["history"].append(row)
        writer.add_scalars("loss", {"train": row["train_loss"], "val": row["val_loss"]}, epoch + 1)
        writer.add_scalar("val/acc", row["val_acc"], epoch + 1)
        writer.add_scalar("lr", row["lr"], epoch + 1)
        improved = val["acc"] > state["best_acc"]
        if improved:
            state["best_acc"], state["bad_epochs"] = val["acc"], 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "val": val}, run_dir / "best.pt"
            )
        else:
            state["bad_epochs"] += 1
        state["epoch"] = epoch + 1
        tmp = run_dir / "last.pt.tmp"
        torch.save(
            {"model": model.state_dict(), "optimizer": opt.state_dict(), "state": state}, tmp
        )
        tmp.replace(last)  # atomic: a session killed mid-save never corrupts the checkpoint
        log(
            f"epoch {epoch + 1}: train {row['train_loss']:.4f}  "
            f"val {val['loss']:.4f}  acc {val['acc']:.3f}" + ("  *" if improved else "")
        )
        if state["bad_epochs"] >= cfg.patience:
            log(f"early stop: no val improvement for {cfg.patience} epochs")
            break
    writer.close()
    (run_dir / "history.json").write_text(json.dumps(state["history"], indent=2) + "\n")
    return {
        "best_val_acc": state["best_acc"],
        "epochs_run": state["epoch"],
        "history": state["history"],
    }


def main() -> None:
    from sanket.data.keypoint_index import load_split_index
    from sanket.data.splits import Split, load_label_map

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, help="existing run dir to resume, or new dir")
    args = ap.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    cfg = TrainConfig(**raw["train"])
    data = raw["data"]
    kp_dir = (REPO_ROOT / data["keypoints_dir"]).resolve()
    n_classes = len(load_label_map(data["dataset"]))

    def load_npz(path: Path) -> Clip:
        return dict(np.load(path))

    def items(split: Split) -> list[tuple[ClipSource, int]]:
        ix = load_split_index(data["dataset"], split, kp_dir)
        if ix.missing:
            print(
                f"{data['dataset']}/{split}: {len(ix.missing)} entries without keypoints "
                "(reported, skipped)"
            )
        return [(partial(load_npz, it.path), it.label_id) for it in ix.items]

    run_dir = args.run_dir or REPO_ROOT / "runs" / args.config.stem
    result = train(cfg, items("train"), items("val"), n_classes, run_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}))


if __name__ == "__main__":
    main()
