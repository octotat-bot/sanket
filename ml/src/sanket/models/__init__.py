"""Model registry: the config's `model.arch` picks the class ("gru" by default)."""

from typing import Any

from torch import nn


def build_model(n_features: int, n_classes: int, T: int, model_cfg: dict[str, Any]) -> nn.Module:
    from sanket.models.gru_baseline import GRUClassifier
    from sanket.models.transformer import TransformerClassifier

    kw = {k: v for k, v in model_cfg.items() if k != "arch"}
    arch = model_cfg.get("arch", "gru")
    if arch == "gru":
        return GRUClassifier(n_features, n_classes, **kw)
    if arch == "transformer":
        return TransformerClassifier(n_features, n_classes, T=T, **kw)
    raise ValueError(f"unknown model.arch {arch!r}")
