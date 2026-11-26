"""Phase 3 main model: a small Transformer encoder over frame tokens (fixed T, ADR 008 features).

input (B, T, F) → LayerNorm → Linear(F, d) → prepend a learned CLS token → + learned positions
      → N pre-norm encoder layers (multi-head self-attention + GELU feed-forward)
      → LayerNorm(CLS) → dropout → Linear(d, classes)

Pre-norm (norm_first) trains more stably at this small size and data scale. The nested-tensor fast
path is turned off so the eval graph is plain ops, which keeps the ONNX export simple (Phase 4).
No padding mask is needed: every input is resampled to exactly T frames.
"""

import torch
from torch import nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        T: int = 64,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.T = T
        self.embed = nn.Sequential(nn.LayerNorm(n_features), nn.Linear(n_features, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, T + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.T:
            raise ValueError(f"expected T={self.T} frames, got {x.shape[1]}")
        h = self.embed(x)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1) + self.pos
        h = self.encoder(self.drop(h))
        logits: torch.Tensor = self.head(h[:, 0])
        return logits
