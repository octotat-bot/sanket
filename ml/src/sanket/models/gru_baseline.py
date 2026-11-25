"""Phase 2 baseline: a bidirectional GRU over per-frame features (ADR 008 layout, fixed T).

input (B, T, F) → LayerNorm → Linear(F, hidden) → GELU → dropout
      → GRU(hidden, hidden, layers, bidirectional) → mean over time → LayerNorm → dropout → Linear
Mean pooling (not the last state) because the input is resampled to a fixed T, and the whole sign
matters, not just its end.
"""

import torch
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.LayerNorm(n_features), nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(out), nn.Dropout(dropout), nn.Linear(out, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.embed(x))
        logits: torch.Tensor = self.head(h.mean(dim=1))
        return logits


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
