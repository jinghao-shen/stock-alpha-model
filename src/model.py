"""
model.py — model architectures and loss functions.

MLPRegressor: Qwen embedding + numeric price features → predicted alpha.
LSTMRegressor: 20-day price sequence → predicted alpha (baseline).
combined_loss: MSE + λ·(1 - batch_IC), rewards correct cross-sectional ranking.
"""

import torch
import torch.nn as nn


# ── Loss ──────────────────────────────────────────────────────────

def ic_penalty(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    1 - Pearson correlation within the batch.
    Minimizing this directly rewards correct ranking, which is what
    the downstream long/short strategy needs — not just low MSE.
    """
    pred   = pred   - pred.mean()
    target = target - target.mean()
    num = (pred * target).sum()
    den = torch.sqrt((pred ** 2).sum() * (target ** 2).sum() + eps)
    return 1.0 - num / den


def combined_loss(pred: torch.Tensor, target: torch.Tensor, lam: float = 0.5) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target) + lam * ic_penalty(pred, target)


# ── Main model ────────────────────────────────────────────────────

class MLPRegressor(nn.Module):
    """
    Embedding → bottleneck projection → concat with numeric features → MLP head.

    The bottleneck is necessary: the Qwen embedding is 1024-dim while
    numeric features are only 5-dim. Without projection, the numeric
    features get drowned out in the subsequent linear layers.
    """

    def __init__(self, emb_dim: int, num_dim: int,
                 emb_proj: int = 128, hidden: tuple = (64,), dropout: float = 0.3):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_proj),
            nn.LayerNorm(emb_proj),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layers, prev = [], emb_proj + num_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = x[:, :self.emb_dim]
        num = x[:, self.emb_dim:]
        z   = torch.cat([self.emb_proj(emb), num], dim=1)
        return self.head(z).squeeze(-1)


# ── LSTM baseline ─────────────────────────────────────────────────

class LSTMRegressor(nn.Module):
    """
    Two-layer LSTM on 20-day [ret_1d, vol_ratio] sequences.
    Used as a stronger price-only baseline than Ridge.
    """

    def __init__(self, n_feat: int = 2, hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(self.drop(h[-1])).squeeze(-1)
