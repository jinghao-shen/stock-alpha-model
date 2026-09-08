"""
train.py — training loop and ensemble inference.

Early stopping is on val IC (Pearson correlation), not val loss,
because IC directly measures cross-sectional ranking quality.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model import combined_loss


# ── Dataset ───────────────────────────────────────────────────────

class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ── Metric ────────────────────────────────────────────────────────

def pearson_ic(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum()) + eps
    return float((a * b).sum() / denom)


# ── Single-seed training ──────────────────────────────────────────

def train_one(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    ic_lambda: float,
    device: str,
    wandb_run=None,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_ic, best_state, patience_count, best_epoch = -float("inf"), None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = combined_loss(model(xb), yb, lam=ic_lambda)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(device)).cpu().numpy())
                tgts.append(yb.numpy())
        val_ic = pearson_ic(np.concatenate(preds), np.concatenate(tgts))

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "val/ic": val_ic})

        if val_ic > best_ic + 1e-6:
            best_ic, best_epoch = val_ic, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            break

    model.load_state_dict(best_state)
    print(f"  seed {seed}: best val IC {best_ic:+.4f} @ epoch {best_epoch} "
          f"(stopped at {epoch})")
    return model


# ── Ensemble inference ────────────────────────────────────────────

@torch.no_grad()
def ensemble_predict(models: list[nn.Module], X: np.ndarray, device: str) -> np.ndarray:
    xb = torch.from_numpy(X).to(device)
    preds = []
    for m in models:
        m.eval()
        preds.append(m(xb).cpu().numpy())
    return np.mean(preds, axis=0)
