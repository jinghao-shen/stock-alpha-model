"""
evaluate.py — prediction quality metrics.

IC (Pearson) over cross-sectional alpha is the primary signal metric.
MSE/MAE are reported for completeness but not used for model selection.
"""

import numpy as np


def compute_metrics(preds: np.ndarray, y_alpha: np.ndarray,
                    y_raw: np.ndarray) -> dict:
    mse     = float(np.mean((preds - y_alpha) ** 2))
    mae     = float(np.mean(np.abs(preds - y_alpha)))
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y_alpha)))
    ic_a    = _pearson(preds, y_alpha)
    ic_r    = _pearson(preds, y_raw)
    return dict(n=len(preds), mse=mse, mae=mae,
                dir_acc=dir_acc, ic_alpha=ic_a, ic_raw=ic_r)


def print_metrics(split: str, metrics: dict) -> None:
    print(
        f"{split:<6} | n={metrics['n']:>5} | "
        f"MSE={metrics['mse']:.5f} | MAE={metrics['mae']:.5f} | "
        f"DirAcc={metrics['dir_acc']:.3f} | "
        f"IC(alpha)={metrics['ic_alpha']:+.3f} | IC(raw)={metrics['ic_raw']:+.3f}"
    )


def _pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    if len(a) < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum()) + eps
    return float((a * b).sum() / denom)
