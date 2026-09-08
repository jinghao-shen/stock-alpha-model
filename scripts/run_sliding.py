"""
run_sliding.py — walk-forward generalization test (12 × 59-day windows, 2018-2019).

Train/val mirrors run_17h2.py (2017-06-14 → 2017-10-18 / 2017-10-19 → 2017-10-31).
The trained ensemble is then frozen and evaluated on 12 consecutive non-overlapping
59-day windows across 2018-01-01 → 2019-12-31 without any retraining.

This is a stress test of temporal generalization — the model never sees 2018-2019
data during training.

Usage:
    python scripts/run_sliding.py --data-dir data/ --cache-dir cache/ --out-dir outputs/
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

import torch

from src.data import (load_price, load_news, add_price_features,
                      build_samples, add_alpha_target, time_split, NUMERIC_FEATURES)
from src.prompts import build_prompts
from src.embed import get_embeddings, align_embeddings
from src.model import MLPRegressor
from src.train import ArrayDataset, train_one, ensemble_predict
from src.evaluate import compute_metrics, print_metrics
from src.backtest import run_backtest, summarize

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

# Train/val boundaries match the reference 17H2 notebook exactly
TRAIN_END = "2017-10-18"
VAL_END   = "2017-10-31"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  default="data/")
    p.add_argument("--cache-dir", default="cache/")
    p.add_argument("--out-dir",   default="outputs/")
    p.add_argument("--config",    default="configs/base.yaml")
    return p.parse_args()


def build_windows(wf_start: str, wf_end: str, window_days: int):
    """Non-overlapping fixed-width calendar-day windows."""
    ws = pd.Timestamp(wf_start)
    we_limit = pd.Timestamp(wf_end)
    windows = []
    while True:
        we = ws + pd.Timedelta(days=window_days - 1)
        if we > we_limit:
            break
        windows.append((ws, we))
        ws = we + pd.Timedelta(days=1)
    return windows


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_cfg = cfg["training"]
    bt_cfg = cfg["backtest"]
    sl_cfg = cfg["sliding"]
    SEED   = tr_cfg["seed"]
    np.random.seed(SEED); torch.manual_seed(SEED)

    # ── 1. Load full 2017-2019 dataset ────────────────────────────
    print("Loading data (2017-2019)...")
    price_17 = load_price(data_dir / "yahoo_2017_12_stock_data.csv")
    news_17  = load_news(data_dir  / "financials_jun-dec2017_news.csv")
    price_18 = load_price(data_dir / "stock_prices_2018.csv")
    news_18  = load_news(data_dir  / "news_filtered_2018(1).csv")
    price_19 = load_price(data_dir / "stock_prices_2019.csv")
    news_19  = load_news(data_dir  / "news_filtered_2019(1).csv")

    price = pd.concat([price_17, price_18, price_19], ignore_index=True)
    news  = pd.concat([news_17,  news_18,  news_19 ], ignore_index=True)

    # Drop duplicate (stock, date) rows that arise from overlapping files
    price = price.drop_duplicates(subset=["stock", "date"], keep="first").reset_index(drop=True)
    price = add_price_features(price)

    samples = build_samples(news, price)
    samples = add_alpha_target(samples)
    print(f"Samples: {len(samples)} | {samples['date'].min().date()} → {samples['date'].max().date()}")

    # ── 2. Train/val/walk-forward masks ───────────────────────────
    train_mask, val_mask, _ = time_split(samples, TRAIN_END, VAL_END)
    wf_mask = (samples["date"] >= pd.Timestamp(sl_cfg["wf_start"])) & \
              (samples["date"] <= pd.Timestamp(sl_cfg["wf_end"]))
    print(f"Train: {train_mask.sum()} | Val: {val_mask.sum()} | WF pool: {wf_mask.sum()}")

    # ── 3. Features ───────────────────────────────────────────────
    num_feats = samples[NUMERIC_FEATURES].values.astype(np.float32)
    # Scaler fit on training data only — no look-ahead into 2018-2019
    scaler  = StandardScaler().fit(num_feats[train_mask.values])
    num_std = scaler.transform(num_feats).astype(np.float32)

    y_alpha = samples["alpha_7d"].values.astype(np.float32)
    y_raw   = samples["7th_day_return"].values.astype(np.float32)

    # ── 4. Embed ──────────────────────────────────────────────────
    prompts    = build_prompts(samples)
    embeddings = get_embeddings(prompts, cache_dir, device=DEVICE,
                                model_name=cfg["model"]["emb_model"])
    embeddings = align_embeddings(embeddings, prompts, prompts)

    X_all = np.concatenate([embeddings, num_std], axis=1)

    # ── 5. Train on 17H2, then freeze ─────────────────────────────
    emb_dim, num_dim = embeddings.shape[1], num_std.shape[1]
    train_ds = ArrayDataset(X_all[train_mask.values], y_alpha[train_mask.values])
    val_ds   = ArrayDataset(X_all[val_mask.values],   y_alpha[val_mask.values])

    print(f"\nTraining {tr_cfg['n_seeds']}-seed ensemble on 17H2...")
    nets = []
    for s in range(tr_cfg["n_seeds"]):
        net = MLPRegressor(emb_dim, num_dim,
                           emb_proj=cfg["model"]["emb_proj"],
                           hidden=tuple(cfg["model"]["hidden"]),
                           dropout=cfg["model"]["dropout"])
        nets.append(train_one(net, train_ds, val_ds, seed=SEED + s,
                              epochs=tr_cfg["epochs"], lr=tr_cfg["lr"],
                              weight_decay=tr_cfg["weight_decay"],
                              batch_size=tr_cfg["batch_size"],
                              patience=tr_cfg["patience"],
                              ic_lambda=tr_cfg["ic_lambda"], device=DEVICE))
    print("Ensemble frozen. Evaluating walk-forward windows...")

    # ── 6. Per-window evaluation ──────────────────────────────────
    windows = build_windows(sl_cfg["wf_start"], sl_cfg["wf_end"], sl_cfg["window_days"])
    print(f"\n{len(windows)} windows of {sl_cfg['window_days']} days each")

    rows, oos_frames = [], []
    for i, (ws, we) in enumerate(windows, 1):
        mask = (samples["date"] >= ws) & (samples["date"] <= we)
        n = mask.sum()
        if n == 0:
            continue
        preds = ensemble_predict(nets, X_all[mask.values], DEVICE)
        m = compute_metrics(preds, y_alpha[mask.values], y_raw[mask.values])
        rows.append({"window": f"W{i:02d}", "start": ws.date(), "end": we.date(), **m})
        sub = samples.loc[mask, ["stock", "date", "7th_day_return"]].copy()
        sub["pred"] = preds; sub["window"] = f"W{i:02d}"
        oos_frames.append(sub)

    metrics = pd.DataFrame(rows)
    print("\n" + metrics.to_string(index=False))

    # Pooled OOS metric
    pool_mask = wf_mask
    pool_preds = ensemble_predict(nets, X_all[pool_mask.values], DEVICE)
    pool_m = compute_metrics(pool_preds, y_alpha[pool_mask.values], y_raw[pool_mask.values])
    print("\n── Pooled 2018-2019 OOS ──")
    print_metrics("ALL-WF", pool_m)

    # ── 7. Backtest across all OOS windows ────────────────────────
    oos_df = pd.concat(oos_frames, ignore_index=True)
    hold   = bt_cfg["hold_days"]
    ppy    = 252 / hold

    strat = run_backtest(oos_df, price, bt_cfg["top_q"], bt_cfg["bot_q"],
                         bt_cfg["min_pool"], bt_cfg["min_leg"], hold)

    print(f"\n--- OOS backtest ({len(windows)} windows, {hold}-day rebalance) ---")
    summarize("Long-only",      strat["long_ret"],   ppy)
    summarize("Long-Short",     strat["ls_ret"],     ppy)
    summarize("Pool benchmark", strat["pool_bench"], ppy)
    summarize("Mkt benchmark",  strat["mkt_bench"],  ppy)

    # ── 8. Plots ──────────────────────────────────────────────────
    # 8a. Per-window IC bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.bar(metrics["window"], metrics["ic_alpha"], color="steelblue")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Per-window IC (vs alpha_7d)")
    ax.set_ylabel("IC"); ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelrotation=45)

    ax = axes[1]
    ax.bar(metrics["window"], metrics["dir_acc"], color="darkorange")
    ax.axhline(0.5, color="gray", lw=0.5, linestyle="--")
    ax.set_title("Per-window directional accuracy")
    ax.set_ylabel("DirAcc"); ax.set_ylim(0.3, 0.7); ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelrotation=45)

    plt.tight_layout()
    fig.savefig(out_dir / "sliding_per_window_ic.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "sliding_per_window_ic.pdf", bbox_inches="tight")
    print(f"\nSaved sliding_per_window_ic.png / .pdf")
    plt.show(); plt.close(fig)

    # 8b. Cumulative return across all OOS windows
    fig, ax = plt.subplots(figsize=(14, 5))
    for col, label, color, ls in [
        ("long_ret",   "Long-only",      "steelblue",  "-"),
        ("ls_ret",     "Long-Short",     "darkorange", "-"),
        ("pool_bench", "Pool benchmark", "#A9A9A9",    "--"),
        ("mkt_bench",  "Mkt benchmark",  "#7E5BB7",    ":"),
    ]:
        curve = (1 + strat[col].fillna(0)).cumprod() - 1
        ax.plot(strat["date"], curve, ls, color=color, lw=1.8,
                label=f"{label} ({curve.iloc[-1]:+.2%})")

    for ws, _ in windows:
        ax.axvline(ws, color="lightgray", lw=0.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"Walk-forward cumulative return — {len(windows)} × {sl_cfg['window_days']}-day windows")
    ax.set_xlabel("rebalance date"); ax.set_ylabel("cumulative return")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()

    fig.savefig(out_dir / "sliding_cumulative_return.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "sliding_cumulative_return.pdf", bbox_inches="tight")
    print("Saved sliding_cumulative_return.png / .pdf")
    plt.show(); plt.close(fig)


if __name__ == "__main__":
    main()
