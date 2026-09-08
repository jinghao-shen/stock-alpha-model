"""
run_17h2.py — 17H2 experiment (training window: 2017-06-14 → 2017-10-18).

Trains the main Qwen+MLP model, then runs Ridge and LSTM baselines on the
same test split (2017-11-01 → 2017-12-29). Outputs a cumulative-return
comparison plot to outputs/.

Usage:
    python scripts/run_17h2.py --data-dir data/ --cache-dir cache/ --out-dir outputs/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader

from src.data import (load_price, load_news, add_price_features,
                      build_samples, add_alpha_target, time_split, NUMERIC_FEATURES)
from src.prompts import build_prompts
from src.embed import get_embeddings, align_embeddings
from src.model import MLPRegressor, LSTMRegressor, combined_loss
from src.train import ArrayDataset, SeqDataset, train_one, ensemble_predict, pearson_ic
from src.evaluate import compute_metrics, print_metrics
from src.backtest import run_backtest, summarize, plot_cumulative

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  default="data/")
    p.add_argument("--cache-dir", default="cache/")
    p.add_argument("--out-dir",   default="outputs/")
    p.add_argument("--config",    default="configs/base.yaml")
    return p.parse_args()


def build_lstm_sequences(samples, price, seq_len, seq_features):
    price_seq = price.dropna(subset=seq_features).sort_values(["stock", "date"]).reset_index(drop=True)
    arr, dates = {}, {}
    for stock, sub in price_seq.groupby("stock", sort=False):
        arr[stock]   = sub[seq_features].values.astype(np.float32)
        dates[stock] = sub["date"].values

    n = len(samples)
    seqs  = np.zeros((n, seq_len, len(seq_features)), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, row in enumerate(samples[["stock", "date"]].itertuples(index=False)):
        s, d = str(row.stock), np.datetime64(row.date)
        if s not in arr:
            continue
        ds  = dates[s]
        pos = np.searchsorted(ds, d)
        if pos >= len(ds) or ds[pos] != d or pos < seq_len - 1:
            continue
        seqs[i]  = arr[s][pos - seq_len + 1: pos + 1]
        valid[i] = True
    return seqs, valid


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_cfg  = cfg["training"]
    bt_cfg  = cfg["backtest"]
    bl_cfg  = cfg["baselines"]
    SEED    = tr_cfg["seed"]
    np.random.seed(SEED); torch.manual_seed(SEED)

    # ── 1. Load & prepare data ────────────────────────────────────
    print("Loading data...")
    price = load_price(data_dir / "yahoo_2017_12_stock_data.csv")
    news  = load_news(data_dir  / "financials_jun-dec2017_news.csv")
    price = add_price_features(price)

    samples = build_samples(news, price)
    samples = add_alpha_target(samples)
    print(f"Samples: {len(samples)} | {samples['date'].min().date()} → {samples['date'].max().date()}")

    # ── 2. Split ──────────────────────────────────────────────────
    train_mask, val_mask, test_mask = time_split(samples, "2017-10-18", "2017-10-31")
    print(f"Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

    # ── 3. Standardize numeric features ──────────────────────────
    num_feats = samples[NUMERIC_FEATURES].values.astype(np.float32)
    scaler    = StandardScaler().fit(num_feats[train_mask.values])
    num_std   = scaler.transform(num_feats).astype(np.float32)

    y_alpha = samples["alpha_7d"].values.astype(np.float32)
    y_raw   = samples["7th_day_return"].values.astype(np.float32)

    # ── 4. Embed prompts ──────────────────────────────────────────
    prompts    = build_prompts(samples)
    embeddings = get_embeddings(prompts, cache_dir, device=DEVICE,
                                model_name=cfg["model"]["emb_model"])
    embeddings = align_embeddings(embeddings, prompts, prompts)

    X_all = np.concatenate([embeddings, num_std], axis=1)
    print(f"Feature matrix: {X_all.shape}")

    # ── 5. Train main model ───────────────────────────────────────
    emb_dim, num_dim = embeddings.shape[1], num_std.shape[1]
    train_ds = ArrayDataset(X_all[train_mask.values], y_alpha[train_mask.values])
    val_ds   = ArrayDataset(X_all[val_mask.values],   y_alpha[val_mask.values])

    print(f"\nTraining {tr_cfg['n_seeds']}-seed MLP ensemble...")
    nets = []
    for s in range(tr_cfg["n_seeds"]):
        net = MLPRegressor(emb_dim, num_dim,
                           emb_proj=cfg["model"]["emb_proj"],
                           hidden=tuple(cfg["model"]["hidden"]),
                           dropout=cfg["model"]["dropout"])
        net = train_one(net, train_ds, val_ds, seed=SEED + s,
                        epochs=tr_cfg["epochs"], lr=tr_cfg["lr"],
                        weight_decay=tr_cfg["weight_decay"],
                        batch_size=tr_cfg["batch_size"],
                        patience=tr_cfg["patience"],
                        ic_lambda=tr_cfg["ic_lambda"], device=DEVICE)
        nets.append(net)

    # ── 6. Evaluate main model ────────────────────────────────────
    print("\n--- Main model metrics ---")
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        preds = ensemble_predict(nets, X_all[mask.values], DEVICE)
        print_metrics(name, compute_metrics(preds, y_alpha[mask.values], y_raw[mask.values]))

    pred_main_test = ensemble_predict(nets, X_all[test_mask.values], DEVICE)

    # ── 7. Ridge baseline ─────────────────────────────────────────
    ridge = RidgeCV(alphas=bl_cfg["ridge_alphas"]).fit(
        num_std[train_mask.values], y_alpha[train_mask.values])
    pred_ridge_test = ridge.predict(num_std[test_mask.values]).astype(np.float32)
    print(f"\n--- Ridge baseline (alpha={ridge.alpha_}) ---")
    print_metrics("test", compute_metrics(pred_ridge_test,
                                          y_alpha[test_mask.values], y_raw[test_mask.values]))

    # ── 8. LSTM baseline ──────────────────────────────────────────
    bl_lstm = bl_cfg["lstm"]
    seqs, valid = build_lstm_sequences(
        samples, price, bl_lstm["seq_len"], bl_lstm["seq_features"])

    tr_idx = np.where(valid & train_mask.values)[0]
    va_idx = np.where(valid & val_mask.values)[0]
    te_idx = np.where(valid & test_mask.values)[0]

    flat = seqs[tr_idx].reshape(-1, len(bl_lstm["seq_features"]))
    seq_mean, seq_std = flat.mean(0), flat.std(0) + 1e-8
    seqs_std = ((seqs - seq_mean) / seq_std).astype(np.float32)

    lstm_train_ds = SeqDataset(seqs_std[tr_idx], y_alpha[tr_idx])
    lstm_val_ds   = SeqDataset(seqs_std[va_idx], y_alpha[va_idx])

    print(f"\nTraining {bl_lstm['n_seeds']}-seed LSTM ensemble...")
    lstm_nets = []
    for s in range(bl_lstm["n_seeds"]):
        net = LSTMRegressor(n_feat=len(bl_lstm["seq_features"]),
                            hidden=bl_lstm["hidden"], num_layers=bl_lstm["num_layers"],
                            dropout=bl_lstm["dropout"])
        net = train_one(net, lstm_train_ds, lstm_val_ds, seed=SEED + s,
                        epochs=tr_cfg["epochs"], lr=3e-4,
                        weight_decay=tr_cfg["weight_decay"],
                        batch_size=tr_cfg["batch_size"],
                        patience=tr_cfg["patience"],
                        ic_lambda=tr_cfg["ic_lambda"], device=DEVICE)
        lstm_nets.append(net)

    pred_lstm_test_full = np.full(int(test_mask.sum()), np.nan, dtype=np.float32)
    pred_lstm_test_full[valid[test_mask.values]] = ensemble_predict(
        lstm_nets, seqs_std[te_idx], DEVICE)

    print("\n--- LSTM baseline ---")
    te_valid = valid[test_mask.values]
    print_metrics("test", compute_metrics(
        pred_lstm_test_full[te_valid],
        y_alpha[test_mask.values][te_valid],
        y_raw[test_mask.values][te_valid]))

    # ── 9. Backtest ───────────────────────────────────────────────
    hold = bt_cfg["hold_days"]
    ppy  = 252 / hold

    test_df = samples.loc[test_mask, ["stock", "date", "7th_day_return"]].copy().reset_index(drop=True)

    def _strat(preds):
        df = test_df.copy(); df["pred"] = preds
        return run_backtest(df, price, bt_cfg["top_q"], bt_cfg["bot_q"],
                            bt_cfg["min_pool"], bt_cfg["min_leg"], hold)

    strat_main  = _strat(pred_main_test)
    strat_ridge = _strat(pred_ridge_test)
    strat_lstm  = _strat(pred_lstm_test_full)

    print(f"\n--- Backtest (non-overlapping {hold}-day windows) ---")
    summarize("Main long-only",   strat_main["long_ret"],  ppy)
    summarize("Ridge long-only",  strat_ridge["long_ret"], ppy)
    summarize("LSTM long-only",   strat_lstm["long_ret"],  ppy)
    summarize("Pool benchmark",   strat_main["pool_bench"], ppy)
    summarize("Mkt benchmark",    strat_main["mkt_bench"],  ppy)

    # ── 10. Plot & save ───────────────────────────────────────────
    # Build unified strat_dict; benchmarks reference pool/mkt columns
    strat_pool = strat_main.rename(columns={"pool_bench": "long_ret"})
    strat_mkt  = strat_main.rename(columns={"mkt_bench":  "long_ret"})

    plot_cumulative(
        {
            "Main":              strat_main,
            "LSTM":              strat_lstm,
            "Ridge":             strat_ridge,
            "Pool | pool_bench": strat_main.assign(long_ret=strat_main["pool_bench"]),
            "Mkt  | mkt_bench":  strat_main.assign(long_ret=strat_main["mkt_bench"]),
        },
        title=f"17H2 cumulative return — Main vs baselines (TOP_Q=20%, HOLD={hold}d)",
        output_path=out_dir / "17h2_cumulative_return",
    )


if __name__ == "__main__":
    main()
