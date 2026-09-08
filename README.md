# News-Driven Stock Alpha Prediction via Qwen Embeddings + IC-Aware MLP

## Overview

This project predicts **cross-sectional stock alpha** by combining financial news
embeddings with price-based technical features. The core contribution is a custom
**IC-penalized loss function** that directly optimizes cross-sectional ranking quality,
rather than minimizing point prediction error.

The model is evaluated across three experimental windows (2017-H2, 2018-2019, and a
12-window walk-forward test) to assess both in-sample fit and temporal generalization.

---

## Problem Framing

Standard regression on raw 7-day returns conflates stock-specific signal with
market-wide moves. We instead predict **cross-sectional alpha**:

```
alpha_7d(stock, date) = 7d_return(stock, date) − mean(7d_return of all stocks on date)
```

This demean removes the common daily factor, making the target zero-sum per date and
forcing the model to learn *relative* outperformance rather than market direction.

---

## Model

```
news headlines + price context
        │
   Qwen3-Embedding-0.6B          (1024-dim, weights frozen)
        │
   Linear(1024 → 128)            (bottleneck projection)
        │
   concat with 5 numeric features
   [ret_1d, ret_5d, vol_5d, vol_ratio, range_pct]
        │
   MLP  [64]  dropout=0.3
        │
   scalar  →  alpha_7d
```

Numeric features are standardized using training-set statistics only (no look-ahead).
The embedding bottleneck reduces the 1024-dim text representation to 128 dims before
concatenation, preventing the numeric features from being drowned out.

---

## Loss Function

Training minimizes a combined objective:

```
L = MSE(pred, alpha) + λ · (1 − PearsonIC(pred, alpha))
```

The IC penalty term directly rewards cross-sectional rank correlation within each
mini-batch. Setting λ = 0.5 balances absolute error against ranking quality.
Early stopping uses **validation IC** (not validation loss) as the selection criterion.

An ensemble of 5 independently-seeded models is trained; final predictions are the
mean across seeds.

---

## Baselines

| Model | Features | Notes |
|---|---|---|
| **Ridge** | 5 numeric price features | RidgeCV, no text signal |
| **LSTM** | 20-day sequences of [ret_1d, vol_ratio] | Price dynamics only |
| **Main (Qwen+MLP)** | News embeddings + 5 numeric features | IC-penalized loss |

All baselines share the same train/val/test split and early-stopping criterion.

---

## Experiments

### Experiment 1 — 17H2 (Fixed Split)

| | Dates |
|---|---|
| Train | 2017-06-14 → 2017-10-18 |
| Validation | 2017-10-19 → 2017-10-31 |
| Test | 2017-11-01 → 2017-12-29 |

Evaluates whether the model captures news-driven alpha over a 7-month in-sample period.
Compared against Ridge and LSTM on the same held-out test window.

```bash
python scripts/run_17h2.py
```

Output: `outputs/17h2_cumulative_return.png / .pdf`

---

### Experiment 2 — 2018-2019 (Dynamic Split)

Split: **80 / 10 / 10** by unique calendar dates across the full 2018-2019 dataset.

Tests whether the same architecture generalizes to a different two-year regime with
substantially different market conditions than the 17H2 training window.

```bash
python scripts/run_18_19.py
```

Output: `outputs/18_19_cumulative_return.png / .pdf`

---

### Experiment 3 — Walk-Forward Generalization (12 × 59-day Windows)

The 17H2-trained ensemble is **frozen** and evaluated on 12 consecutive non-overlapping
59-day windows across 2018-01-01 → 2019-12-31, without any retraining.

This is a strict out-of-sample test: the model never observes 2018-2019 data during
training. Per-window IC and directional accuracy measure whether the learned signal
degrades over time.

```bash
python scripts/run_sliding.py
```

Outputs:
- `outputs/sliding_per_window_ic.png / .pdf` — IC and directional accuracy per window
- `outputs/sliding_cumulative_return.png / .pdf` — cumulative return across all windows

---

## Backtest Methodology

Strategy: long top-20%, short bottom-20% by predicted alpha on each rebalance date.
Rebalance is **non-overlapping every 7 days**, matching the 7-day forward return horizon.

> Overlapping rebalance windows would double-count the same return across consecutive
> periods. Non-overlapping windows ensure each realized return is counted exactly once.

Two benchmarks are tracked alongside the strategy:
- **Pool benchmark** — equal-weight return across all stocks in the news pool per date
- **Market benchmark** — equal-weight return across the full price universe per date

---

## Repository Structure

```
stock-alpha-model/
├── configs/base.yaml      # all hyperparameters
├── src/
│   ├── data.py            # feature engineering, alpha target, splits
│   ├── prompts.py         # text prompt construction per (stock, date)
│   ├── embed.py           # Qwen inference + MD5-keyed disk cache
│   ├── model.py           # MLPRegressor, LSTMRegressor, IC loss
│   ├── train.py           # training loop, early stopping, ensemble
│   ├── evaluate.py        # IC, directional accuracy, MSE/MAE
│   └── backtest.py        # strategy simulation, benchmarks, plots
└── scripts/
    ├── run_17h2.py        # Experiment 1
    ├── run_18_19.py       # Experiment 2
    └── run_sliding.py     # Experiment 3
```

Place raw CSV files under `data/` and run any script directly:

```bash
pip install -r requirements.txt
python scripts/run_17h2.py --data-dir data/ --out-dir outputs/
```

Embeddings are cached to `cache/` after the first run (keyed by MD5 hash of the prompt
list) so subsequent runs skip re-inference.
