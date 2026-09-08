"""
backtest.py — strategy simulation and performance reporting.

Strategy: top-Q% long, bottom-Q% short, non-overlapping HOLD-day rebalance.
We use non-overlapping windows because 7th_day_return is a 7-day forward return;
compounding it daily would double-count overlapping periods.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Per-date strategy ─────────────────────────────────────────────

def daily_strategy(group: pd.DataFrame, top_q: float, bot_q: float,
                   min_pool: int, min_leg: int) -> pd.Series:
    g = group.dropna(subset=["pred"]).sort_values("pred", ascending=False).reset_index(drop=True)
    n = len(g)
    if n < min_pool:
        return pd.Series({"n_pool": n, "long_ret": np.nan,
                          "short_ret": np.nan, "ls_ret": np.nan})
    n_long  = max(min_leg, int(round(n * top_q)))
    n_short = max(min_leg, int(round(n * bot_q)))
    long_ret  = g.head(n_long)["7th_day_return"].mean()
    short_ret = g.tail(n_short)["7th_day_return"].mean()
    return pd.Series({"n_pool": n, "long_ret": long_ret,
                      "short_ret": short_ret, "ls_ret": long_ret - short_ret})


def run_backtest(test_df: pd.DataFrame, price: pd.DataFrame,
                 top_q: float, bot_q: float, min_pool: int,
                 min_leg: int, hold_days: int) -> pd.DataFrame:
    """
    Returns one row per rebalance date with strategy and benchmark returns.
    Pool benchmark = equal-weight return across the news-pool stocks each date.
    Market benchmark = equal-weight return across the full price universe.
    """
    pool_bench = test_df.groupby("date")["7th_day_return"].mean().rename("pool_bench")
    mkt_bench = (
        price[price["7th_day_return"].notna()]
        .groupby("date")["7th_day_return"].mean()
        .rename("mkt_bench")
    )
    all_dates = (
        test_df.groupby("date", group_keys=False)
        .apply(lambda g: daily_strategy(g, top_q, bot_q, min_pool, min_leg),
               include_groups=False)
        .reset_index()
        .merge(pool_bench.reset_index(), on="date", how="left")
        .merge(mkt_bench.reset_index(), on="date", how="left")
        .sort_values("date").reset_index(drop=True)
    )
    # Sample every hold_days rows for non-overlapping windows
    strat = all_dates.iloc[::hold_days].dropna(subset=["ls_ret"]).reset_index(drop=True)
    return strat


# ── Performance summary ───────────────────────────────────────────

def summarize(name: str, ret: pd.Series, periods_per_year: float) -> dict:
    ret = pd.Series(ret).dropna()
    if len(ret) == 0:
        print(f"{name:<28}  (no data)")
        return {}
    mean = ret.mean()
    std  = ret.std(ddof=1)
    hit  = float((ret > 0).mean())
    cum  = float((1 + ret).prod() - 1)
    ann_ret = (1 + mean) ** periods_per_year - 1
    ann_vol = std * math.sqrt(periods_per_year)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    print(
        f"{name:<28} n={len(ret):>3} | mean={mean:+.4f} | hit={hit:.2%} | "
        f"cum={cum:+.2%} | AnnRet={ann_ret:+.2%} | AnnVol={ann_vol:.2%} | Sharpe={sharpe:.2f}"
    )
    return dict(name=name, n=len(ret), mean=mean, hit=hit, cum=cum,
                ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe)


# ── Plot ──────────────────────────────────────────────────────────

def plot_cumulative(strat_dict: dict[str, pd.DataFrame],
                    title: str, output_path=None):
    """
    strat_dict: {label: strat_df} where strat_df has columns date, long_ret,
                pool_bench, mkt_bench (at minimum).
    Saves PNG + PDF if output_path is given (stem used for both).
    """
    COLORS = {
        "main":  "#f16c23",
        "lstm":  "#1b7c3d",
        "ridge": "#2b6a99",
        "pool":  "#A9A9A9",
        "mkt":   "#7E5BB7",
    }

    fig, ax = plt.subplots(figsize=(11, 5))

    for label, strat in strat_dict.items():
        col  = "long_ret" if "bench" not in label.lower() else label.split("|")[1].strip()
        key  = label.lower().split()[0]
        color = COLORS.get(key, "#555555")
        lw    = 2.4 if key == "main" else 1.6
        ls    = "-" if "bench" not in label.lower() else ("--" if "pool" in label.lower() else ":")
        curve = (1 + strat[col].fillna(0)).cumprod() - 1
        ax.plot(strat["date"], curve, ls, color=color, lw=lw,
                label=f"{label} ({curve.iloc[-1]:+.2%})")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(title)
    ax.set_xlabel("rebalance date")
    ax.set_ylabel("cumulative return")
    ax.tick_params(axis="x", labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        from pathlib import Path
        p = Path(output_path)
        fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Saved {p.stem}.png / .pdf")

    plt.show()
    plt.close(fig)
