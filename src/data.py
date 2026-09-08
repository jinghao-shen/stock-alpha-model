"""
data.py — data loading and feature engineering.

All features are computed using only information available at date t,
so there is no look-ahead leakage into price-derived inputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["ret_1d", "ret_5d", "vol_5d", "vol_ratio", "range_pct"]


# ── Loading ───────────────────────────────────────────────────────

def load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["stock"] = df["stock"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def load_news(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "article_id"})
    df = df[["stock", "title", "date"]].copy()
    df["datetime"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["date"] = df["datetime"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    df["stock"] = df["stock"].astype(str).str.upper().str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df = df.dropna(subset=["datetime", "title", "stock"])
    df = df.loc[~df.duplicated(subset=["title", "date", "stock"])].reset_index(drop=True)
    return df


# ── Feature engineering ───────────────────────────────────────────

def add_price_features(price: pd.DataFrame) -> pd.DataFrame:
    """Add rolling return and vol features, grouped per ticker."""
    price = price.sort_values(["stock", "date"]).reset_index(drop=True)
    g = price.groupby("stock", sort=False)
    price["prev_close"] = g["Close"].shift(1)
    price["ret_1d"]     = price["Close"] / price["prev_close"] - 1.0
    price["ret_5d"]     = g["Close"].transform(lambda s: s / s.shift(5) - 1.0)
    price["vol_5d"]     = g["ret_1d"].transform(lambda s: s.rolling(5).std())
    price["vol_ratio"]  = price["Volume"] / g["Volume"].transform(lambda s: s.rolling(10).mean())
    price["range_pct"]  = (price["High"] - price["Low"]) / price["Close"]
    return price


# ── Sample table ──────────────────────────────────────────────────

def build_samples(news: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """
    Join news with same-day price to produce one row per (stock, date).
    Drops rows missing the 7-day forward return or any numeric feature
    (happens for tickers with fewer than 5 trading days of history).
    """
    news_sd = (
        news.sort_values(["stock", "date", "datetime"])
        .groupby(["stock", "date"])["title"]
        .apply(list)
        .reset_index()
        .rename(columns={"title": "titles"})
    )
    news_sd["n_titles"] = news_sd["titles"].apply(len)

    samples = news_sd.merge(price, on=["stock", "date"], how="inner")
    samples = samples.dropna(subset=["7th_day_return"])
    samples = samples.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)
    samples = samples.sort_values("date").reset_index(drop=True)
    return samples


def add_alpha_target(samples: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional alpha = stock 7d return minus the equal-weighted mean
    across all news-bearing stocks on the same date.

    Training on alpha rather than raw return removes the common market
    factor, so the model focuses on stock-specific signal.
    """
    samples["day_mean_ret"] = samples.groupby("date")["7th_day_return"].transform("mean")
    samples["alpha_7d"] = samples["7th_day_return"] - samples["day_mean_ret"]
    return samples


# ── Time split ────────────────────────────────────────────────────

def time_split(samples: pd.DataFrame, train_end: str, val_end: str):
    """Return boolean masks for train / val / test on a sorted samples df."""
    train_end = pd.Timestamp(train_end)
    val_end   = pd.Timestamp(val_end)
    train = samples["date"] <= train_end
    val   = (samples["date"] > train_end) & (samples["date"] <= val_end)
    test  = samples["date"] > val_end
    return train, val, test


def dynamic_split(samples: pd.DataFrame, train_frac=0.8, val_frac=0.1):
    """
    Split by unique dates rather than rows, so every date lands fully
    in one split (no date straddles train/test boundary).
    """
    dates = sorted(samples["date"].unique())
    n = len(dates)
    train_end = dates[int(n * train_frac) - 1]
    val_end   = dates[int(n * (train_frac + val_frac)) - 1]
    return time_split(samples, str(train_end.date()), str(val_end.date()))
