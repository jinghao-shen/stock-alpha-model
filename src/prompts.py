"""
prompts.py — prompt construction for Qwen embedding.

Each prompt merges same-day price context with news headlines.
Keeping the format fixed makes embeddings comparable across dates.
"""

import pandas as pd

MAX_TITLES = 5


def _pct(x, digits=2) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def _label(r) -> str:
    if pd.isna(r):
        return "unknown"
    if r >  0.02:  return "up strongly"
    if r >  0.005: return "up"
    if r < -0.02:  return "down strongly"
    if r < -0.005: return "down"
    return "flat"


def build_prompt(row, max_titles: int = MAX_TITLES) -> str:
    titles = row["titles"][:max_titles]
    if len(row["titles"]) > max_titles:
        titles = titles + [f"(+{len(row['titles']) - max_titles} more)"]
    titles_block = "\n".join(f"- {t}" for t in titles)
    return (
        f"Stock: {row['stock']}\n"
        f"Date: {row['date'].date()}\n"
        f"Price: open={row['Open']:.2f}, high={row['High']:.2f}, "
        f"low={row['Low']:.2f}, close={row['Close']:.2f}, volume={int(row['Volume'])}\n"
        f"Recent move: 1d={_pct(row['ret_1d'])} ({_label(row['ret_1d'])}), "
        f"5d={_pct(row['ret_5d'])}, 5d_vol={_pct(row['vol_5d'])}, "
        f"range={_pct(row['range_pct'])}, vol_ratio={row['vol_ratio']:.2f}x\n"
        f"News ({row['n_titles']}):\n{titles_block}"
    )


def build_prompts(samples: pd.DataFrame, max_titles: int = MAX_TITLES) -> list[str]:
    return [build_prompt(row, max_titles) for _, row in samples.iterrows()]
