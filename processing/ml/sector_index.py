"""Synthetic, point-in-time-safe sector price indices — equal-weighted OHLC
average across each sector's member tickers (each ticker's own OHLC rebased
to 100 at the index's start date, then averaged elementwise). Averaging valid
OHLC bars elementwise preserves high >= close >= low (since it holds for every
member on every day, it holds for the elementwise average), so the resulting
synthetic series is a well-formed OHLC frame the existing indicators/features
pipeline can run on UNCHANGED — this is deliberate reuse, not a new feature
pipeline: build_dataset()-style construction just runs on a sector's synthetic
series instead of one ticker's real series.
"""

from __future__ import annotations

import pandas as pd

from processing.ml.universe import EXPANDED_UNIVERSE_V2
from storage.cache import get_ohlcv

MIN_MEMBER_BARS = 1000  # ~4y — excludes bad/short fetches (e.g. a stale/broken cache row) from dragging the whole sector index down to their date range


def build_sector_index(sector: str, refresh: bool = False) -> pd.DataFrame:
    tickers = EXPANDED_UNIVERSE_V2[sector]
    frames = []
    for t in tickers:
        try:
            df = get_ohlcv(t, force=refresh)
        except Exception:
            continue
        if df.empty or len(df) < MIN_MEMBER_BARS:
            continue
        # yfinance occasionally returns an all-NaN OHLC bar (seen on the very
        # first cached row for some tickers) — dropping it here, not just at
        # the tail, matters because NaN propagates through both the rebase
        # division AND every subsequent elementwise sum across the sector's
        # members, silently poisoning the WHOLE synthetic index, not just one
        # date, if left in.
        df = df.dropna(subset=["open", "high", "low", "close"])
        if len(df) < MIN_MEMBER_BARS:
            continue
        base = df[["open", "high", "low", "close"]].iloc[0]
        if (base <= 0).any():
            continue
        rebased = df[["open", "high", "low", "close"]] / base * 100
        rebased["volume"] = df["volume"]
        frames.append(rebased)

    if len(frames) < 3:
        raise RuntimeError(f"sector {sector!r} has too few usable tickers ({len(frames)}) to build an index")

    # Inner-join on dates every member has, so the index never averages over a
    # day where only some members reported (which would silently shift its level).
    common_dates = frames[0].index
    for f in frames[1:]:
        common_dates = common_dates.intersection(f.index)
    aligned = [f.loc[common_dates] for f in frames]

    index_df = sum(aligned) / len(aligned)
    index_df["volume"] = sum(f["volume"] for f in aligned)  # sum, not average — a scale-free proxy is fine, only used for z-scoring
    return index_df.sort_index()


def build_all_sector_indices(refresh: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for sector in EXPANDED_UNIVERSE_V2:
        try:
            out[sector] = build_sector_index(sector, refresh=refresh)
        except Exception as exc:
            print(f"  skip sector {sector}: {exc}")
    return out
