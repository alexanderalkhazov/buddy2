"""Point-in-time-safe feature computation.

Every column here is computed with pandas `.rolling()` / `.shift()` operations
that, at row t, only ever read rows <= t. That is the entire leakage-prevention
mechanism for the price/technical family: there is no separate "as-of" filter
step because the arithmetic itself cannot see the future. This is verified by
`tests/test_ml_leakage.py`.

Deliberately restricted to price/technical + relative-strength features.
Fundamentals, analyst estimates, earnings, options and news are NOT included
here — yfinance only exposes their CURRENT snapshot, not what was known as of
a past date, so any historical row built from them would silently leak future
information into the label. See processing/ml/README.md for the full
data-availability audit. Building a real point-in-time fundamentals table
requires a vendor that sells point-in-time data (e.g. a fundamentals feed with
as-reported + revision history); this module is structured so that once such a
source exists, its features can be joined into `dataset.py` the same way these
are, without changing the labeling or leakage-testing machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from processing.indicators import enrich

FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "rsi14", "macd_hist_norm",
    "atr_pct", "realized_vol_20d",
    "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
    "volume_zscore",
    "rel_ret_20d_vs_benchmark",
]


def compute_features(df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """df: raw OHLCV (ticker). benchmark_df: raw OHLCV for the relative-strength
    benchmark (e.g. SPY), aligned by date-intersection, also trailing-only.
    Returns a frame indexed the same as df with FEATURE_COLUMNS added — rows
    before enough history exists (SMA200 needs 200 bars) are NaN, not guessed."""
    out = enrich(df).copy()
    close = out["close"]

    out["ret_1d"] = close.pct_change(1) * 100
    out["ret_5d"] = close.pct_change(5) * 100
    out["ret_10d"] = close.pct_change(10) * 100
    out["ret_20d"] = close.pct_change(20) * 100
    out["ret_60d"] = close.pct_change(60) * 100

    # macd_hist scaled by price so it's comparable across tickers of very
    # different price levels (a $5 histogram move means nothing on a $2000 stock).
    out["macd_hist_norm"] = out["hist"] / close * 100
    out["atr_pct"] = out["atr14"] / close * 100
    out["realized_vol_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    out["dist_sma20_pct"] = (close - out["sma20"]) / out["sma20"] * 100
    out["dist_sma50_pct"] = (close - out["sma50"]) / out["sma50"] * 100
    out["dist_sma200_pct"] = (close - out["sma200"]) / out["sma200"] * 100

    vol_mean = out["volume"].rolling(60).mean()
    vol_std = out["volume"].rolling(60).std()
    out["volume_zscore"] = (out["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    if benchmark_df is not None and not benchmark_df.empty:
        bench_close = benchmark_df["close"].reindex(out.index).ffill()
        bench_ret_20d = bench_close.pct_change(20) * 100
        out["rel_ret_20d_vs_benchmark"] = out["ret_20d"] - bench_ret_20d
    else:
        out["rel_ret_20d_vs_benchmark"] = np.nan

    return out
