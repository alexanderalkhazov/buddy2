"""Point-in-time market-regime labels, built ONLY from the benchmark's own
trailing price path — never from an individual stock's future, and never from
information not available as of each row's own date. This module is separate
from processing/regime.py (which answers "what's the regime right now" for a
live report) — this one produces a full historical, row-by-row regime label
so it can be joined into the ML dataset for conditional testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_regime_features(bench_df: pd.DataFrame) -> pd.DataFrame:
    """bench_df: SPY's own OHLCV. Returns a frame indexed the same way with
    trend_regime (bull/bear, from price vs its own 200SMA — trailing) and
    vol_regime (low/mid/high, from a 252-day EXPANDING percentile rank of
    20D realized vol, so the label at row t only ranks against history up to
    t, never against the whole sample including the future)."""
    close = bench_df["close"]
    sma200 = close.rolling(200).mean()
    trend_regime = np.where(close > sma200, "bull", "bear")

    realized_vol = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    # expanding rank against ONLY prior history (min_periods=60 so the first
    # ~60 rows, with too little history to rank meaningfully, are NaN not a
    # fabricated percentile)
    vol_percentile = realized_vol.expanding(min_periods=60).apply(
        lambda window: pd.Series(window).rank(pct=True).iloc[-1], raw=False
    )
    vol_regime = pd.cut(
        vol_percentile, bins=[-0.01, 0.33, 0.67, 1.01], labels=["low_vol", "mid_vol", "high_vol"]
    )

    out = pd.DataFrame(
        {
            "trend_regime": pd.Series(trend_regime, index=bench_df.index),
            "vol_regime": vol_regime,
            "spy_realized_vol_20d": realized_vol,
        },
        index=bench_df.index,
    )
    out.loc[sma200.isna(), "trend_regime"] = None
    return out
