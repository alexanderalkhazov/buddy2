"""Forward-looking labels — the only place in this module allowed to look ahead
of the observation date, and only in the direction of the future (never used as
a feature). Each label at row t is computed strictly from rows (t, t+horizon],
never from row t itself or earlier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS_DAYS = (1, 3, 5, 10, 20, 40, 60)


def compute_labels(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS_DAYS) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    out = pd.DataFrame(index=df.index)

    for h in horizons:
        fwd_close = close.shift(-h)
        out[f"fwd_ret_{h}d"] = (fwd_close / close - 1) * 100
        out[f"fwd_up_{h}d"] = (out[f"fwd_ret_{h}d"] > 0).astype("float")

        # Maximum favorable/adverse excursion over (t, t+h] — highest high / lowest
        # low in the h bars strictly AFTER t, relative to the entry (close at t).
        fwd_high_max = high.shift(-1).rolling(h).max().shift(-(h - 1))
        fwd_low_min = low.shift(-1).rolling(h).min().shift(-(h - 1))
        out[f"fwd_mfe_{h}d"] = (fwd_high_max / close - 1) * 100
        out[f"fwd_mae_{h}d"] = (fwd_low_min / close - 1) * 100

        # NaN out the label wherever the forward window would run off the end of
        # the available data — an incomplete future window must never silently
        # become "labeled as flat."
        insufficient = df.index >= df.index[max(len(df.index) - h, 0)]
        out.loc[insufficient, [f"fwd_ret_{h}d", f"fwd_up_{h}d", f"fwd_mfe_{h}d", f"fwd_mae_{h}d"]] = np.nan

    return out
