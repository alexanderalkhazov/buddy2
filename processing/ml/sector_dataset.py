"""Sector-level point-in-time dataset — DATE x SECTOR -> forward sector return.
Reuses the exact same features.py/labels.py machinery as the stock-level
dataset, run on each sector's synthetic index instead of one ticker's real
price series. Also attaches sector BREADTH (computed from the already-built
stock-level Phase 3 dataset — % of that sector's member-stock rows above their
own SMA20/50/200 on each date — free, no new fetches) and market regime.
"""

from __future__ import annotations

import pandas as pd

from processing.ml.features import FEATURE_COLUMNS, compute_features
from processing.ml.labels import HORIZONS_DAYS, compute_labels
from processing.ml.regime import compute_regime_features
from processing.ml.sector_index import build_all_sector_indices
from processing.ml.universe import BENCHMARK, EXPANDED_UNIVERSE_V2
from storage.cache import get_ohlcv

MIN_HISTORY_BARS = 210
SAMPLE_STRIDE = 5


def build_sector_dataset(
    horizons: tuple[int, ...] = HORIZONS_DAYS,
    refresh: bool = False,
    stock_dataset: pd.DataFrame | None = None,
) -> pd.DataFrame:
    bench_df = get_ohlcv(BENCHMARK, force=refresh)
    bench_labs = compute_labels(bench_df, horizons=horizons)
    regime = compute_regime_features(bench_df)

    sector_indices = build_all_sector_indices(refresh=refresh)

    rows = []
    for sector, df in sector_indices.items():
        feats = compute_features(df, benchmark_df=bench_df)
        labs = compute_labels(df, horizons=horizons)
        merged = feats[FEATURE_COLUMNS].join(labs)

        bench_aligned = bench_labs.reindex(merged.index)
        for h in horizons:
            merged[f"fwd_rel_ret_{h}d"] = merged[f"fwd_ret_{h}d"] - bench_aligned[f"fwd_ret_{h}d"]
            merged[f"fwd_up_rel_{h}d"] = (merged[f"fwd_rel_ret_{h}d"] > 0).astype("float")
            merged.loc[bench_aligned[f"fwd_ret_{h}d"].isna(), [f"fwd_rel_ret_{h}d", f"fwd_up_rel_{h}d"]] = float("nan")

        merged = merged.join(regime.reindex(merged.index))
        merged = merged.iloc[MIN_HISTORY_BARS:]
        merged = merged.iloc[::SAMPLE_STRIDE]
        merged = merged.dropna(subset=FEATURE_COLUMNS)

        merged.insert(0, "sector", sector)
        merged.insert(1, "date", merged.index)
        rows.append(merged.reset_index(drop=True))

    out = pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)

    if stock_dataset is not None:
        breadth = _compute_breadth(stock_dataset)
        out = out.merge(breadth, on=["sector", "date"], how="left")

    return out


def _compute_breadth(stock_dataset: pd.DataFrame) -> pd.DataFrame:
    df = stock_dataset.copy()
    df["above_sma20"] = df["dist_sma20_pct"] > 0
    df["above_sma50"] = df["dist_sma50_pct"] > 0
    df["above_sma200"] = df["dist_sma200_pct"] > 0
    df["positive_20d"] = df["ret_20d"] > 0
    df["positive_60d"] = df["ret_60d"] > 0

    breadth = df.groupby(["sector", "date"]).agg(
        breadth_pct_above_sma20=("above_sma20", "mean"),
        breadth_pct_above_sma50=("above_sma50", "mean"),
        breadth_pct_above_sma200=("above_sma200", "mean"),
        breadth_pct_positive_20d=("positive_20d", "mean"),
        breadth_pct_positive_60d=("positive_60d", "mean"),
        breadth_median_rsi=("rsi14", "median"),
        breadth_n_stocks=("ticker", "nunique"),
    ).reset_index()
    for c in breadth.columns:
        if c.startswith("breadth_pct"):
            breadth[c] = (breadth[c] * 100).round(2)
    return breadth


if __name__ == "__main__":
    from processing.ml.dataset_v2 import build_dataset_v2

    stock_ds, _ = build_dataset_v2()
    sector_ds = build_sector_dataset(stock_dataset=stock_ds)
    print(sector_ds.shape)
    print(sector_ds[["sector", "date", "atr_pct", "realized_vol_20d", "fwd_ret_20d", "breadth_pct_above_sma50"]].tail(10))
