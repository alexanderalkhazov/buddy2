"""Cross-sectional point-in-time dataset builder: TICKER + DATE + FEATURES -> LABELS.

Samples every SAMPLE_STRIDE trading days per ticker (not every single day) for
two reasons: (1) daily-overlapping 20D-forward labels are almost the same
observation repeated 20 times, which inflates apparent sample size without
adding independent information, and (2) it keeps a from-scratch build to a few
minutes on a laptop. This is a real, if modest, mitigation of the
"multiple-testing on the same path" problem — not a full purge, which happens
separately at train time via the embargo in train.py.
"""

from __future__ import annotations

import pandas as pd

from processing.ml.features import FEATURE_COLUMNS, compute_features
from processing.ml.labels import HORIZONS_DAYS, compute_labels
from processing.ml.universe import BENCHMARK, DEFAULT_UNIVERSE
from storage.cache import get_ohlcv

SAMPLE_STRIDE = 5
MIN_HISTORY_BARS = 210  # SMA200 + a warmup buffer


def build_dataset(
    universe: list[str] | None = None,
    horizons: tuple[int, ...] = HORIZONS_DAYS,
    refresh: bool = False,
) -> pd.DataFrame:
    universe = universe or DEFAULT_UNIVERSE
    bench_df = get_ohlcv(BENCHMARK, force=refresh)
    # Benchmark's OWN forward returns, computed the identical point-in-time-safe
    # way as any ticker's — needed to build RELATIVE ("did this stock beat SPY")
    # targets below without ever looking at a benchmark bar from the future.
    bench_labs = compute_labels(bench_df, horizons=horizons)

    rows = []
    for ticker in universe:
        try:
            df = get_ohlcv(ticker, force=refresh)
        except Exception as exc:
            print(f"  skip {ticker}: {exc}")
            continue
        if len(df) < MIN_HISTORY_BARS:
            print(f"  skip {ticker}: only {len(df)} bars, need {MIN_HISTORY_BARS}+")
            continue

        feats = compute_features(df, benchmark_df=bench_df)
        labs = compute_labels(df, horizons=horizons)

        merged = feats[FEATURE_COLUMNS].join(labs)

        # Relative targets: ticker's forward return minus the benchmark's forward
        # return over the SAME calendar dates (outer alignment by date, since a
        # ticker's trading-day index and SPY's are effectively identical for US
        # equities but this is defensive against holidays/data gaps).
        bench_aligned = bench_labs.reindex(merged.index)
        for h in horizons:
            merged[f"fwd_rel_ret_{h}d"] = merged[f"fwd_ret_{h}d"] - bench_aligned[f"fwd_ret_{h}d"]
            merged[f"fwd_up_rel_{h}d"] = (merged[f"fwd_rel_ret_{h}d"] > 0).astype("float")
            merged.loc[bench_aligned[f"fwd_ret_{h}d"].isna(), [f"fwd_rel_ret_{h}d", f"fwd_up_rel_{h}d"]] = float("nan")

        merged = merged.iloc[MIN_HISTORY_BARS:]  # drop warmup NaN region
        merged = merged.iloc[::SAMPLE_STRIDE]  # stride-sample to reduce label overlap
        merged = merged.dropna(subset=FEATURE_COLUMNS)  # keep only fully-formed feature rows

        merged.insert(0, "ticker", ticker)
        merged.insert(1, "date", merged.index)
        rows.append(merged.reset_index(drop=True))

    if not rows:
        raise RuntimeError("no tickers in the universe produced usable data")

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    return out


if __name__ == "__main__":
    ds = build_dataset()
    print(f"dataset: {len(ds)} rows across {ds['ticker'].nunique()} tickers, "
          f"{ds['date'].min().date()} to {ds['date'].max().date()}")
    print(ds[["ticker", "date", *FEATURE_COLUMNS[:3], "fwd_up_20d"]].tail(10))
