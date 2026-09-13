"""The genuine non-overlapping forward-return test that Phase 5 flagged as
NOT_TESTED (it only approximated this via a coarser resample of an
already-stride-5 dataset). This builds the real thing: for each sector's
full daily synthetic index, observations are spaced EXACTLY `horizon`
trading days apart, so no two observations' forward-return windows ever
share a single day.

Observation i sits at raw row index i*horizon (after the feature warmup
region). Its forward return uses close[i*horizon + horizon] / close[i*horizon]
- 1 — by construction this window is disjoint from every other observation's
window for the same sector, unlike the existing stride-5 dataset where
consecutive rows are only 5 trading days apart against a 20-day label (75%
day-overlap).

This does NOT modify dataset.py/dataset_v2.py/sector_dataset.py (Phase 1-4's
stride-5 pipeline) — it's a separate, stricter construction reading the same
underlying sector indices.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from processing.ml.features import FEATURE_COLUMNS, compute_features
from processing.ml.registry import ARTIFACT_DIR, log_stat_experiment
from processing.ml.sector_index import build_sector_index
from processing.ml.universe import BENCHMARK, EXPANDED_UNIVERSE_V2
from storage.cache import get_ohlcv

MIN_WARMUP_BARS = 210  # SMA200 + buffer, same as the rest of the pipeline


def _sector_non_overlapping_observations(sector: str, horizon: int, bench_df: pd.DataFrame) -> pd.DataFrame:
    idx_df = build_sector_index(sector)
    feats = compute_features(idx_df, benchmark_df=bench_df)
    valid = feats.dropna(subset=["close", *FEATURE_COLUMNS])
    close = valid["close"]

    rows = []
    n = len(valid)
    # Start after warmup, step by exactly `horizon` rows so consecutive
    # observations' forward windows are disjoint by construction.
    starts = range(MIN_WARMUP_BARS, n - horizon, horizon)
    for i in starts:
        entry_date = valid.index[i]
        exit_idx = i + horizon
        if exit_idx >= n:
            break
        entry_close = close.iloc[i]
        exit_close = close.iloc[exit_idx]
        if entry_close <= 0:
            continue
        fwd_ret = (exit_close / entry_close - 1) * 100
        row = {"sector": sector, "date": entry_date, "fwd_ret": fwd_ret}
        for col in FEATURE_COLUMNS:
            row[col] = valid[col].iloc[i]
        rows.append(row)
    return pd.DataFrame(rows)


def run(horizon: int = 20) -> dict:
    bench_df = get_ohlcv(BENCHMARK)
    all_obs = []
    for sector in EXPANDED_UNIVERSE_V2:
        try:
            obs = _sector_non_overlapping_observations(sector, horizon, bench_df)
            all_obs.append(obs)
        except Exception as exc:
            print(f"  skip {sector}: {exc}")

    df = pd.concat(all_obs, ignore_index=True)
    df = df.dropna(subset=["realized_vol_20d", "ret_20d", "fwd_ret"])

    ic, n = None, len(df)
    if n >= 20:
        r = spearmanr(df["realized_vol_20d"], df["fwd_ret"]).statistic
        ic = round(float(r), 4) if r == r else None

    # High-vol/down bucket effect, same definition as Phase 3/4/5, now on
    # observations with ZERO shared days between any two rows for the same
    # sector.
    bucket_effect = None
    if n >= 40:
        d = df.copy()
        d["vol_bucket"] = pd.qcut(d["realized_vol_20d"], 2, labels=["low_vol", "high_vol"], duplicates="drop")
        d["mom_bucket"] = pd.qcut(d["ret_20d"], 2, labels=["down", "up"], duplicates="drop")
        agg = d.groupby(["vol_bucket", "mom_bucket"], observed=True)["fwd_ret"].mean()
        if ("high_vol", "down") in agg.index:
            bucket_effect = round(float(agg[("high_vol", "down")] - d["fwd_ret"].mean()), 3)

    # A per-sector observation count matters here: with only 13 sectors and
    # non-overlapping ~20-trading-day spacing over ~4 years of data, each
    # sector contributes roughly 4y * 252 / 20 ≈ 50 observations, so total n
    # is expected to be small (a genuine, disclosed limitation of testing
    # this rigorously on only 13 sectors instead of hundreds of stocks).
    per_sector_counts = df.groupby("sector").size().to_dict()

    result = {
        "horizon_days": horizon,
        "n_total_observations": n,
        "n_sectors": df["sector"].nunique() if n else 0,
        "per_sector_observation_counts": per_sector_counts,
        "ic": ic,
        "high_vol_down_bucket_effect": bucket_effect,
        "status": "PASS" if (ic is not None and ic > 0) else ("FAIL" if ic is not None else "INCONCLUSIVE — insufficient non-overlapping observations"),
        "note": (
            "TRUE non-overlapping test: observations are spaced exactly `horizon_days` trading days apart "
            "per sector, so no two observations for the same sector share a single day in their forward-"
            "return window (unlike the Phase 1-4 stride-5 dataset, where consecutive 20D-labeled "
            "observations share 15 of 20 days). Sample size is necessarily small (13 sectors x ~4y of "
            "non-overlapping 20-day windows) — a real, disclosed limitation of this stricter test, not a "
            "bug. This is the test previously logged as NOT_TESTED in phase5_edge_stress_test.py."
        ),
    }
    log_stat_experiment("non_overlapping_forward_return_test", "phase5", result)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "non_overlapping_test_report.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2, default=str))
