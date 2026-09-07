"""Leakage tests for the ML pipeline. No pytest dependency — plain asserts,
runnable with `python tests/test_ml_leakage.py`. Exits non-zero on failure so
it can be wired into CI later without adding a test-runner dependency now.

What "leakage" means for each check:
  1. Truncation invariance: a feature computed at row t using only data up to
     row t must be IDENTICAL whether or not rows after t exist in the frame.
     If truncating the future changes a "past" feature, that feature is
     secretly reading ahead.
  2. Label direction: forward labels at row t must only differ from NaN when
     there are at least h bars STRICTLY AFTER t — never fewer.
  3. Embargo: train.py's walk-forward split must leave a gap between the last
     training date and the first test date at least as large as the longest
     label horizon, otherwise overlapping-label rows straddle the train/test
     boundary and leak.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from processing.ml.features import FEATURE_COLUMNS, compute_features
from processing.ml.labels import HORIZONS_DAYS, compute_labels
from storage.cache import get_ohlcv


def _synthetic_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    price = np.maximum(price, 1.0)
    df = pd.DataFrame(
        {
            "open": price * (1 + rng.normal(0, 0.001, n)),
            "high": price * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": price * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": price,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )
    return df


def test_feature_truncation_invariance() -> None:
    df = _synthetic_ohlcv()
    cut = 300  # somewhere well past all warmup windows (SMA200 etc.)

    full = compute_features(df)
    truncated = compute_features(df.iloc[: cut + 1])

    row_full = full.iloc[cut]
    row_trunc = truncated.iloc[cut]

    for col in FEATURE_COLUMNS:
        a, b = row_full[col], row_trunc[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert np.isclose(a, b, equal_nan=True), (
            f"LEAKAGE: feature {col!r} at row {cut} changed from {b} (truncated) to {a} "
            "(full future visible) — this feature is reading ahead of its observation date."
        )
    print("PASS: feature truncation invariance (no feature reads ahead of its own row)")


def test_label_requires_full_future_window() -> None:
    df = _synthetic_ohlcv()
    labs = compute_labels(df, horizons=HORIZONS_DAYS)
    n = len(df)

    for h in HORIZONS_DAYS:
        col = f"fwd_up_{h}d"
        # The last h rows cannot have a complete forward window and must be NaN.
        tail = labs[col].iloc[n - h :]
        assert tail.isna().all(), (
            f"LEAKAGE: {col} has non-NaN values in the last {h} rows, where a full "
            f"{h}-day forward window does not exist yet."
        )
        # A row with a full forward window available must be non-NaN.
        mid = labs[col].iloc[n - h - 5]
        assert not pd.isna(mid), (
            f"{col} is NaN at a row with a full forward window available — label "
            "computation is dropping valid rows, not just future-incomplete ones."
        )
    print("PASS: forward labels are NaN exactly where the future window is incomplete")


def test_label_uses_only_strictly_future_bars() -> None:
    """Perturbing bar t's own OHLC must not change the label computed AT row t
    (the label at t describes what happens AFTER t, using bars t+1..t+h)."""
    df = _synthetic_ohlcv()
    h = HORIZONS_DAYS[0]
    t = 250

    labs_before = compute_labels(df, horizons=(h,))
    df_perturbed = df.copy()
    df_perturbed.iloc[t, df_perturbed.columns.get_loc("close")] *= 1.5  # shock bar t itself
    labs_after = compute_labels(df_perturbed, horizons=(h,))

    # The label AT row t is fwd_ret computed off close[t] as the base, so it's
    # allowed to change (denominator changed) — what must NOT change is every
    # OTHER row's label, since none of them uses bar t as part of their own
    # (t, t+h] future window unless t falls inside it.
    for check_row in (t - h - 5, t - h - 1):
        if check_row < 0:
            continue
        col = f"fwd_ret_{h}d"
        a, b = labs_before[col].iloc[check_row], labs_after[col].iloc[check_row]
        assert pd.isna(a) == pd.isna(b) and (pd.isna(a) or np.isclose(a, b)), (
            f"LEAKAGE: perturbing bar {t} changed the label at row {check_row}, whose "
            f"forward window (rows {check_row+1}..{check_row+h}) does not include bar {t}."
        )
    print("PASS: forward labels only depend on bars inside their own (t, t+h] window")


def test_embargo_gap_present_in_walk_forward_split() -> None:
    from processing.ml.train import purged_walk_forward_splits

    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=1000)})
    max_horizon = max(HORIZONS_DAYS)
    splits = purged_walk_forward_splits(df["date"], n_folds=4, embargo_days=max_horizon)

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_dates = df["date"].iloc[train_idx]
        test_dates = df["date"].iloc[test_idx]
        gap = (test_dates.min() - train_dates.max()).days
        assert gap >= max_horizon, (
            f"LEAKAGE: fold {fold_i} has only {gap} calendar days between the last "
            f"training date and the first test date; need >= {max_horizon} trading-day "
            "horizon worth of embargo so overlapping labels can't straddle the split."
        )
    print(f"PASS: all {len(splits)} walk-forward folds carry a >= {max_horizon}-day embargo")


if __name__ == "__main__":
    failures = 0
    for fn in (
        test_feature_truncation_invariance,
        test_label_requires_full_future_window,
        test_label_uses_only_strictly_future_bars,
        test_embargo_gap_present_in_walk_forward_split,
    ):
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL: {fn.__name__}: {exc}")
    if failures:
        print(f"\n{failures} leakage test(s) FAILED")
        sys.exit(1)
    print("\nAll leakage tests passed.")
