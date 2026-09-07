"""Walk-forward training + calibration for the direction model.

Two models (a genuine small ensemble, not a single black box):
  - LogisticRegression (linear baseline — if the boosted model can't beat this
    out-of-sample, the boosted model isn't earning its complexity)
  - HistGradientBoostingClassifier (sklearn's built-in boosting; no extra
    dependency needed beyond what's already installed)

Both are probability-CALIBRATED (sigmoid/Platt, fit only on each fold's own
training data) rather than trusted raw, because raw tree/logit outputs are
scores, not real probabilities, and the whole point of this system is that a
stated 70% must actually resolve ~70% of the time.

Validation is purged, embargoed, walk-forward (chronological folds, with a gap
of `embargo_days` between the end of a fold's training data and the start of
its test data) — never a random shuffle-split, which would let overlapping
labels from adjacent dates leak between train and test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from processing.ml.dataset import build_dataset
from processing.ml.features import FEATURE_COLUMNS

ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / "models"
TARGET_COLUMN = "fwd_up_20d"  # primary direction target trained by default
N_FOLDS = 4


def purged_walk_forward_splits(dates: pd.Series, n_folds: int = N_FOLDS, embargo_days: int = 20):
    """Chronological expanding-window folds with an embargo gap. Returns a list
    of (train_positional_indices, test_positional_indices) over `dates`, sorted
    ascending. embargo_days is CALENDAR days (dates here are trading days, so
    this is conservative — a 20-trading-day horizon spans ~28 calendar days,
    and using the raw day count as the gap threshold undercounts weekends,
    erring toward a slightly wider embargo than strictly necessary, never
    narrower)."""
    order = np.argsort(dates.values)
    sorted_dates = dates.values[order]
    n = len(sorted_dates)
    fold_edges = np.linspace(0, n, n_folds + 1, dtype=int)

    splits = []
    for i in range(1, n_folds):
        test_start_pos, test_end_pos = fold_edges[i], fold_edges[i + 1]
        if test_start_pos >= test_end_pos:
            continue
        test_start_date = sorted_dates[test_start_pos]
        embargo_cutoff = pd.Timestamp(test_start_date) - pd.Timedelta(days=embargo_days)

        train_mask = sorted_dates < np.datetime64(embargo_cutoff)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_start_pos:test_end_pos] = True

        train_idx = order[train_mask]
        test_idx = order[test_mask]
        if len(train_idx) < 200 or len(test_idx) < 20:
            continue
        splits.append((train_idx, test_idx))
    return splits


def _calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_buckets: int = 5) -> list[dict]:
    edges = np.linspace(0, 1, n_buckets + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        n = int(mask.sum())
        rows.append(
            {
                "predicted_range": f"{lo:.0%}-{hi:.0%}",
                "n": n,
                "predicted_avg": round(float(y_prob[mask].mean()), 3) if n else None,
                "actual_rate": round(float(y_true[mask].mean()), 3) if n else None,
            }
        )
    return rows


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    test_start: str
    test_end: str
    brier_logreg: float
    brier_hgb: float
    logloss_logreg: float
    logloss_hgb: float
    auc_logreg: float
    auc_hgb: float
    calibration_logreg: list = field(default_factory=list)
    calibration_hgb: list = field(default_factory=list)


def _embargo_for(target: str) -> int:
    """Embargo must be >= the label horizon baked into `target`'s name (e.g.
    fwd_up_40d -> 40) so overlapping-label rows never straddle train/test."""
    import re

    m = re.search(r"_(\d+)d$", target)
    return int(m.group(1)) if m else 20


def run_walk_forward(
    dataset: pd.DataFrame,
    target: str = TARGET_COLUMN,
    feature_columns: list[str] = FEATURE_COLUMNS,
    embargo_days: int | None = None,
    return_predictions: bool = False,
    return_column: str | None = None,
):
    """Returns fold_results, and (if return_predictions) a DataFrame of every
    OOS prediction made across all folds (ticker, date, y_true, p_hgb, p_logreg,
    plus the raw return column of the same horizon if present) — this is the
    single source of every downstream analysis (rank IC, decile spreads,
    calibration-by-bucket) so nothing is ever computed off an in-sample score."""
    df = dataset.dropna(subset=[*feature_columns, target]).reset_index(drop=True)
    embargo_days = embargo_days if embargo_days is not None else _embargo_for(target)
    splits = purged_walk_forward_splits(df["date"], embargo_days=embargo_days)
    results = []
    pred_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = df.loc[train_idx, feature_columns], df.loc[test_idx, feature_columns]
        y_train, y_test = df.loc[train_idx, target].values, df.loc[test_idx, target].values

        scaler = StandardScaler().fit(X_train)
        Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

        logreg = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight="balanced"), method="sigmoid", cv=3
        ).fit(Xtr, y_train)
        hgb = CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_depth=4, max_iter=150), method="sigmoid", cv=3
        ).fit(Xtr, y_train)

        p_logreg = logreg.predict_proba(Xte)[:, 1]
        p_hgb = hgb.predict_proba(Xte)[:, 1]

        results.append(
            FoldResult(
                fold=fold_i,
                n_train=len(train_idx),
                n_test=len(test_idx),
                test_start=str(pd.Timestamp(df.loc[test_idx, "date"].min()).date()),
                test_end=str(pd.Timestamp(df.loc[test_idx, "date"].max()).date()),
                brier_logreg=round(float(brier_score_loss(y_test, p_logreg)), 4),
                brier_hgb=round(float(brier_score_loss(y_test, p_hgb)), 4),
                logloss_logreg=round(float(log_loss(y_test, p_logreg, labels=[0, 1])), 4),
                logloss_hgb=round(float(log_loss(y_test, p_hgb, labels=[0, 1])), 4),
                auc_logreg=round(float(roc_auc_score(y_test, p_logreg)), 4) if len(set(y_test)) > 1 else float("nan"),
                auc_hgb=round(float(roc_auc_score(y_test, p_hgb)), 4) if len(set(y_test)) > 1 else float("nan"),
                calibration_logreg=_calibration_table(y_test, p_logreg),
                calibration_hgb=_calibration_table(y_test, p_hgb),
            )
        )

        if return_predictions:
            chunk = pd.DataFrame(
                {
                    "fold": fold_i,
                    "ticker": df.loc[test_idx, "ticker"].values,
                    "date": df.loc[test_idx, "date"].values,
                    "y_true": y_test,
                    "p_logreg": p_logreg,
                    "p_hgb": p_hgb,
                    "p_ensemble": (p_logreg + p_hgb) / 2,
                }
            )
            if return_column:
                ret_col = return_column
            elif target.startswith("fwd_up_rel_"):
                ret_col = target.replace("fwd_up_rel_", "fwd_rel_ret_")
            else:
                ret_col = target.replace("fwd_up_", "fwd_ret_")
            if ret_col in df.columns:
                chunk["fwd_ret"] = df.loc[test_idx, ret_col].values
            pred_rows.append(chunk)

    if return_predictions:
        oos_predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
        return results, oos_predictions
    return results


def eval_explicit_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    feature_columns: list[str] = FEATURE_COLUMNS,
    embargo_days: int | None = None,
    return_column: str | None = None,
) -> dict | None:
    """One explicit train-on-A / test-on-B evaluation — used for the golden
    holdout, where the split boundary is fixed by the caller (dev vs holdout),
    not derived from quantiles of a combined frame. Applies the same embargo:
    any training row within embargo_days of the test period's start is dropped,
    so a training label whose forward window overlaps the holdout is excluded."""
    embargo_days = embargo_days if embargo_days is not None else _embargo_for(target)
    train_df = train_df.dropna(subset=[*feature_columns, target])
    test_df = test_df.dropna(subset=[*feature_columns, target])
    if train_df.empty or test_df.empty:
        return None

    embargo_cutoff = pd.Timestamp(test_df["date"].min()) - pd.Timedelta(days=embargo_days)
    train_df = train_df[train_df["date"] < embargo_cutoff]
    if len(train_df) < 200 or len(test_df) < 20:
        return None

    X_train, X_test = train_df[feature_columns], test_df[feature_columns]
    y_train, y_test = train_df[target].values, test_df[target].values

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    hgb = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_depth=4, max_iter=150), method="sigmoid", cv=3
    ).fit(Xtr, y_train)
    p_hgb = hgb.predict_proba(Xte)[:, 1]

    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_end": str(pd.Timestamp(train_df["date"].max()).date()),
        "test_start": str(pd.Timestamp(test_df["date"].min()).date()),
        "test_end": str(pd.Timestamp(test_df["date"].max()).date()),
        "auc_hgb": round(float(roc_auc_score(y_test, p_hgb)), 4) if len(set(y_test)) > 1 else None,
        "brier_hgb": round(float(brier_score_loss(y_test, p_hgb)), 4),
        "logloss_hgb": round(float(log_loss(y_test, p_hgb, labels=[0, 1])), 4),
        "calibration_hgb": _calibration_table(y_test, p_hgb),
    }


def train_and_save(universe: list[str] | None = None, refresh: bool = False) -> dict:
    """Builds the dataset, runs walk-forward OOS evaluation for reporting, then
    fits a FINAL model on all available data (for live prediction only — never
    used to produce the OOS numbers reported alongside it) and saves it plus the
    training dataset (needed for the analog engine) and metadata to disk."""
    import joblib

    print("Building dataset...")
    dataset = build_dataset(universe=universe, refresh=refresh)
    print(f"  {len(dataset)} rows, {dataset['ticker'].nunique()} tickers")

    print("Running purged walk-forward evaluation...")
    fold_results = run_walk_forward(dataset)
    if not fold_results:
        raise RuntimeError("not enough data to form any walk-forward fold — expand the universe or date range")

    df = dataset.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN]).reset_index(drop=True)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    print("Fitting final models on full dataset (for live prediction only)...")
    final_logreg = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, class_weight="balanced"), method="sigmoid", cv=3
    ).fit(Xs, y)
    final_hgb = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_depth=4, max_iter=150), method="sigmoid", cv=3
    ).fit(Xs, y)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "logreg": final_logreg,
            "hgb": final_hgb,
            "feature_columns": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "analog_dataset": df,  # standardized at predict time using the same scaler
        },
        ARTIFACT_DIR / "direction_model.joblib",
    )

    oos_summary = {
        "target": TARGET_COLUMN,
        "n_rows": len(dataset),
        "n_tickers": int(dataset["ticker"].nunique()),
        "date_range": [str(dataset["date"].min().date()), str(dataset["date"].max().date())],
        "n_folds": len(fold_results),
        "folds": [vars(r) for r in fold_results],
        "mean_brier_logreg": round(float(np.mean([r.brier_logreg for r in fold_results])), 4),
        "mean_brier_hgb": round(float(np.mean([r.brier_hgb for r in fold_results])), 4),
        "mean_auc_logreg": round(float(np.nanmean([r.auc_logreg for r in fold_results])), 4),
        "mean_auc_hgb": round(float(np.nanmean([r.auc_hgb for r in fold_results])), 4),
        "note": (
            "brier/auc/calibration above are OUT-OF-SAMPLE, computed per walk-forward fold with a "
            "20-trading-day embargo between train and test. The saved model used for live prediction "
            "is fit on ALL rows (including these test folds) and is NOT the model that produced these "
            "numbers — that's standard practice (use every available observation for the deployed "
            "model) but means these OOS numbers describe the METHOD's historical reliability, not a "
            "guarantee about this exact saved model's future accuracy. Brier score of 0.25 is what a "
            "coin-flip (always predicting 50%) scores on a balanced target — meaningfully below 0.25 "
            "is where a real edge would show up; AUC of 0.5 is no better than random."
        ),
    }
    (ARTIFACT_DIR / "oos_report.json").write_text(json.dumps(oos_summary, indent=2, default=str))
    print(f"Saved model + OOS report to {ARTIFACT_DIR}")
    return oos_summary


if __name__ == "__main__":
    summary = train_and_save()
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))
