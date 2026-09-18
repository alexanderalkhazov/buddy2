"""Multi-horizon + regression-target research extension to sector_prediction_model.

Does sector-rotation predictability differ by horizon, and is a direct
excess-return regression target more or less stable than the existing
top-3-of-13 classification target? Both were open questions the production
model (20D classification only) never actually tested — this answers them,
following the same walk-forward, ablation-first discipline as every other
finding this project reports (Phase 1-5, the macro/regime feature ablation,
the sector-ETF and cross-sectional-rank rejections).

This is a RESEARCH script, like phase1-5_*.py — it does NOT replace or
retrain the production model. sector_prediction_model.py's 20D classifier
stays exactly as-is unless a result found here earns its way into
production the same way the regime features did: implemented, ablated,
and only kept if it demonstrably helps.

Reuses the EXACT production feature set (_add_interaction_features,
_feature_set from sector_prediction_model.py) and the EXACT walk-forward
machinery (purged_walk_forward_splits from train.py) — never a separate,
untested pipeline. The multi-horizon labels themselves
(fwd_ret_Xd/fwd_rel_ret_Xd for X in 1/3/5/10/20/40/60) already existed in
every row of the dataset via labels.py/build_sector_dataset — this was a
"labels computed, never used beyond one horizon" gap, the same class of
issue as the breadth-features and regime-features bugs fixed earlier this
session, just for horizons rather than features.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from processing.ml.dataset_v2 import build_dataset_v2
from processing.ml.registry import ARTIFACT_DIR, log_stat_experiment
from processing.ml.sector_dataset import build_sector_dataset
from processing.ml.sector_prediction_model import TOP_K, _add_interaction_features, _feature_set
from processing.ml.train import purged_walk_forward_splits

HORIZONS_TO_TEST = (1, 5, 20, 60)  # short / short-swing / the production horizon / long-swing
REPORT_PATH = ARTIFACT_DIR / "multi_horizon_research_report.json"


def _build_dataset() -> tuple[pd.DataFrame, list[str]]:
    stock_full, _ = build_dataset_v2()
    sector_full = build_sector_dataset(stock_dataset=stock_full)
    sector_full, extra_cols = _add_interaction_features(sector_full)
    return sector_full, extra_cols


def _label_top_k_for_horizon(df: pd.DataFrame, horizon_days: int, k: int = TOP_K) -> pd.DataFrame:
    """Generalizes sector_prediction_model.py:_label_top_k() to any horizon
    already present in the dataset (fwd_ret_Xd for X in labels.py:
    HORIZONS_DAYS), instead of the hardcoded 20D column."""
    col = f"fwd_ret_{horizon_days}d"
    d = df.copy()
    d["rank_that_date"] = d.groupby("date")[col].rank(ascending=False, method="first")
    d["is_top_k"] = np.where(d[col].isna(), np.nan, (d["rank_that_date"] <= k).astype(float))
    return d


def evaluate_classification_horizon(df: pd.DataFrame, feature_cols: list[str], horizon_days: int) -> dict:
    """Same GBM+LR calibrated-ensemble architecture as the production
    model, at a different horizon. embargo_days = horizon_days (in trading
    days, per train.py's convention) since a label spanning (t, t+h] needs
    exactly that much separation from the test window, no more, no less."""
    d = _label_top_k_for_horizon(df, horizon_days)
    data = d.dropna(subset=[*feature_cols, "is_top_k"]).reset_index(drop=True)
    if len(data) < 300:
        return {"horizon_days": horizon_days, "target": "classification", "status": "INCONCLUSIVE", "reason": "too few rows", "n_rows": len(data)}

    splits = purged_walk_forward_splits(data["date"], n_folds=4, embargo_days=horizon_days)
    fold_results = []
    for train_idx, test_idx in splits:
        X_train, X_test = data.loc[train_idx, feature_cols], data.loc[test_idx, feature_cols]
        y_train, y_test = data.loc[train_idx, "is_top_k"].values, data.loc[test_idx, "is_top_k"].values
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue
        scaler = StandardScaler().fit(X_train)
        Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
        hgb = CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=4, max_iter=150), method="sigmoid", cv=3).fit(Xtr, y_train)
        lr = CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight="balanced"), method="sigmoid", cv=3).fit(Xtr, y_train)
        p_ens = (hgb.predict_proba(Xte)[:, 1] + lr.predict_proba(Xte)[:, 1]) / 2
        fold_results.append((roc_auc_score(y_test, p_ens), brier_score_loss(y_test, p_ens)))

    if not fold_results:
        return {"horizon_days": horizon_days, "target": "classification", "status": "INCONCLUSIVE", "reason": "no usable walk-forward folds", "n_rows": len(data)}

    aucs, briers = zip(*fold_results)
    base_rate = float(data["is_top_k"].mean())
    coin_flip_brier = base_rate * (1 - base_rate)
    mean_auc, mean_brier = float(np.mean(aucs)), float(np.mean(briers))
    return {
        "horizon_days": horizon_days, "target": "classification",
        "n_rows": len(data), "n_folds": len(fold_results),
        "mean_auc": round(mean_auc, 4), "mean_brier": round(mean_brier, 4),
        "coin_flip_brier": round(coin_flip_brier, 4),
        "status": "PASS" if mean_auc > 0.55 and mean_brier < coin_flip_brier else "NO_DEMONSTRATED_EDGE",
    }


def evaluate_regression_horizon(df: pd.DataFrame, feature_cols: list[str], horizon_days: int) -> dict:
    """Predicts fwd_rel_ret_{h}d (excess return vs. the SPY benchmark, cross-
    sectional) directly via Ridge — the spec's own recommended linear
    baseline, deliberately not the GBM ensemble, since a regression target
    with this little data is exactly where a heavily regularized linear
    model is the right first thing to try, not the last. Evaluated by
    Spearman rank IC (the standard cross-sectional metric — this project's
    own classification target is itself a rank-threshold, so IC is the
    directly comparable measure of the SAME underlying rank information a
    regression model would need to recover) and MAE, walk-forward, same
    purged/embargoed splits as the classifier so results are apples-to-
    apples comparable."""
    target_col = f"fwd_rel_ret_{horizon_days}d"
    if target_col not in df.columns:
        return {"horizon_days": horizon_days, "target": "regression", "status": "INCONCLUSIVE", "reason": f"{target_col} not in dataset"}
    data = df.dropna(subset=[*feature_cols, target_col]).reset_index(drop=True)
    if len(data) < 300:
        return {"horizon_days": horizon_days, "target": "regression", "status": "INCONCLUSIVE", "reason": "too few rows", "n_rows": len(data)}

    splits = purged_walk_forward_splits(data["date"], n_folds=4, embargo_days=horizon_days)
    fold_results = []
    for train_idx, test_idx in splits:
        if len(train_idx) < 50 or len(test_idx) < 10:
            continue
        X_train, X_test = data.loc[train_idx, feature_cols], data.loc[test_idx, feature_cols]
        y_train, y_test = data.loc[train_idx, target_col].values, data.loc[test_idx, target_col].values
        scaler = StandardScaler().fit(X_train)
        Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
        model = Ridge(alpha=1.0).fit(Xtr, y_train)
        pred = model.predict(Xte)
        ic, _ = spearmanr(pred, y_test)
        if np.isnan(ic):
            continue
        fold_results.append((ic, mean_absolute_error(y_test, pred)))

    if not fold_results:
        return {"horizon_days": horizon_days, "target": "regression", "status": "INCONCLUSIVE", "reason": "no usable walk-forward folds", "n_rows": len(data)}

    ics, maes = zip(*fold_results)
    mean_ic = float(np.mean(ics))
    return {
        "horizon_days": horizon_days, "target": "regression",
        "n_rows": len(data), "n_folds": len(fold_results),
        "mean_rank_ic": round(mean_ic, 4), "mean_mae_pct": round(float(np.mean(maes)), 4),
        # IC > 0.05 is a commonly-cited (not universally agreed) threshold for
        # "weak but real" cross-sectional predictive information in the
        # quant literature — stated as a rule of thumb, not a fitted cutoff.
        "status": "PASS" if mean_ic > 0.05 else "NO_DEMONSTRATED_EDGE",
    }


def run() -> dict:
    df, extra_cols = _build_dataset()
    feature_cols = _feature_set(extra_cols)

    results = {"classification": {}, "regression": {}}
    for h in HORIZONS_TO_TEST:
        results["classification"][h] = evaluate_classification_horizon(df, feature_cols, h)
        results["regression"][h] = evaluate_regression_horizon(df, feature_cols, h)

    note = (
        "Research extension, not a production change: tests whether sector-rotation "
        "predictability differs by horizon (1/5/20/60 trading days) and whether a direct "
        "excess-return regression target (Ridge, Spearman rank IC) is more or less stable "
        "than the production 20D top-3-of-13 classification target (GBM+LR ensemble, AUC/"
        "Brier). Same feature set and walk-forward machinery as the production model. Does "
        "NOT retrain or replace sector_prediction_model.py's 20D classifier."
    )
    report = {"results": results, "note": note}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    log_stat_experiment(
        "multi_horizon_regression_research", "multi_horizon_research", report,
        notes="Tests horizon-dependence and classification-vs-regression stability for sector rotation.",
    )
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, default=str))
