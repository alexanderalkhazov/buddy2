"""Append-only experiment registry — every experiment run through
`run_experiment()` is logged here and NEVER overwritten, so Phase 2's claims
("relative target beat absolute target by X") are always re-derivable from a
permanent record, not just this session's printed output.

Storage: one JSON object per line in storage/models/experiments.jsonl. Not a
database — this repo already treats DuckDB as the OHLCV/news cache, and a
plain append-only log is the right tool for "never overwrite, append forever."
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from processing.ml.features import FEATURE_COLUMNS
from processing.ml.train import ARTIFACT_DIR, run_walk_forward

REGISTRY_PATH = ARTIFACT_DIR / "experiments.jsonl"


def _decile_spread(oos_predictions: pd.DataFrame) -> dict:
    """Cross-sectional top-decile vs bottom-decile spread: on each date with
    enough tickers present, rank by predicted probability and compare the
    realized forward return of the top 10% vs bottom 10%. This is answer #18
    from the brief (top-K selection) computed directly off OOS predictions —
    never off in-sample scores."""
    if oos_predictions.empty or "fwd_ret" not in oos_predictions.columns:
        return {"available": False, "reason": "no OOS predictions or no return column"}

    df = oos_predictions.dropna(subset=["p_ensemble", "fwd_ret"])
    spreads, dates_used = [], 0
    for _, group in df.groupby("date"):
        if len(group) < 10:  # need a real cross-section to rank within
            continue
        g = group.sort_values("p_ensemble")
        n = len(g)
        bottom = g.iloc[: max(1, n // 10)]
        top = g.iloc[-max(1, n // 10):]
        spreads.append(top["fwd_ret"].mean() - bottom["fwd_ret"].mean())
        dates_used += 1

    if not spreads:
        return {"available": False, "reason": "no date had >=10 tickers to rank cross-sectionally"}

    spreads = np.array(spreads)
    return {
        "available": True,
        "n_dates": dates_used,
        "mean_top_minus_bottom_decile_return_pct": round(float(spreads.mean()), 3),
        "median_top_minus_bottom_decile_return_pct": round(float(np.median(spreads)), 3),
        "pct_dates_positive_spread": round(float((spreads > 0).mean() * 100), 1),
        "spread_sharpe_like": round(float(spreads.mean() / spreads.std()), 3) if spreads.std() > 0 else None,
        "note": (
            "Per OOS-prediction date with >=10 tickers ranked by predicted probability: mean forward "
            "return of the top decile minus the bottom decile. spread_sharpe_like is mean/std of the "
            "per-date spread series — NOT annualized, NOT a real backtest Sharpe (no costs, no position "
            "sizing, overlapping dates since predictions are stride-5-sampled) — a rough signal-vs-noise "
            "ratio only, to be read directionally, not quoted as a strategy Sharpe."
        ),
    }


def run_experiment(
    dataset: pd.DataFrame,
    target: str,
    label: str,
    feature_columns: list[str] = FEATURE_COLUMNS,
    universe_desc: str = "default_universe",
    notes: str = "",
    return_column: str | None = None,
) -> dict:
    """Runs the standard purged/embargoed walk-forward evaluation for one
    (feature set, target) combination, logs a permanent record, and returns it.
    `label` is a short human name for the experiment (e.g. "baseline_20d",
    "relative_target_20d") — experiment_id is a separate, always-unique UUID."""
    fold_results, oos_preds = run_walk_forward(
        dataset, target=target, feature_columns=feature_columns, return_predictions=True, return_column=return_column
    )

    if not fold_results:
        record = {
            "experiment_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "target": target,
            "status": "FAILED — not enough data for any walk-forward fold",
        }
        _append(record)
        return record

    aucs_hgb = [r.auc_hgb for r in fold_results if r.auc_hgb == r.auc_hgb]
    aucs_lr = [r.auc_logreg for r in fold_results if r.auc_logreg == r.auc_logreg]
    briers_hgb = [r.brier_hgb for r in fold_results]
    briers_lr = [r.brier_logreg for r in fold_results]
    loglosses_hgb = [r.logloss_hgb for r in fold_results]

    # Rank IC: Spearman correlation between the ensemble's predicted probability
    # and the REAL-VALUED forward return, pooled across all OOS rows. A proxy
    # for per-date cross-sectional IC (not a strict per-date average) — noted
    # honestly as such rather than dressed up as textbook rank IC.
    rank_ic = None
    if "fwd_ret" in oos_preds.columns and len(oos_preds.dropna(subset=["fwd_ret"])) > 30:
        valid = oos_preds.dropna(subset=["p_ensemble", "fwd_ret"])
        rank_ic = round(float(spearmanr(valid["p_ensemble"], valid["fwd_ret"]).statistic), 4)

    calib_err_hgb = None
    buckets = [b for r in fold_results for b in r.calibration_hgb if b["n"] >= 20]
    if buckets:
        calib_err_hgb = round(float(np.mean([abs(b["predicted_avg"] - b["actual_rate"]) for b in buckets])), 4)

    decile = _decile_spread(oos_preds)

    record = {
        "experiment_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "notes": notes,
        "universe": universe_desc,
        "target": target,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "n_rows": int(len(dataset)),
        "n_folds": len(fold_results),
        "date_range": [str(pd.Timestamp(dataset["date"].min()).date()), str(pd.Timestamp(dataset["date"].max()).date())],
        "model": "logreg+hgb (calibrated, sigmoid, ensemble-averaged)",
        "auc_hgb": round(float(np.mean(aucs_hgb)), 4) if aucs_hgb else None,
        "auc_logreg": round(float(np.mean(aucs_lr)), 4) if aucs_lr else None,
        "brier_hgb": round(float(np.mean(briers_hgb)), 4),
        "brier_logreg": round(float(np.mean(briers_lr)), 4),
        "logloss_hgb": round(float(np.mean(loglosses_hgb)), 4),
        "calibration_error_hgb": calib_err_hgb,
        "rank_ic": rank_ic,
        "decile_spread": decile,
        "status": "OK",
    }
    _append(record)
    return record


def _append(record: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def log_stat_experiment(label: str, phase: str, result: dict, notes: str = "") -> dict:
    """For statistical/diagnostic tests that aren't a walk-forward model run
    (IC scans, placebo tests, interaction grids, adversarial permutations,
    portfolio spread tests) — same append-only, never-overwritten log, so
    Phase 3's non-ML findings are just as permanently recorded as Phase 1/2's
    model runs."""
    record = {
        "experiment_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "label": label,
        "notes": notes,
        "kind": "statistical",
        "result": result,
    }
    _append(record)
    return record


def load_all() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    return [json.loads(line) for line in REGISTRY_PATH.read_text().splitlines() if line.strip()]
