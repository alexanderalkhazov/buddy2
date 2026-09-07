"""Live prediction: calibrated P(up 20D) from the saved ensemble + a historical
analog engine (k-NN over the same standardized feature space). This is the
ONLY module that talks to ui/report.py — report.py never touches sklearn or
the model artifact directly.

Reliability gating: the model's OOS Brier/AUC from oos_report.json are the
system's own self-assessment of whether ANY of this is worth reading, not a
disclaimer bolted on after the fact. If OOS AUC is not meaningfully above 0.5,
`reliability` is LOW and `recommended_gate` is "NO TRADE — model has no
demonstrated OOS edge," regardless of what the point prediction says. A
probability estimate from a model with no measured edge is not information.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from processing.ml.features import FEATURE_COLUMNS, compute_features
from processing.ml.train import ARTIFACT_DIR
from processing.ml.universe import BENCHMARK
from storage.cache import get_ohlcv

_MODEL_PATH = ARTIFACT_DIR / "direction_model.joblib"
_OOS_PATH = ARTIFACT_DIR / "oos_report.json"

K_NEIGHBORS = 30
MIN_ANALOGS_FOR_CONFIDENCE = 15
AUC_EDGE_THRESHOLD = 0.55  # below this, OOS evidence doesn't distinguish the model from a coin flip


def model_available() -> bool:
    return _MODEL_PATH.exists() and _OOS_PATH.exists()


def _load():
    import joblib

    artifact = joblib.load(_MODEL_PATH)
    oos = json.loads(_OOS_PATH.read_text())
    return artifact, oos


def predict(ticker: str, refresh: bool = False) -> dict:
    if not model_available():
        return {
            "error": "no trained model on disk — run `python -m processing.ml.train` first",
            "reliability": "UNAVAILABLE",
        }

    artifact, oos = _load()
    scaler, logreg, hgb = artifact["scaler"], artifact["logreg"], artifact["hgb"]
    analog_df = artifact["analog_dataset"]

    df = get_ohlcv(ticker, force=refresh)
    bench_df = get_ohlcv(BENCHMARK, force=refresh)
    feats = compute_features(df, benchmark_df=bench_df)
    last_row = feats[FEATURE_COLUMNS].iloc[-1]

    if last_row.isna().any():
        missing = [c for c in FEATURE_COLUMNS if pd.isna(last_row[c])]
        return {
            "ticker": ticker.upper(),
            "error": f"insufficient price history to compute: {missing}",
            "reliability": "UNAVAILABLE",
        }

    x = last_row.to_frame().T[FEATURE_COLUMNS]
    xs = scaler.transform(x)

    p_logreg = float(logreg.predict_proba(xs)[0, 1])
    p_hgb = float(hgb.predict_proba(xs)[0, 1])
    p_ensemble = (p_logreg + p_hgb) / 2
    model_agreement = round(1 - abs(p_logreg - p_hgb), 4)  # 1.0 = perfect agreement, 0.0 = maximal disagreement

    # Historical analog engine: k nearest neighbors in the SAME standardized
    # feature space the model trained on, excluding this ticker's own rows so a
    # ticker can't be "similar to itself."
    analog_X = scaler.transform(analog_df[FEATURE_COLUMNS])
    dists = np.linalg.norm(analog_X - xs, axis=1)
    candidate_mask = analog_df["ticker"].values != ticker.upper()
    order = np.argsort(np.where(candidate_mask, dists, np.inf))[:K_NEIGHBORS]
    neighbors = analog_df.iloc[order]
    fwd_rets = neighbors["fwd_ret_20d"].dropna().values

    analogs = {
        "n": int(len(fwd_rets)),
        "median_fwd_ret_20d_pct": round(float(np.median(fwd_rets)), 2) if len(fwd_rets) else None,
        "p10_fwd_ret_20d_pct": round(float(np.percentile(fwd_rets, 10)), 2) if len(fwd_rets) else None,
        "p25_fwd_ret_20d_pct": round(float(np.percentile(fwd_rets, 25)), 2) if len(fwd_rets) else None,
        "p75_fwd_ret_20d_pct": round(float(np.percentile(fwd_rets, 75)), 2) if len(fwd_rets) else None,
        "p90_fwd_ret_20d_pct": round(float(np.percentile(fwd_rets, 90)), 2) if len(fwd_rets) else None,
        "win_rate_pct": round(float((fwd_rets > 0).mean() * 100), 1) if len(fwd_rets) else None,
        "note": (
            f"The {K_NEIGHBORS} historical ticker-dates (across the training universe, excluding "
            f"{ticker.upper()} itself) whose standardized feature vector was closest (Euclidean) to "
            "today's — a nearest-neighbor lookup, not a model. Read this as 'what happened next in "
            "situations that looked similar on these price/technical features,' not a prediction."
        ),
    }

    oos_auc = max(oos.get("mean_auc_logreg", 0.5), oos.get("mean_auc_hgb", 0.5))
    has_edge = oos_auc >= AUC_EDGE_THRESHOLD
    enough_analogs = analogs["n"] >= MIN_ANALOGS_FOR_CONFIDENCE

    if not has_edge:
        reliability = "LOW"
        gate = "NO TRADE — model has no demonstrated out-of-sample edge (OOS AUC %.3f, threshold %.2f). See oos_report.json." % (oos_auc, AUC_EDGE_THRESHOLD)
    elif model_agreement < 0.85 or not enough_analogs:
        reliability = "MEDIUM"
        gate = "CAUTION — low model agreement or thin analog sample; treat as a weak signal, not a trade trigger."
    else:
        reliability = "MEDIUM-HIGH (relative to this system's other components — still not a validated trading edge)"
        gate = "Model shows some OOS signal; still require independent confirmation (scorecard/thesis/risk) before any trade."

    return {
        "ticker": ticker.upper(),
        "target": "P(return over next 20 trading days > 0)",
        "p_up_20d_logreg": round(p_logreg, 4),
        "p_up_20d_hgb": round(p_hgb, 4),
        "p_up_20d_ensemble": round(p_ensemble, 4),
        "model_agreement": model_agreement,
        "historical_analogs": analogs,
        "oos_reliability": {
            "mean_oos_auc": round(oos_auc, 4),
            "mean_oos_brier": min(oos.get("mean_brier_logreg", 0.25), oos.get("mean_brier_hgb", 0.25)),
            "n_training_rows": oos.get("n_rows"),
            "n_training_tickers": oos.get("n_tickers"),
            "training_date_range": oos.get("date_range"),
            "n_folds": oos.get("n_folds"),
        },
        "reliability": reliability,
        "recommended_gate": gate,
        "note": (
            "This is a SEPARATE, experimental prediction engine layered on top of the deterministic "
            "scorecard — it does not replace scorecard.overall/decision (now the BASELINE SCORE), and "
            "the two are not guaranteed to agree. Built only from price/technical features: this system's "
            "data source (yfinance) does not expose point-in-time historical fundamentals, analyst "
            "estimates, earnings, options, or news, so none of those are in the training features — "
            "adding them without point-in-time data would leak the future into the training set, which "
            "is worse than not having the feature at all. Trust `reliability`/`recommended_gate` over "
            "the raw probability number: a stated 68% from a model with no measured OOS edge is not "
            "68% real information."
        ),
    }
