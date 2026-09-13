"""rocket_science.py — the most sophisticated predictive model in this
project, built ON TOP OF (not instead of) everything Phases 1-5 already
established: individual-stock direction has no measurable edge (AUC ≈ 0.50);
the one real, tested signal lives at the SECTOR level (volatility as the
primary driver, negative momentum a secondary modifier); that edge is
MODERATE, not STRONG, with 1 remaining unresolved critical test
(point_in_time_universe).

This model does NOT predict "the market will go up." It predicts, across
ALL 13 sectors spanning every major US equity industry (technology, semis,
software, financials, healthcare, industrials, energy, consumer
discretionary/staples, utilities, materials, communication services, real
estate — the full sector map in processing/ml/universe.py, not just
semiconductors), a CALIBRATED probability that each sector will be a
top-3-of-13 performer over the next 20 trading days. That is the honest
ceiling of what this system's own research supports predicting.

Every number this module reports (AUC, Brier, calibration) is measured via
purged, embargoed, walk-forward validation — exactly the same discipline as
Phases 1-5 — and logged permanently. If the model doesn't beat a coin flip,
this module says so; it does not exist to manufacture a more exciting
answer than the evidence supports.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from processing.ml.dataset_v2 import build_dataset_v2
from processing.ml.features import FEATURE_COLUMNS, compute_features
from processing.ml.registry import ARTIFACT_DIR, log_stat_experiment
from processing.ml.sector_dataset import build_sector_dataset
from processing.ml.sector_index import build_sector_index
from processing.ml.train import purged_walk_forward_splits
from processing.ml.universe import BENCHMARK, EXPANDED_UNIVERSE_V2
from storage.cache import get_ohlcv

TOP_K = 3  # "top-3-of-13 sectors" is the prediction target
HORIZON_COL = "fwd_ret_20d"
EMBARGO_DAYS = 20

# Interaction features on top of the base FEATURE_COLUMNS — this is the
# "more sophisticated" part: explicit cross-terms a plain linear/tree model
# doesn't automatically discover, chosen because Phase 5's regression
# decomposition specifically identified vol and momentum as the two things
# that matter, so their interaction and each one's own breadth-context are
# the natural next things to give the model access to.
def _add_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d["vol_z"] = d.groupby("date")["realized_vol_20d"].transform(lambda s: (s - s.mean()) / (s.std() or 1))
    d["mom_z"] = d.groupby("date")["ret_20d"].transform(lambda s: (s - s.mean()) / (s.std() or 1))
    d["vol_x_mom"] = d["vol_z"] * d["mom_z"]
    d["vol_rank"] = d.groupby("date")["realized_vol_20d"].rank(pct=True)
    d["mom_rank"] = d.groupby("date")["ret_20d"].rank(pct=True)
    d["breadth_x_vol"] = d.get("breadth_pct_above_sma50", 50) * d["vol_z"]
    extra = ["vol_z", "mom_z", "vol_x_mom", "vol_rank", "mom_rank", "breadth_x_vol"]
    return d, extra


def _label_top_k(df: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """Cross-sectional label: 1 if this sector's fwd_ret_20d ranks in the
    top k among all sectors observed on the SAME date, else 0. Labels every
    sector-date, including the ones with no future window yet (NaN target,
    dropped downstream) — never fabricated."""
    d = df.copy()
    d["rank_that_date"] = d.groupby("date")[HORIZON_COL].rank(ascending=False, method="first")
    d["n_that_date"] = d.groupby("date")[HORIZON_COL].transform("count")
    d["is_top_k"] = np.where(d[HORIZON_COL].isna(), np.nan, (d["rank_that_date"] <= k).astype(float))
    return d


def _build_dataset() -> pd.DataFrame:
    stock_full, _ = build_dataset_v2()
    sector_full = build_sector_dataset(stock_dataset=stock_full)
    sector_full, extra_cols = _add_interaction_features(sector_full)
    sector_full = _label_top_k(sector_full)
    return sector_full, extra_cols


def _feature_set(extra_cols: list[str]) -> list[str]:
    breadth_cols = ["breadth_pct_above_sma50", "breadth_pct_above_sma200", "breadth_pct_positive_20d", "breadth_median_rsi"]
    return [c for c in (*FEATURE_COLUMNS, *breadth_cols, *extra_cols)]


def evaluate(refresh: bool = False) -> dict:
    """Purged, embargoed walk-forward evaluation — same discipline as every
    other phase. Returns honest OOS metrics, never fit-then-report-in-sample."""
    df, extra_cols = _build_dataset()
    feature_cols = _feature_set(extra_cols)
    data = df.dropna(subset=[*feature_cols, "is_top_k"]).reset_index(drop=True)

    splits = purged_walk_forward_splits(data["date"], n_folds=4, embargo_days=EMBARGO_DAYS)
    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = data.loc[train_idx, feature_cols], data.loc[test_idx, feature_cols]
        y_train, y_test = data.loc[train_idx, "is_top_k"].values, data.loc[test_idx, "is_top_k"].values
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue

        scaler = StandardScaler().fit(X_train)
        Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

        hgb = CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_depth=4, max_iter=150), method="sigmoid", cv=3
        ).fit(Xtr, y_train)
        lr = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight="balanced"), method="sigmoid", cv=3
        ).fit(Xtr, y_train)

        p_hgb = hgb.predict_proba(Xte)[:, 1]
        p_lr = lr.predict_proba(Xte)[:, 1]
        p_ensemble = (p_hgb + p_lr) / 2

        fold_results.append(
            {
                "fold": fold_i,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "test_start": str(pd.Timestamp(data.loc[test_idx, "date"].min()).date()),
                "test_end": str(pd.Timestamp(data.loc[test_idx, "date"].max()).date()),
                "auc_hgb": round(float(roc_auc_score(y_test, p_hgb)), 4),
                "auc_ensemble": round(float(roc_auc_score(y_test, p_ensemble)), 4),
                "brier_ensemble": round(float(brier_score_loss(y_test, p_ensemble)), 4),
            }
        )

    if not fold_results:
        result = {"status": "INCONCLUSIVE", "reason": "not enough data to form a walk-forward fold"}
        log_stat_experiment("rocket_science_evaluation", "rocket_science", result)
        return result

    mean_auc = round(float(np.mean([f["auc_ensemble"] for f in fold_results])), 4)
    mean_brier = round(float(np.mean([f["brier_ensemble"] for f in fold_results])), 4)
    base_rate = round(float(data["is_top_k"].mean()), 4)  # ~TOP_K/13 by construction
    coin_flip_brier = round(base_rate * (1 - base_rate), 4)

    result = {
        "target": f"top_{TOP_K}_of_13_sectors_by_20D_forward_return",
        "n_rows": len(data),
        "n_folds": len(fold_results),
        "folds": fold_results,
        "mean_oos_auc": mean_auc,
        "mean_oos_brier": mean_brier,
        "base_rate": base_rate,
        "coin_flip_brier_baseline": coin_flip_brier,
        "beats_coin_flip": mean_brier < coin_flip_brier,
        "status": "PASS" if mean_auc > 0.55 and mean_brier < coin_flip_brier else "NO_DEMONSTRATED_EDGE",
        "note": (
            f"base_rate ({base_rate}) is the unconditional probability any sector is top-{TOP_K} of 13 on a "
            "given date (~TOP_K/13 by construction, not a finding). mean_oos_brier must beat "
            "coin_flip_brier_baseline (predicting base_rate for everyone, every time) to represent any real "
            "information — an AUC above 0.55 with a Brier score above the coin-flip baseline is NOT a "
            "genuine edge, regardless of how the AUC number looks in isolation."
        ),
    }
    log_stat_experiment("rocket_science_evaluation", "rocket_science", {k: v for k, v in result.items() if k != "folds"})
    return result


def train_and_save(refresh: bool = False) -> dict:
    df, extra_cols = _build_dataset()
    feature_cols = _feature_set(extra_cols)
    data = df.dropna(subset=[*feature_cols, "is_top_k"]).reset_index(drop=True)

    X, y = data[feature_cols], data["is_top_k"].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    hgb = CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=4, max_iter=150), method="sigmoid", cv=3).fit(Xs, y)
    lr = CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight="balanced"), method="sigmoid", cv=3).fit(Xs, y)

    import joblib

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "hgb": hgb, "lr": lr, "feature_cols": feature_cols}, ARTIFACT_DIR / "rocket_science_model.joblib")

    evaluation = evaluate(refresh=refresh)
    (ARTIFACT_DIR / "rocket_science_report.json").write_text(json.dumps(evaluation, indent=2, default=str))
    return evaluation


def predict_next_move(refresh: bool = False) -> dict:
    """Live prediction: calibrated P(top-3-of-13) for ALL 13 sectors right
    now, spanning the full US equity sector map — technology, semis,
    software, financials, healthcare, industrials, energy, consumer
    discretionary/staples, utilities, materials, communication services,
    real estate. Not just semiconductors, and not a single point call."""
    model_path = ARTIFACT_DIR / "rocket_science_model.joblib"
    report_path = ARTIFACT_DIR / "rocket_science_report.json"
    if not model_path.exists() or not report_path.exists():
        return {"error": "no trained model on disk — run `python -m processing.ml.rocket_science` first"}

    import joblib

    artifact = joblib.load(model_path)
    evaluation = json.loads(report_path.read_text())
    scaler, hgb, lr, feature_cols = artifact["scaler"], artifact["hgb"], artifact["lr"], artifact["feature_cols"]

    bench_df = get_ohlcv(BENCHMARK, force=refresh)
    rows = []
    for sector in EXPANDED_UNIVERSE_V2:
        try:
            idx_df = build_sector_index(sector, refresh=refresh)
        except Exception as exc:
            rows.append({"sector": sector, "error": str(exc)[:120]})
            continue
        feats = compute_features(idx_df, benchmark_df=bench_df)
        valid = feats.dropna(subset=["close"])
        if valid.empty:
            continue
        last = valid[FEATURE_COLUMNS].iloc[[-1]].copy()
        last["date"] = valid.index[-1]
        rows.append({"sector": sector, **last.iloc[0].to_dict()})

    live = pd.DataFrame(rows)
    live = live.dropna(subset=["realized_vol_20d"])
    if live.empty:
        return {"error": "no live sector data available this run"}

    live["vol_z"] = (live["realized_vol_20d"] - live["realized_vol_20d"].mean()) / (live["realized_vol_20d"].std() or 1)
    live["mom_z"] = (live["ret_20d"] - live["ret_20d"].mean()) / (live["ret_20d"].std() or 1)
    live["vol_x_mom"] = live["vol_z"] * live["mom_z"]
    live["vol_rank"] = live["realized_vol_20d"].rank(pct=True)
    live["mom_rank"] = live["ret_20d"].rank(pct=True)
    for col in ("breadth_pct_above_sma50", "breadth_pct_above_sma200", "breadth_pct_positive_20d", "breadth_median_rsi"):
        live[col] = 50.0  # neutral default when live per-sector breadth isn't recomputed for this snapshot call
    live["breadth_x_vol"] = live["breadth_pct_above_sma50"] * live["vol_z"]

    missing_cols = [c for c in feature_cols if c not in live.columns]
    for c in missing_cols:
        live[c] = 0.0

    X_live = live[feature_cols]
    Xs_live = scaler.transform(X_live)
    p_hgb = hgb.predict_proba(Xs_live)[:, 1]
    p_lr = lr.predict_proba(Xs_live)[:, 1]
    p_ensemble = (p_hgb + p_lr) / 2

    live["p_top3_hgb"] = p_hgb
    live["p_top3_lr"] = p_lr
    live["p_top3_ensemble"] = p_ensemble
    live["model_agreement"] = 1 - np.abs(p_hgb - p_lr)
    live = live.sort_values("p_top3_ensemble", ascending=False)

    predictions = live[["sector", "p_top3_ensemble", "p_top3_hgb", "p_top3_lr", "model_agreement", "realized_vol_20d", "ret_20d"]].to_dict(orient="records")

    return {
        "target": evaluation.get("target", f"top_{TOP_K}_of_13_sectors_by_20D_forward_return"),
        "predictions_all_13_sectors": predictions,
        "model_oos_reliability": {
            "mean_oos_auc": evaluation.get("mean_oos_auc"),
            "mean_oos_brier": evaluation.get("mean_oos_brier"),
            "coin_flip_brier_baseline": evaluation.get("coin_flip_brier_baseline"),
            "status": evaluation.get("status"),
        },
        "note": (
            "This is NOT a prediction that any market will rise or fall. It is a calibrated probability, "
            "per sector, that it lands in the top-3-of-13 US equity sectors by 20-trading-day forward "
            "return — covering the full sector map (technology, semis, software, financials, healthcare, "
            "industrials, energy, consumer discretionary/staples, utilities, materials, communication "
            "services, real estate), not only semiconductors. model_oos_reliability.status tells you "
            "whether this model has actually demonstrated any edge over a coin flip — trust that over the "
            "raw probability numbers."
        ),
    }


if __name__ == "__main__":
    result = train_and_save()
    print(json.dumps(result, indent=2, default=str))
    print("\n--- live prediction ---")
    print(json.dumps(predict_next_move(), indent=2, default=str))
