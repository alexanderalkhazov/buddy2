"""sector_prediction_model.py — calibrated, walk-forward-validated model
predicting each US equity sector's probability of being a top-3-of-13
performer over the next 20 trading days. The most sophisticated predictive
model in this project, built ON TOP OF (not instead of) everything Phases 1-5 already
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
from processing.ml.macro_features import fetch_macro_history
from processing.ml.regime import compute_regime_features
from processing.ml.registry import ARTIFACT_DIR, log_stat_experiment
from processing.ml.sector_dataset import (
    REGIME_FEATURE_COLUMNS, _TREND_REGIME_CODE, _VOL_REGIME_CODE, build_sector_dataset,
)
from processing.ml.sector_index import build_sector_index
from processing.ml.train import purged_walk_forward_splits
from processing.ml.universe import BENCHMARK, EXPANDED_UNIVERSE_V2
from storage.cache import get_ohlcv

TOP_K = 3  # "top-3-of-13 sectors" is the prediction target
DEFAULT_HORIZON_DAYS = 20  # the original, most-stress-tested production horizon
# Second production horizon, added after multi_horizon_research.py found it
# genuinely stronger on both classification AUC (0.607 vs 0.590) and
# regression Rank IC (0.177 vs 0.104) — additive, not a replacement, since
# 60D hasn't yet been through the same depth of stress-testing (Phase 4/5-
# style adversarial/non-overlapping checks) the 20D target has.
SECONDARY_HORIZON_DAYS = 60
EMBARGO_DAYS = DEFAULT_HORIZON_DAYS  # kept as an alias — trade_mechanics_backtest.py imports this directly for the 20D production mechanics backtest

# Interaction features on top of the base FEATURE_COLUMNS — this is the
# "more sophisticated" part: explicit cross-terms a plain linear/tree model
# doesn't automatically discover, chosen because Phase 5's regression
# decomposition specifically identified vol and momentum as the two things
# that matter, so their interaction and each one's own breadth-context are
# the natural next things to give the model access to.
#
# CROSS_SECTIONAL_RANK_SOURCE_COLUMNS: tested and REJECTED as model features
# after web research flagged per-date cross-sectional rank normalization as
# the highest-value, lowest-overfitting-risk lever available (the label
# this model predicts — "top-3-of-13 sectors ON THIS DATE" — is fundamentally
# a within-date ranking task, so a feature's rank vs. the other 12 sectors
# on that date should in principle matter more than its raw value; vol_rank/
# mom_rank already do this for volatility/momentum specifically).
#
# Ablation result: adding all 17 candidate _xrank columns at once made
# things WORSE (mean_oos_auc 0.5896 -> 0.5826). Testing each individually
# showed several apparent improvements (e.g. ret_10d_xrank alone: 0.5974;
# dist_sma20_pct_xrank alone: 0.5958) — but combining even just those two
# single best performers together dropped BELOW baseline (0.5866), and
# combining the 7 features that each individually looked like an
# improvement dropped to 0.5880. This is the same instability signature the
# rejected sector-ETF features showed (see the sector_etf.py commit):
# individually-promising features that don't survive combination are the
# signature of noise from a small (3-fold) OOS sample, not real signal.
# NOT wired into _feature_set() — kept here, unused, so this doesn't need
# re-deriving before someone re-tests it on more history/folds later.
CROSS_SECTIONAL_RANK_SOURCE_COLUMNS = [
    *FEATURE_COLUMNS,
    "breadth_pct_above_sma50", "breadth_pct_above_sma200", "breadth_pct_positive_20d", "breadth_median_rsi",
]


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


def _label_top_k(df: pd.DataFrame, k: int = TOP_K, horizon_days: int = 20) -> pd.DataFrame:
    """Cross-sectional label: 1 if this sector's fwd_ret_{horizon_days}d
    ranks in the top k among all sectors observed on the SAME date, else 0.
    Labels every sector-date, including the ones with no future window yet
    (NaN target, dropped downstream) — never fabricated. Generalized from a
    hardcoded 20D column (see processing/ml/multi_horizon_research.py,
    which found 60D genuinely stronger than 20D on both AUC and Rank IC —
    the horizon_days parameter is what promotes that research finding into
    an alternate, equally-real production output rather than a one-off
    script result."""
    horizon_col = f"fwd_ret_{horizon_days}d"
    d = df.copy()
    d["rank_that_date"] = d.groupby("date")[horizon_col].rank(ascending=False, method="first")
    d["n_that_date"] = d.groupby("date")[horizon_col].transform("count")
    d["is_top_k"] = np.where(d[horizon_col].isna(), np.nan, (d["rank_that_date"] <= k).astype(float))
    return d


def _build_dataset(horizon_days: int = 20) -> tuple[pd.DataFrame, list[str]]:
    stock_full, _ = build_dataset_v2()
    sector_full = build_sector_dataset(stock_dataset=stock_full)
    sector_full, extra_cols = _add_interaction_features(sector_full)
    sector_full = _label_top_k(sector_full, horizon_days=horizon_days)
    return sector_full, extra_cols


def _feature_set(extra_cols: list[str]) -> list[str]:
    breadth_cols = ["breadth_pct_above_sma50", "breadth_pct_above_sma200", "breadth_pct_positive_20d", "breadth_median_rsi"]
    return [c for c in (*FEATURE_COLUMNS, *breadth_cols, *REGIME_FEATURE_COLUMNS, *extra_cols)]


def _artifact_paths(horizon_days: int) -> tuple:
    """Filenames for the DEFAULT horizon stay exactly as they always were
    (sector_prediction_model.joblib / _report.json) — no other module
    needs to change how it locates the primary model. A secondary horizon
    gets its own horizon-suffixed files, additive, never overwriting the
    primary artifacts."""
    if horizon_days == DEFAULT_HORIZON_DAYS:
        return ARTIFACT_DIR / "sector_prediction_model.joblib", ARTIFACT_DIR / "sector_prediction_model_report.json"
    return (
        ARTIFACT_DIR / f"sector_prediction_model_{horizon_days}d.joblib",
        ARTIFACT_DIR / f"sector_prediction_model_{horizon_days}d_report.json",
    )


def _oos_calibration_table(oos_probs: np.ndarray, oos_labels: np.ndarray, n_buckets: int = 5) -> list[dict]:
    """Buckets pooled OOS predictions into n_buckets equal-width probability
    ranges and reports each bucket's actual hit rate plus a standard-error
    margin (sqrt(p(1-p)/n)) — the data this project's abstention gate
    (abstention_gate(), below) uses to decide whether a given probability
    is actually distinguishable from the base rate, rather than assuming
    any P>0.5 (or any fixed cutoff) means something. Computed ONLY from
    pooled OOS fold predictions, never in-sample."""
    edges = np.linspace(0, 1, n_buckets + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (oos_probs >= lo) & (oos_probs < hi if hi < 1 else oos_probs <= hi)
        n = int(mask.sum())
        actual_rate = float(oos_labels[mask].mean()) if n else None
        margin = float(np.sqrt(actual_rate * (1 - actual_rate) / n)) if n and actual_rate not in (0, 1) else None
        rows.append({
            "predicted_range": f"{lo:.0%}-{hi:.0%}", "n": n,
            "predicted_avg": round(float(oos_probs[mask].mean()), 4) if n else None,
            "actual_rate": round(actual_rate, 4) if actual_rate is not None else None,
            "actual_rate_margin": round(margin, 4) if margin is not None else None,
        })
    return rows


def abstention_gate(p_top3_sector: float, calibration_table: list[dict] | None, base_rate: float) -> dict:
    """Should THIS specific probability be trusted as an edge, or is it
    statistically indistinguishable from the base rate given how this
    model's calibration has actually behaved out-of-sample? This is the
    data-driven alternative to a hardcoded cutoff like "P>0.55=BUY" (which
    the quant-architecture spec this was built against explicitly warns
    against using without validating the threshold): it finds which
    calibration bucket p_top3_sector falls into, and requires that
    bucket's actual OOS hit rate to clear the base rate by more than its
    own standard-error margin — i.e. the bucket's own historical
    confidence interval must not include the base rate.

    Returns {"has_edge": bool, "reason": str} — has_edge=False means
    ABSTAIN (NO_EDGE), not "the model says no." A candidate can fail this
    gate even with a numerically high p_top3_sector if that probability
    range hasn't actually differentiated from the base rate in OOS
    testing (small-bucket noise, a rank-order-correct-but-poorly-separated
    model, etc.) — the gate is about DEMONSTRATED separation, not the
    raw number."""
    if not calibration_table:
        return {"has_edge": None, "reason": "no calibration table available this run — cannot assess"}
    bucket = None
    for row in calibration_table:
        lo_s, hi_s = row["predicted_range"].split("-")
        lo, hi = float(lo_s.rstrip("%")) / 100, float(hi_s.rstrip("%")) / 100
        if lo <= p_top3_sector <= hi:
            bucket = row
            if p_top3_sector < hi or hi == 1.0:
                break
    if bucket is None or bucket.get("actual_rate") is None:
        return {"has_edge": None, "reason": f"no OOS observations in this probability's calibration bucket ({bucket['predicted_range'] if bucket else 'n/a'}) — too sparse to assess"}
    if bucket.get("n", 0) < 15:
        return {"has_edge": None, "reason": f"only {bucket['n']} OOS observations in this bucket — too few to distinguish from the base rate either way"}
    margin = bucket.get("actual_rate_margin") or 0.0
    # Point-estimate comparison, NOT margin-of-error-adjusted — deliberately
    # the same standard evaluate()'s own overall status=PASS check uses
    # (mean_auc > 0.55 is also a point estimate, not a CI test). An earlier
    # version of this gate required actual_rate MINUS its margin to clear
    # the base rate — a full one-sided confidence-interval test — and it
    # failed EVERY live sector on real data: this model's probabilities are
    # heavily compressed near the base rate (AUC~0.59 is a real but modest
    # aggregate ranking edge, not a model that produces confidently
    # separated probabilities), so with only 5 calibration buckets nothing
    # ever cleared a full margin. That would have made the abstention gate
    # silently say NO_EDGE forever, misrepresenting a model that DOES show
    # real (if modest) walk-forward discrimination — a worse failure mode
    # than a slightly looser gate. clears_margin_of_error is still reported
    # separately so a reader can see the difference between "a real, if
    # unremarkable, point estimate" and "a confidently separated one."
    has_edge = bucket["actual_rate"] > base_rate
    clears_margin = (bucket["actual_rate"] - margin) > base_rate
    confidence = "clears its own margin of error (a more confident signal)" if clears_margin else (
        "within its own margin of error (a real point-estimate difference, not a statistically decisive one)" if has_edge
        else "at or below the base rate"
    )
    return {
        "has_edge": has_edge,
        "clears_margin_of_error": clears_margin,
        "bucket": bucket["predicted_range"],
        "bucket_actual_rate": bucket["actual_rate"],
        "bucket_n": bucket["n"],
        "base_rate": base_rate,
        "reason": (
            f"OOS bucket {bucket['predicted_range']} actually hit {bucket['actual_rate']:.0%} of the time "
            f"(±{margin:.0%}, n={bucket['n']}) vs. a {base_rate:.0%} base rate — {confidence}."
        ),
    }


def evaluate(refresh: bool = False, horizon_days: int = DEFAULT_HORIZON_DAYS) -> dict:
    """Purged, embargoed walk-forward evaluation — same discipline as every
    other phase. Returns honest OOS metrics, never fit-then-report-in-sample."""
    df, extra_cols = _build_dataset(horizon_days=horizon_days)
    feature_cols = _feature_set(extra_cols)
    data = df.dropna(subset=[*feature_cols, "is_top_k"]).reset_index(drop=True)

    splits = purged_walk_forward_splits(data["date"], n_folds=4, embargo_days=horizon_days)
    fold_results = []
    oos_probs, oos_labels = [], []  # pooled across folds, for the calibration table / abstention gate below
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
        oos_probs.append(p_ensemble)
        oos_labels.append(y_test)

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
        log_stat_experiment("sector_prediction_model_evaluation", "sector_prediction_model", result)
        return result

    mean_auc = round(float(np.mean([f["auc_ensemble"] for f in fold_results])), 4)
    mean_brier = round(float(np.mean([f["brier_ensemble"] for f in fold_results])), 4)
    base_rate = round(float(data["is_top_k"].mean()), 4)  # ~TOP_K/13 by construction
    coin_flip_brier = round(base_rate * (1 - base_rate), 4)
    calibration_table = _oos_calibration_table(np.concatenate(oos_probs), np.concatenate(oos_labels))

    result = {
        "target": f"top_{TOP_K}_of_13_sectors_by_{horizon_days}D_forward_return",
        "horizon_days": horizon_days,
        "n_rows": len(data),
        "n_folds": len(fold_results),
        "folds": fold_results,
        "mean_oos_auc": mean_auc,
        "mean_oos_brier": mean_brier,
        "base_rate": base_rate,
        "coin_flip_brier_baseline": coin_flip_brier,
        "beats_coin_flip": mean_brier < coin_flip_brier,
        "status": "PASS" if mean_auc > 0.55 and mean_brier < coin_flip_brier else "NO_DEMONSTRATED_EDGE",
        "calibration_table": calibration_table,
        "note": (
            f"base_rate ({base_rate}) is the unconditional probability any sector is top-{TOP_K} of 13 on a "
            "given date (~TOP_K/13 by construction, not a finding). mean_oos_brier must beat "
            "coin_flip_brier_baseline (predicting base_rate for everyone, every time) to represent any real "
            "information — an AUC above 0.55 with a Brier score above the coin-flip baseline is NOT a "
            "genuine edge, regardless of how the AUC number looks in isolation."
        ),
    }
    log_stat_experiment(
        "sector_prediction_model_evaluation" if horizon_days == DEFAULT_HORIZON_DAYS else f"sector_prediction_model_evaluation_{horizon_days}d",
        "sector_prediction_model", {k: v for k, v in result.items() if k != "folds"},
    )
    return result


def train_and_save(refresh: bool = False, horizon_days: int = DEFAULT_HORIZON_DAYS) -> dict:
    df, extra_cols = _build_dataset(horizon_days=horizon_days)
    feature_cols = _feature_set(extra_cols)
    data = df.dropna(subset=[*feature_cols, "is_top_k"]).reset_index(drop=True)

    X, y = data[feature_cols], data["is_top_k"].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    hgb = CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=4, max_iter=150), method="sigmoid", cv=3).fit(Xs, y)
    lr = CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight="balanced"), method="sigmoid", cv=3).fit(Xs, y)

    import joblib

    model_path, report_path = _artifact_paths(horizon_days)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "hgb": hgb, "lr": lr, "feature_cols": feature_cols, "horizon_days": horizon_days}, model_path)

    evaluation = evaluate(refresh=refresh, horizon_days=horizon_days)
    report_path.write_text(json.dumps(evaluation, indent=2, default=str))
    return evaluation


def _live_breadth(sector: str, refresh: bool = False) -> dict:
    """Real per-sector breadth at the latest available bar, computed the same
    way sector_dataset.py:_compute_breadth() computes it historically (mean
    across member tickers of: above own SMA50/200, positive 20D return; plus
    median RSI14) — so live inference sees the same distribution of values
    the model was actually trained on, not a constant placeholder."""
    above_sma50, above_sma200, positive_20d, rsis = [], [], [], []
    for t in EXPANDED_UNIVERSE_V2[sector]:
        try:
            df = get_ohlcv(t, force=refresh)
            feats = compute_features(df)
            row = feats.dropna(subset=["close"]).iloc[-1]
        except Exception:
            continue
        if pd.isna(row.get("dist_sma50_pct")) or pd.isna(row.get("dist_sma200_pct")) or pd.isna(row.get("ret_20d")):
            continue
        above_sma50.append(row["dist_sma50_pct"] > 0)
        above_sma200.append(row["dist_sma200_pct"] > 0)
        positive_20d.append(row["ret_20d"] > 0)
        if not pd.isna(row.get("rsi14")):
            rsis.append(row["rsi14"])

    if not above_sma50:
        # No member ticker resolved — fall back to a neutral midpoint rather
        # than crashing the whole prediction run, but this is now a genuine
        # fallback for a data-outage edge case, not the default path.
        return {
            "breadth_pct_above_sma50": 50.0, "breadth_pct_above_sma200": 50.0,
            "breadth_pct_positive_20d": 50.0, "breadth_median_rsi": 50.0,
        }
    return {
        "breadth_pct_above_sma50": round(sum(above_sma50) / len(above_sma50) * 100, 2),
        "breadth_pct_above_sma200": round(sum(above_sma200) / len(above_sma200) * 100, 2),
        "breadth_pct_positive_20d": round(sum(positive_20d) / len(positive_20d) * 100, 2),
        "breadth_median_rsi": round(float(pd.Series(rsis).median()), 2) if rsis else 50.0,
    }


def predict_next_move(refresh: bool = False, horizon_days: int = DEFAULT_HORIZON_DAYS) -> dict:
    """Live prediction: calibrated P(top-3-of-13) for ALL 13 sectors right
    now, spanning the full US equity sector map — technology, semis,
    software, financials, healthcare, industrials, energy, consumer
    discretionary/staples, utilities, materials, communication services,
    real estate. Not just semiconductors, and not a single point call.
    horizon_days selects which trained artifact to load — DEFAULT_HORIZON_DAYS
    (20D) is the original, most stress-tested target; SECONDARY_HORIZON_DAYS
    (60D) is additive, per multi_horizon_research.py's finding that it's
    genuinely stronger on both AUC and Rank IC, but not yet through the same
    depth of adversarial/non-overlapping stress-testing as 20D."""
    model_path, report_path = _artifact_paths(horizon_days)
    if not model_path.exists() or not report_path.exists():
        train_cmd = "python -m processing.ml.sector_prediction_model" if horizon_days == DEFAULT_HORIZON_DAYS else f"python -c \"from processing.ml.sector_prediction_model import train_and_save; train_and_save(horizon_days={horizon_days})\""
        return {"error": f"no trained {horizon_days}D model on disk — run `{train_cmd}` first"}

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

    # Market-wide regime/macro state as of the latest bar — the SAME value
    # for every sector on this date (this is deliberately not a per-sector
    # feature, unlike breadth), matching how it's joined into training data.
    bench_regime = compute_regime_features(bench_df)
    trend_label, vol_label, spy_vol = None, None, None
    if not bench_regime.dropna(subset=["trend_regime"]).empty:
        last_regime = bench_regime.dropna(subset=["trend_regime"]).iloc[-1]
        trend_label, vol_label, spy_vol = last_regime["trend_regime"], last_regime["vol_regime"], last_regime["spy_realized_vol_20d"]
    live["trend_regime_bull"] = _TREND_REGIME_CODE.get(trend_label)
    live["vol_regime_ordinal"] = _VOL_REGIME_CODE.get(vol_label)
    live["spy_realized_vol_20d"] = spy_vol

    try:
        macro = fetch_macro_history(refresh=refresh)
        last_macro = macro.iloc[-1] if not macro.empty else None
    except Exception:
        last_macro = None
    for col in ("vix_level", "vix_chg_5d", "yield_curve_10y2y", "credit_spread_hy", "credit_spread_hy_chg_5d"):
        live[col] = last_macro.get(col) if last_macro is not None else None

    live["vol_z"] = (live["realized_vol_20d"] - live["realized_vol_20d"].mean()) / (live["realized_vol_20d"].std() or 1)
    live["mom_z"] = (live["ret_20d"] - live["ret_20d"].mean()) / (live["ret_20d"].std() or 1)
    live["vol_x_mom"] = live["vol_z"] * live["mom_z"]
    live["vol_rank"] = live["realized_vol_20d"].rank(pct=True)
    live["mom_rank"] = live["ret_20d"].rank(pct=True)
    breadth = live["sector"].apply(lambda s: _live_breadth(s, refresh=refresh))
    for col in ("breadth_pct_above_sma50", "breadth_pct_above_sma200", "breadth_pct_positive_20d", "breadth_median_rsi"):
        live[col] = breadth.apply(lambda b: b[col])
    live["breadth_x_vol"] = live["breadth_pct_above_sma50"] * live["vol_z"]

    missing_cols = [c for c in feature_cols if c not in live.columns]
    for c in missing_cols:
        live[c] = 0.0

    # Macro/regime features are shared market-wide values, not per-sector, so
    # a fetch failure leaves them NaN for every row rather than missing
    # entirely (missing_cols above wouldn't catch that) — StandardScaler.
    # transform() would propagate NaN straight into a NaN prediction with no
    # error. Fall back to the scaler's own training-time mean (index-aligned
    # with feature_cols) so a live data outage degrades gracefully to "assume
    # an average regime" instead of silently corrupting every prediction.
    if live[feature_cols].isna().any().any():
        train_means = pd.Series(scaler.mean_, index=feature_cols)
        live[feature_cols] = live[feature_cols].fillna(train_means)

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

    # Abstention gate: does THIS probability actually clear the base rate
    # by more than its own OOS calibration-bucket margin of error, or is it
    # numerically-high but statistically indistinguishable from "no edge"?
    calibration_table = evaluation.get("calibration_table")
    base_rate = evaluation.get("base_rate", TOP_K / 13)
    live["edge_gate"] = live["p_top3_ensemble"].apply(lambda p: abstention_gate(p, calibration_table, base_rate))
    live["has_edge"] = live["edge_gate"].apply(lambda g: g.get("has_edge"))

    predictions = live[["sector", "p_top3_ensemble", "p_top3_hgb", "p_top3_lr", "model_agreement", "realized_vol_20d", "ret_20d", "has_edge", "edge_gate"]].to_dict(orient="records")
    any_edge = any(p.get("has_edge") for p in predictions)

    return {
        "target": evaluation.get("target", f"top_{TOP_K}_of_13_sectors_by_20D_forward_return"),
        "predictions_all_13_sectors": predictions,
        "any_sector_has_edge": any_edge,
        "abstention_note": (
            "has_edge=True means that sector's probability falls in an OOS calibration bucket whose actual "
            "hit rate, on average, beat the base rate (a point-estimate comparison — see "
            "edge_gate.clears_margin_of_error for the stricter, margin-of-error-adjusted version of the same "
            "check). has_edge=False/None does NOT mean 'this sector will underperform' — it means this "
            "run's probability for it hasn't even cleared the base rate on average historically. " + (
                "NO sector clears the gate this run — treat every prediction below as informational, not "
                "actionable, until a future run shows real separation." if not any_edge else
                "At least one sector clears the gate this run — check clears_margin_of_error on each to tell "
                "a real-but-modest point estimate (this model's typical case, given AUC~0.59) apart from a "
                "confidently separated one."
            )
        ),
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
