"""Phase 3: does the ATR%/realized-vol finding from Phase 2 survive broader
testing, or was it a 32-ticker, mega-cap-tech, single-bull-regime artifact?

Everything below runs on a DEV split (first 85% of dates in the expanded,
103-ticker, sector-balanced universe). The final 15% is a NEW golden holdout
— separate from Phase 2's own holdout, which stays frozen and untouched —
evaluated exactly once, at the end, on whichever result looks most decisive
on dev data.

Run: `python -m processing.ml.phase3_volatility`
Output: storage/models/phase3_report.json + entries appended to
storage/models/experiments.jsonl (never overwritten).

SCOPE CUTS (disclosed, not silent): no beta-neutral portfolio construction,
no market-cap/liquidity bucketing, no true daily-rebalanced turnover/Sharpe/
drawdown backtest (a real position-level backtest across 103 names is a
separate, larger build) — the cross-sectional spread analysis here uses
mean/std of the per-rebalance-date spread as a signal-vs-noise proxy, same
caveat as Phase 2's decile_spread.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from processing.ml.dataset_v2 import build_dataset_v2, universe_report
from processing.ml.registry import ARTIFACT_DIR, log_stat_experiment

VOL_FEATURES = ["atr_pct", "realized_vol_20d"]
PRIMARY_RETURN = "fwd_ret_20d"
PRIMARY_DIRECTION = "fwd_up_20d"
HOLDOUT_FRACTION = 0.15
COST_BPS_GRID = (0, 5, 10, 25, 50)
N_EXPERIMENTS_RUN = 0  # incremented as we go, for the multiple-testing disclosure


def _ic(df: pd.DataFrame, feature: str, ret_col: str = PRIMARY_RETURN) -> tuple[float | None, int]:
    sub = df.dropna(subset=[feature, ret_col])
    if len(sub) < 30:
        return None, len(sub)
    r = spearmanr(sub[feature], sub[ret_col]).statistic
    return (round(float(r), 4) if r == r else None), len(sub)


def _auc(df: pd.DataFrame, feature: str, dir_col: str = PRIMARY_DIRECTION) -> tuple[float | None, int]:
    sub = df.dropna(subset=[feature, dir_col])
    if len(sub) < 30 or sub[dir_col].nunique() < 2:
        return None, len(sub)
    a = roc_auc_score(sub[dir_col], sub[feature])
    return round(float(a), 4), len(sub)


def _decile_spread(df: pd.DataFrame, feature: str, ret_col: str = PRIMARY_RETURN, min_n: int = 15) -> dict:
    """Per-date cross-section: rank by `feature`, spread = mean(top decile) -
    mean(bottom decile) forward return. Dates with < min_n names are skipped
    (not enough of a cross-section to rank meaningfully)."""
    sub = df.dropna(subset=[feature, ret_col])
    spreads = []
    for _, g in sub.groupby("date"):
        if len(g) < min_n:
            continue
        gs = g.sort_values(feature)
        n = len(gs)
        k = max(1, n // 10)
        spreads.append(gs.iloc[-k:][ret_col].mean() - gs.iloc[:k][ret_col].mean())
    if not spreads:
        return {"available": False}
    arr = np.array(spreads)
    return {
        "available": True,
        "n_dates": len(arr),
        "mean_spread_pct": round(float(arr.mean()), 3),
        "median_spread_pct": round(float(np.median(arr)), 3),
        "pct_dates_positive": round(float((arr > 0).mean() * 100), 1),
        "spread_std": round(float(arr.std()), 3),
        "spread_sharpe_like": round(float(arr.mean() / arr.std()), 3) if arr.std() > 0 else None,
    }


def _sector_neutral_spread(df: pd.DataFrame, feature: str, ret_col: str = PRIMARY_RETURN, min_n: int = 6) -> dict:
    """Within each (date, sector) group with enough names, top-half minus
    bottom-half forward return by `feature` rank — then averaged first within
    sector-dates, then across sectors, so no single sector's size dominates."""
    sub = df.dropna(subset=[feature, ret_col])
    per_sector = {}
    for sector, g_sector in sub.groupby("sector"):
        spreads = []
        for _, g in g_sector.groupby("date"):
            if len(g) < min_n:
                continue
            gs = g.sort_values(feature)
            n = len(gs)
            half = max(1, n // 2)
            spreads.append(gs.iloc[-half:][ret_col].mean() - gs.iloc[:half][ret_col].mean())
        if spreads:
            per_sector[sector] = {"n_dates": len(spreads), "mean_spread_pct": round(float(np.mean(spreads)), 3)}
    if not per_sector:
        return {"available": False}
    overall = np.mean([v["mean_spread_pct"] for v in per_sector.values()])
    return {
        "available": True,
        "per_sector": per_sector,
        "mean_across_sectors_pct": round(float(overall), 3),
        "n_sectors_with_data": len(per_sector),
    }


def _cost_adjusted(spread_result: dict, rebalance_every_n_days: int = 5) -> dict:
    """Round-trip cost per rebalance ≈ 4x the one-way bps (close old long,
    open new long, close old short, open new short) — a rough, disclosed
    approximation, not a real execution-cost model."""
    if not spread_result.get("available"):
        return {"available": False}
    out = {}
    for bps in COST_BPS_GRID:
        cost_pct = 4 * bps / 100
        out[f"{bps}bps"] = round(spread_result["mean_spread_pct"] - cost_pct, 3)
    return out


def _block_bootstrap_ic_ci(df: pd.DataFrame, feature: str, ret_col: str = PRIMARY_RETURN, n_boot: int = 500, seed: int = 11) -> dict:
    """Resamples whole DATES (blocks), not individual rows, since rows sharing
    a date are cross-sectionally correlated — a row-level bootstrap would
    understate the true uncertainty."""
    sub = df.dropna(subset=[feature, ret_col])
    dates = sub["date"].unique()
    if len(dates) < 20:
        return {"available": False}
    rng = np.random.default_rng(seed)
    ics = []
    for _ in range(n_boot):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        boot = pd.concat([sub[sub["date"] == d] for d in np.unique(sample_dates)], ignore_index=True)
        if len(boot) < 30:
            continue
        r = spearmanr(boot[feature], boot[ret_col]).statistic
        if r == r:
            ics.append(r)
    if not ics:
        return {"available": False}
    ics = np.array(ics)
    return {
        "available": True,
        "n_boot": len(ics),
        "mean_ic": round(float(ics.mean()), 4),
        "ci_5_95": [round(float(np.percentile(ics, 5)), 4), round(float(np.percentile(ics, 95)), 4)],
        "pct_boot_positive": round(float((ics > 0).mean() * 100), 1),
    }


def main() -> dict:
    print("Building expanded (Phase 3) dataset...")
    full, excluded = build_dataset_v2()
    dates_sorted = np.sort(full["date"].unique())
    cutoff = dates_sorted[int(len(dates_sorted) * (1 - HOLDOUT_FRACTION))]
    dev = full[full["date"] < cutoff].reset_index(drop=True)
    holdout = full[full["date"] >= cutoff].reset_index(drop=True)
    print(f"dev: {len(dev)} rows, holdout (untouched): {len(holdout)} rows, cutoff {pd.Timestamp(cutoff).date()}")

    report: dict = {"universe": universe_report(full, excluded), "dev_rows": len(dev), "holdout_rows": len(holdout)}

    # ---- A. IC / AUC / decile, whole dev set, both vol features -----------
    print("\n[A] Baseline IC/AUC/decile on expanded universe...")
    core = {}
    for feat in VOL_FEATURES:
        ic, n_ic = _ic(dev, feat)
        auc, n_auc = _auc(dev, feat)
        decile = _decile_spread(dev, feat)
        core[feat] = {"ic": ic, "n_ic": n_ic, "auc": auc, "n_auc": n_auc, "cross_sectional_decile_spread": decile}
        print(f"  {feat}: IC={ic} (n={n_ic}) AUC={auc} decile_spread={decile.get('mean_spread_pct')}")
    report["core_expanded_universe"] = core
    log_stat_experiment("phase3_core_ic_auc_decile", "phase3", core)

    # ---- B. Stability: by year, sector, regime -----------------------------
    print("\n[B] Stability across years / sectors / regimes...")
    stability = {}
    for feat in VOL_FEATURES:
        by_year = {int(y): _ic(g, feat)[0] for y, g in dev.assign(year=dev["date"].dt.year).groupby("year")}
        by_sector = {s: _ic(g, feat)[0] for s, g in dev.groupby("sector")}
        by_trend_regime = {r: _ic(g, feat)[0] for r, g in dev.dropna(subset=["trend_regime"]).groupby("trend_regime")}
        by_vol_regime = {r: _ic(g, feat)[0] for r, g in dev.dropna(subset=["vol_regime"]).groupby("vol_regime", observed=True)}
        stability[feat] = {"by_year": by_year, "by_sector": by_sector, "by_trend_regime": by_trend_regime, "by_vol_regime": by_vol_regime}
        print(f"  {feat} by_trend_regime: {by_trend_regime}")
        print(f"  {feat} by_vol_regime: {by_vol_regime}")
    report["stability"] = stability
    log_stat_experiment("phase3_stability_year_sector_regime", "phase3", stability)

    # ---- C. Placebo tests ---------------------------------------------------
    print("\n[C] Placebo tests (is this a real signal or a proxy)...")
    placebo = {}

    # C1: does current vol predict a rough proxy for FUTURE realized range?
    # (fwd_mfe - fwd_mae over the same 20D window is a crude forward-range proxy)
    dev_c = dev.copy()
    dev_c["fwd_range_20d"] = dev_c["fwd_mfe_20d"] - dev_c["fwd_mae_20d"]
    ic_vol_autocorr, n1 = _ic(dev_c, "realized_vol_20d", ret_col="fwd_range_20d")
    placebo["A_current_vol_predicts_future_range"] = {"ic": ic_vol_autocorr, "n": n1, "interpretation": "expected strongly positive — volatility clusters, this is a sanity check not a finding"}

    # C2: already have — vol -> future return (= core_expanded_universe above)
    placebo["B_vol_predicts_return"] = core["realized_vol_20d"]

    # C3: future realized range -> future return (does the move simply co-occur with the range, independent of what was knowable at t?)
    ic_futvol_futret, n3 = _ic(dev_c, "fwd_range_20d", ret_col=PRIMARY_RETURN)
    placebo["C_future_range_vs_future_return"] = {"ic": ic_futvol_futret, "n": n3, "interpretation": "large moves mechanically have large range in the direction moved — expected strongly positive, not informative about predictability"}

    # C4: residualize vol against 20D momentum — does the vol signal survive?
    sub = dev.dropna(subset=["realized_vol_20d", "ret_20d", PRIMARY_RETURN])
    X = sub[["ret_20d"]].values
    y = sub["realized_vol_20d"].values
    resid = y - LinearRegression().fit(X, y).predict(X)
    resid_ic = spearmanr(resid, sub[PRIMARY_RETURN]).statistic
    placebo["D_vol_residualized_against_momentum"] = {
        "raw_ic": core["realized_vol_20d"]["ic"],
        "residual_ic_after_removing_ret_20d": round(float(resid_ic), 4) if resid_ic == resid_ic else None,
        "n": len(sub),
        "interpretation": (
            "If residual_ic collapses toward 0 relative to raw_ic, volatility was mostly acting as a "
            "momentum proxy (stocks that already moved a lot are both more volatile AND more likely to "
            "keep moving). If residual_ic stays comparable to raw_ic, volatility carries information "
            "beyond momentum."
        ),
    }
    print(f"  D: raw IC {placebo['D_vol_residualized_against_momentum']['raw_ic']} -> residual IC {placebo['D_vol_residualized_against_momentum']['residual_ic_after_removing_ret_20d']}")
    report["placebo_tests"] = placebo
    log_stat_experiment("phase3_placebo_tests", "phase3", placebo)

    # ---- D. Volatility x momentum interaction grid --------------------------
    print("\n[D] Volatility x momentum interaction grid...")
    grid_df = dev.dropna(subset=["realized_vol_20d", "ret_20d", PRIMARY_RETURN]).copy()
    grid_df["vol_tercile"] = pd.qcut(grid_df["realized_vol_20d"], 3, labels=["low_vol", "mid_vol", "high_vol"], duplicates="drop")
    grid_df["mom_tercile"] = pd.qcut(grid_df["ret_20d"], 3, labels=["down", "flat", "up"], duplicates="drop")
    interaction = (
        grid_df.groupby(["vol_tercile", "mom_tercile"], observed=True)
        .agg(n=(PRIMARY_RETURN, "size"), mean_fwd_ret_pct=(PRIMARY_RETURN, "mean"), win_rate_pct=(PRIMARY_RETURN, lambda s: (s > 0).mean() * 100))
        .round(3)
        .reset_index()
    )
    report["volatility_momentum_interaction"] = interaction.to_dict(orient="records")
    log_stat_experiment("phase3_vol_momentum_interaction", "phase3", report["volatility_momentum_interaction"])
    print(interaction.to_string(index=False))

    # ---- E. Sector-neutral spread + costs -----------------------------------
    print("\n[E] Sector-neutral spread + transaction-cost sensitivity...")
    raw_spread = core["realized_vol_20d"]["cross_sectional_decile_spread"]
    sector_neutral = _sector_neutral_spread(dev, "realized_vol_20d")
    costs_raw = _cost_adjusted(raw_spread)
    report["sector_neutral_spread"] = sector_neutral
    report["cost_sensitivity_raw_spread"] = costs_raw
    print(f"  raw decile spread: {raw_spread.get('mean_spread_pct')}%  sector-neutral: {sector_neutral.get('mean_across_sectors_pct')}%")
    print(f"  cost-adjusted (raw): {costs_raw}")
    log_stat_experiment("phase3_sector_neutral_and_costs", "phase3", {"sector_neutral": sector_neutral, "costs": costs_raw})

    # ---- F. Adversarial tests ------------------------------------------------
    print("\n[F] Adversarial tests...")
    rng = np.random.default_rng(7)
    adversarial = {}

    shuffled = dev.copy()
    shuffled["realized_vol_20d_shuffled"] = rng.permutation(shuffled["realized_vol_20d"].values)
    adversarial["shuffled_volatility"] = _ic(shuffled, "realized_vol_20d_shuffled")[0]

    # time-shift: deliberately misalign vol at t with return at t+120 rows away
    # (a nonsense temporal offset — should destroy any real relationship)
    shifted = dev.sort_values(["ticker", "date"]).copy()
    shifted["realized_vol_20d_shifted"] = shifted.groupby("ticker")["realized_vol_20d"].shift(24)  # ~24 stride-5 steps = ~120 trading days
    adversarial["time_shifted_volatility_ic"] = _ic(shifted, "realized_vol_20d_shifted")[0]

    sector_perm = dev.copy()
    sector_perm["sector_shuffled"] = rng.permutation(sector_perm["sector"].values)
    sp_result = _sector_neutral_spread(sector_perm.rename(columns={"sector": "sector_orig", "sector_shuffled": "sector"}), "realized_vol_20d")
    adversarial["sector_permuted_neutral_spread"] = sp_result.get("mean_across_sectors_pct")

    report["adversarial_tests"] = adversarial
    print(f"  shuffled_volatility IC: {adversarial['shuffled_volatility']} (expect ~0)")
    print(f"  time_shifted_volatility IC: {adversarial['time_shifted_volatility_ic']} (expect ~0)")
    print(f"  sector_permuted neutral spread: {adversarial['sector_permuted_neutral_spread']} (expect close to raw sector-neutral value — permuting sector labels shouldn't matter much here since it's the neutralization axis, not the signal)")
    log_stat_experiment("phase3_adversarial_tests", "phase3", adversarial)

    # ---- G. Bootstrap CI on pooled IC ---------------------------------------
    print("\n[G] Block-bootstrap confidence interval on realized_vol_20d IC...")
    ci = _block_bootstrap_ic_ci(dev, "realized_vol_20d")
    report["bootstrap_ci"] = ci
    print(f"  {ci}")
    log_stat_experiment("phase3_bootstrap_ci", "phase3", ci)

    # ---- H. Golden holdout: ONE frozen confirmation -------------------------
    print("\n[H] Golden holdout — evaluated once...")
    holdout_ic, n_h = _ic(holdout, "realized_vol_20d")
    holdout_decile = _decile_spread(holdout, "realized_vol_20d")
    golden = {
        "holdout_period": [str(pd.Timestamp(holdout["date"].min()).date()), str(pd.Timestamp(holdout["date"].max()).date())],
        "n_rows": len(holdout),
        "ic": holdout_ic,
        "n_ic": n_h,
        "cross_sectional_decile_spread": holdout_decile,
        "note": "Frozen — evaluated exactly once against data no experiment above touched. Do not re-run tuning against this.",
    }
    report["golden_holdout"] = golden
    print(f"  holdout IC: {holdout_ic}, decile spread: {holdout_decile.get('mean_spread_pct')}%")
    log_stat_experiment("phase3_golden_holdout", "phase3", golden)

    # ---- I. Multiple-testing disclosure -------------------------------------
    all_ics = (
        [core[f]["ic"] for f in VOL_FEATURES]
        + [v for f in VOL_FEATURES for v in stability[f]["by_year"].values()]
        + [v for f in VOL_FEATURES for v in stability[f]["by_sector"].values()]
        + [v for f in VOL_FEATURES for v in stability[f]["by_trend_regime"].values()]
        + [v for f in VOL_FEATURES for v in stability[f]["by_vol_regime"].values()]
    )
    valid_ics = [v for v in all_ics if v is not None]
    n_positive = sum(1 for v in valid_ics if v > 0)
    report["multiple_testing"] = {
        "n_ic_estimates_computed": len(valid_ics),
        "n_positive": n_positive,
        "pct_positive": round(n_positive / len(valid_ics) * 100, 1) if valid_ics else None,
        "note": (
            f"{len(valid_ics)} separate IC estimates were computed across features/years/sectors/regimes "
            "in this phase. With this many looks, some nominally-positive slices are expected by chance "
            "alone even under the null — read the AGGREGATE pattern (pct_positive, bootstrap CI, holdout) "
            "as the evidence, not any single favorable slice."
        ),
    }
    print(f"\n[I] Multiple testing: {n_positive}/{len(valid_ics)} IC estimates positive ({report['multiple_testing']['pct_positive']}%)")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "phase3_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull report written to {ARTIFACT_DIR / 'phase3_report.json'}")
    return report


if __name__ == "__main__":
    main()
