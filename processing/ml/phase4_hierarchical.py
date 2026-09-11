"""Phase 4: does the sector-level structure from Phase 3 (raw spread ~2.7%,
sector-neutral spread ~0.7%, mostly sector-composition-driven) support a real
hierarchical MARKET -> SECTOR -> STOCK-WITHIN-SECTOR ranking system?

Everything below runs on a DEV split of a NEW dataset pair (13 synthetic
sector indices + the Phase 3 stock-level panel, both point-in-time-safe). A
NEW golden holdout (last 15% of dates) is reserved and touched exactly once,
at the end, for the single most promising rotation config found on dev data.

Phase 1/2/3 datasets, models, holdouts, and reports are untouched — this is a
new module tree (sector_index.py, sector_dataset.py, this file).

Run: `python -m processing.ml.phase4_hierarchical`
Output: storage/models/phase4_report.json + registry entries.

SCOPE CUTS (disclosed): no market-cap/liquidity bucketing (still no
point-in-time size/liquidity data source), no survivorship correction (same
data-source limit as Phase 3), no per-regime probability calibration curves
(no probabilistic model built this phase — everything here is a ranking/
IC-based test per the brief's own "start with simple ranking" instruction),
120D sector momentum (feature not in the existing FEATURE_COLUMNS set,
skipped rather than bolting on a one-off column).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from processing.ml.dataset_v2 import build_dataset_v2
from processing.ml.registry import ARTIFACT_DIR, log_stat_experiment
from processing.ml.sector_dataset import build_sector_dataset

HOLDOUT_FRACTION = 0.15
COST_BPS_GRID = (0, 5, 10, 25, 50)
TOPN_GRID = (1, 2, 3)
N_SECTORS = 13


def _ic(df, feature, ret_col="fwd_ret_20d"):
    sub = df.dropna(subset=[feature, ret_col])
    if len(sub) < 30:
        return None, len(sub)
    r = spearmanr(sub[feature], sub[ret_col]).statistic
    return (round(float(r), 4) if r == r else None), len(sub)


def _topn_bottomn_spread(df, feature, ret_col="fwd_ret_20d", n=3):
    """Sector-rotation-shaped spread: at each date rank the (up to 13)
    sectors by `feature`, compare mean forward return of the top n vs
    bottom n vs the equal-weight-all-sectors benchmark that date."""
    sub = df.dropna(subset=[feature, ret_col])
    top_rets, bottom_rets, bench_rets = [], [], []
    for _, g in sub.groupby("date"):
        if len(g) < max(6, 2 * n):
            continue
        gs = g.sort_values(feature)
        top_rets.append(gs.iloc[-n:][ret_col].mean())
        bottom_rets.append(gs.iloc[:n][ret_col].mean())
        bench_rets.append(g[ret_col].mean())
    if not top_rets:
        return {"available": False}
    top, bot, bench = np.array(top_rets), np.array(bottom_rets), np.array(bench_rets)
    spread = top - bot
    excess_vs_bench = top - bench
    return {
        "available": True,
        "n_dates": len(top),
        "n_sectors_long": n,
        "mean_top_return_pct": round(float(top.mean()), 3),
        "mean_bottom_return_pct": round(float(bot.mean()), 3),
        "mean_bench_equal_weight_pct": round(float(bench.mean()), 3),
        "mean_top_minus_bottom_pct": round(float(spread.mean()), 3),
        "mean_top_minus_bench_pct": round(float(excess_vs_bench.mean()), 3),
        "pct_dates_top_beats_bench": round(float((excess_vs_bench > 0).mean() * 100), 1),
        "top_return_std": round(float(top.std()), 3),
        "top_sharpe_like": round(float(top.mean() / top.std()), 3) if top.std() > 0 else None,
    }


def _cost_grid(spread_available: dict, long_only: bool) -> dict:
    if not spread_available.get("available"):
        return {"available": False}
    sides = 2 if long_only else 4  # long-only: close+open one leg; long-short: both legs
    return {f"{bps}bps": round(spread_available["mean_top_return_pct"] - sides * bps / 100, 3) for bps in COST_BPS_GRID}


def main() -> dict:
    print("Building stock-level (Phase 3) and sector-level (Phase 4) datasets...")
    stock_full, excluded = build_dataset_v2()
    sector_full = build_sector_dataset(stock_dataset=stock_full)

    dates_sorted = np.sort(sector_full["date"].unique())
    cutoff = dates_sorted[int(len(dates_sorted) * (1 - HOLDOUT_FRACTION))]
    sector_dev = sector_full[sector_full["date"] < cutoff].reset_index(drop=True)
    sector_holdout = sector_full[sector_full["date"] >= cutoff].reset_index(drop=True)
    stock_dev = stock_full[stock_full["date"] < cutoff].reset_index(drop=True)
    print(f"sector dev: {len(sector_dev)} rows, sector holdout: {len(sector_holdout)} rows, cutoff {pd.Timestamp(cutoff).date()}")

    report: dict = {"sector_dev_rows": len(sector_dev), "sector_holdout_rows": len(sector_holdout)}

    # ---- 1. Sector volatility -> forward sector return, multi-horizon ------
    print("\n[1] Sector volatility vs forward sector return, multi-horizon...")
    horizon_results = {}
    for h in (5, 20, 60):
        ret_col = f"fwd_ret_{h}d"
        ic, n = _ic(sector_dev, "realized_vol_20d", ret_col=ret_col)
        spread = _topn_bottomn_spread(sector_dev, "realized_vol_20d", ret_col=ret_col, n=3)
        horizon_results[f"{h}d"] = {"ic": ic, "n": n, "top3_bottom3_spread": spread}
        print(f"  {h}D: IC={ic} top3-bottom3={spread.get('mean_top_minus_bottom_pct')}")
    report["sector_volatility_by_horizon"] = horizon_results
    log_stat_experiment("phase4_sector_vol_by_horizon", "phase4", horizon_results)

    # ---- 2. Sector-relative volatility (vs market) --------------------------
    print("\n[2] Sector-relative volatility (sector_vol - market_vol)...")
    sector_dev = sector_dev.copy()
    sector_dev["rel_vol_vs_market"] = sector_dev["realized_vol_20d"] - sector_dev["spy_realized_vol_20d"]
    ic_abs, _ = _ic(sector_dev, "realized_vol_20d")
    ic_rel, _ = _ic(sector_dev, "rel_vol_vs_market")
    rel_vol_result = {"absolute_vol_ic": ic_abs, "relative_vol_ic": ic_rel}
    report["sector_relative_volatility"] = rel_vol_result
    print(f"  absolute IC={ic_abs}  relative-to-market IC={ic_rel}")
    log_stat_experiment("phase4_relative_volatility", "phase4", rel_vol_result)

    # ---- 3. Cross-sector dispersion -----------------------------------------
    print("\n[3] Cross-sector return dispersion...")
    disp_by_date = sector_dev.groupby("date")["ret_20d"].std().rename("cross_sector_dispersion")
    sector_dev = sector_dev.merge(disp_by_date, on="date", how="left")
    median_disp = sector_dev["cross_sector_dispersion"].median()
    high_disp = sector_dev[sector_dev["cross_sector_dispersion"] >= median_disp]
    low_disp = sector_dev[sector_dev["cross_sector_dispersion"] < median_disp]
    spread_high_disp = _topn_bottomn_spread(high_disp, "realized_vol_20d", n=3)
    spread_low_disp = _topn_bottomn_spread(low_disp, "realized_vol_20d", n=3)
    dispersion_result = {
        "median_cross_sector_dispersion": round(float(median_disp), 3),
        "top3_bottom3_spread_high_dispersion_dates": spread_high_disp.get("mean_top_minus_bottom_pct"),
        "top3_bottom3_spread_low_dispersion_dates": spread_low_disp.get("mean_top_minus_bottom_pct"),
        "interpretation": "if high-dispersion spread >> low-dispersion spread, sector selection is more valuable when sectors are diverging",
    }
    report["cross_sector_dispersion"] = dispersion_result
    print(f"  high-dispersion spread={dispersion_result['top3_bottom3_spread_high_dispersion_dates']}  low-dispersion spread={dispersion_result['top3_bottom3_spread_low_dispersion_dates']}")
    log_stat_experiment("phase4_cross_sector_dispersion", "phase4", dispersion_result)

    # ---- 4. Sector momentum (5/10/20/60D) ------------------------------------
    print("\n[4] Sector momentum...")
    momentum_results = {}
    for feat in ("ret_5d", "ret_10d", "ret_20d", "ret_60d", "rel_ret_20d_vs_benchmark"):
        ic, n = _ic(sector_dev, feat)
        momentum_results[feat] = {"ic": ic, "n": n}
        print(f"  {feat}: IC={ic}")
    report["sector_momentum"] = momentum_results
    log_stat_experiment("phase4_sector_momentum", "phase4", momentum_results)

    # ---- 5. Sector-level mean-reversion bucket test --------------------------
    print("\n[5] Sector-level volatility x momentum bucket test (replicating Phase 3's stock-level finding)...")
    bucket_df = sector_dev.dropna(subset=["realized_vol_20d", "ret_20d", "fwd_ret_20d"]).copy()
    bucket_df["vol_bucket"] = pd.qcut(bucket_df["realized_vol_20d"], 2, labels=["low_vol", "high_vol"], duplicates="drop")
    bucket_df["mom_bucket"] = pd.qcut(bucket_df["ret_20d"], 2, labels=["down", "up"], duplicates="drop")
    mean_reversion = (
        bucket_df.groupby(["vol_bucket", "mom_bucket"], observed=True)
        .agg(n=("fwd_ret_20d", "size"), mean_fwd_ret_20d_pct=("fwd_ret_20d", "mean"), win_rate_pct=("fwd_ret_20d", lambda s: (s > 0).mean() * 100))
        .round(3).reset_index()
    )
    report["sector_mean_reversion_buckets"] = mean_reversion.to_dict(orient="records")
    log_stat_experiment("phase4_sector_mean_reversion", "phase4", report["sector_mean_reversion_buckets"])
    print(mean_reversion.to_string(index=False))

    # ---- 6. Breadth impact ----------------------------------------------------
    print("\n[6] Sector breadth...")
    breadth_results = {}
    for feat in ("breadth_pct_above_sma50", "breadth_pct_above_sma200", "breadth_pct_positive_20d", "breadth_median_rsi"):
        if feat in sector_dev.columns:
            ic, n = _ic(sector_dev, feat)
            breadth_results[feat] = {"ic": ic, "n": n}
            print(f"  {feat}: IC={ic}")
    report["sector_breadth"] = breadth_results
    log_stat_experiment("phase4_sector_breadth", "phase4", breadth_results)

    # ---- 7. Regime-conditioned sector vol IC ----------------------------------
    print("\n[7] Regime-conditioned sector volatility IC...")
    regime_cond = {}
    for trend in ("bull", "bear"):
        for vr in ("low_vol", "mid_vol", "high_vol"):
            sub = sector_dev[(sector_dev["trend_regime"] == trend) & (sector_dev["vol_regime"] == vr)]
            ic, n = _ic(sub, "realized_vol_20d")
            regime_cond[f"{trend}_{vr}"] = {"ic": ic, "n": n}
    report["regime_conditioned_sector_vol_ic"] = regime_cond
    for k, v in regime_cond.items():
        print(f"  {k}: IC={v['ic']} (n={v['n']})")
    log_stat_experiment("phase4_regime_conditioned_sector_vol", "phase4", regime_cond)

    # ---- 8. Volatility decomposition: market / sector / idiosyncratic -------
    print("\n[8] Volatility decomposition (market / sector / idiosyncratic components)...")
    # stock_dev already carries its OWN spy_realized_vol_20d (from dataset_v2's
    # own regime merge, computed identically from the same benchmark) — only
    # pull sector_vol from sector_full to avoid a duplicate-column collision.
    stock_merged = stock_dev.merge(
        sector_full[["sector", "date", "realized_vol_20d"]].rename(columns={"realized_vol_20d": "sector_vol"}),
        on=["sector", "date"], how="left",
    )
    stock_merged["idio_vol"] = stock_merged["realized_vol_20d"] - stock_merged["sector_vol"]
    stock_merged["sector_component_vol"] = stock_merged["sector_vol"] - stock_merged["spy_realized_vol_20d"]
    decomp = {}
    for feat, label in (
        ("spy_realized_vol_20d", "market_component"),
        ("sector_component_vol", "sector_component"),
        ("idio_vol", "idiosyncratic_component"),
        ("realized_vol_20d", "total_stock_vol (undecomposed, for reference)"),
    ):
        ic, n = _ic(stock_merged, feat)
        decomp[label] = {"ic": ic, "n": n}
        print(f"  {label}: IC={ic} (n={n})")
    report["volatility_decomposition"] = decomp
    log_stat_experiment("phase4_volatility_decomposition", "phase4", decomp)

    # ---- 9. Conditional within-sector stock selection ------------------------
    print("\n[9] Within-sector stock selection (volatility rank within sector-date)...")
    within_sector_ics = []
    for _, g in stock_dev.dropna(subset=["realized_vol_20d", "fwd_ret_20d"]).groupby(["sector", "date"]):
        if len(g) < 4:
            continue
        r = spearmanr(g["realized_vol_20d"], g["fwd_ret_20d"]).statistic
        if r == r:
            within_sector_ics.append(r)
    within_sector = {
        "n_sector_date_groups": len(within_sector_ics),
        "mean_within_sector_ic": round(float(np.mean(within_sector_ics)), 4) if within_sector_ics else None,
        "pct_groups_positive": round(float((np.array(within_sector_ics) > 0).mean() * 100), 1) if within_sector_ics else None,
        "note": "Benchmark is random stock selection WITHIN the same sector/date, not the raw cross-universe spread.",
    }
    report["within_sector_stock_selection"] = within_sector
    print(f"  {within_sector}")
    log_stat_experiment("phase4_within_sector_selection", "phase4", within_sector)

    # ---- 10. Conditional-on-sector-regime stock selection ---------------------
    print("\n[10] Within-sector stock selection, conditional on sector volatility regime...")
    stock_merged_vr = stock_merged.copy()
    stock_merged_vr["sector_vol_bucket"] = pd.qcut(stock_merged_vr["sector_vol"], 2, labels=["low", "high"], duplicates="drop")
    conditional_selection = {}
    for bucket in ("low", "high"):
        sub = stock_merged_vr[stock_merged_vr["sector_vol_bucket"] == bucket]
        ics = []
        for _, g in sub.dropna(subset=["realized_vol_20d", "fwd_ret_20d"]).groupby(["sector", "date"]):
            if len(g) < 4:
                continue
            r = spearmanr(g["realized_vol_20d"], g["fwd_ret_20d"]).statistic
            if r == r:
                ics.append(r)
        conditional_selection[f"sector_vol_{bucket}"] = {
            "n_groups": len(ics), "mean_within_sector_ic": round(float(np.mean(ics)), 4) if ics else None,
        }
    report["conditional_stock_selection_by_sector_regime"] = conditional_selection
    print(f"  {conditional_selection}")
    log_stat_experiment("phase4_conditional_stock_selection", "phase4", conditional_selection)

    # ---- 11. Sector rotation backtest (vol, momentum, vol+mom combo) --------
    print("\n[11] Sector rotation backtest (long top-N sectors, 20D hold, costs)...")
    combo = sector_dev.dropna(subset=["realized_vol_20d", "ret_20d", "fwd_ret_20d"]).copy()
    combo["vol_rank"] = combo.groupby("date")["realized_vol_20d"].rank(pct=True)
    combo["mom_rank"] = combo.groupby("date")["ret_20d"].rank(pct=True)
    combo["vol_mom_combo"] = combo["vol_rank"] + combo["mom_rank"]

    rotation_results = {}
    for signal, label in (("realized_vol_20d", "volatility_only"), ("ret_20d", "momentum_only"), ("vol_mom_combo", "vol_plus_momentum")):
        for n in TOPN_GRID:
            spread = _topn_bottomn_spread(combo, signal, n=n)
            costs = _cost_grid(spread, long_only=True)
            rotation_results[f"{label}_top{n}"] = {"spread": spread, "cost_adjusted_top_leg_return": costs}
    report["sector_rotation_backtest"] = rotation_results
    for k, v in rotation_results.items():
        print(f"  {k}: top_return={v['spread'].get('mean_top_return_pct')}%  vs_bench={v['spread'].get('mean_top_minus_bench_pct')}%  net@25bps={v['cost_adjusted_top_leg_return'].get('25bps')}")
    log_stat_experiment("phase4_sector_rotation_backtest", "phase4", {k: v for k, v in rotation_results.items()})

    # ---- 12. Adversarial tests -------------------------------------------------
    print("\n[12] Adversarial tests...")
    rng = np.random.default_rng(13)
    adversarial = {}
    shuf = sector_dev.copy()
    shuf["realized_vol_20d_shuffled"] = rng.permutation(shuf["realized_vol_20d"].values)
    adversarial["shuffled_sector_vol_ic"] = _ic(shuf, "realized_vol_20d_shuffled")[0]

    shifted = sector_dev.sort_values(["sector", "date"]).copy()
    shifted["realized_vol_20d_shifted"] = shifted.groupby("sector")["realized_vol_20d"].shift(24)
    adversarial["time_shifted_sector_vol_ic"] = _ic(shifted, "realized_vol_20d_shifted")[0]

    perm = sector_dev.copy()
    perm["sector_shuffled"] = rng.permutation(perm["sector"].values)
    perm_spread = _topn_bottomn_spread(perm.rename(columns={"sector": "sector_orig", "sector_shuffled": "sector"}), "realized_vol_20d", n=3)
    adversarial["sector_label_permuted_top3_bottom3_spread"] = perm_spread.get("mean_top_minus_bottom_pct")

    real_spread = rotation_results["volatility_only_top3"]["spread"].get("mean_top_minus_bottom_pct")
    adversarial["real_top3_bottom3_spread_for_comparison"] = real_spread
    report["adversarial_tests"] = adversarial
    print(f"  {adversarial}")
    log_stat_experiment("phase4_adversarial_tests", "phase4", adversarial)

    # ---- 13. Golden holdout: ONE frozen confirmation ---------------------------
    print("\n[13] Golden holdout — evaluated once, on the best dev-set rotation config...")
    dev_only_results = {k: v for k, v in rotation_results.items()}
    best_key = max(dev_only_results, key=lambda k: dev_only_results[k]["spread"].get("mean_top_minus_bench_pct") or -999)
    signal_used = {"volatility_only": "realized_vol_20d", "momentum_only": "ret_20d", "vol_plus_momentum": "vol_mom_combo"}
    best_signal_label, best_n = best_key.rsplit("_top", 1)
    holdout_combo = sector_holdout.dropna(subset=["realized_vol_20d", "ret_20d", "fwd_ret_20d"]).copy()
    holdout_combo["vol_rank"] = holdout_combo.groupby("date")["realized_vol_20d"].rank(pct=True)
    holdout_combo["mom_rank"] = holdout_combo.groupby("date")["ret_20d"].rank(pct=True)
    holdout_combo["vol_mom_combo"] = holdout_combo["vol_rank"] + holdout_combo["mom_rank"]
    holdout_spread = _topn_bottomn_spread(holdout_combo, signal_used[best_signal_label], n=int(best_n))
    golden = {
        "best_dev_config": best_key,
        "holdout_period": [str(pd.Timestamp(sector_holdout["date"].min()).date()), str(pd.Timestamp(sector_holdout["date"].max()).date())],
        "holdout_result": holdout_spread,
        "note": "Frozen — evaluated exactly once. Do not re-tune against this.",
    }
    report["golden_holdout"] = golden
    print(f"  best dev config: {best_key} -> holdout: {holdout_spread}")
    log_stat_experiment("phase4_golden_holdout", "phase4", golden)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "phase4_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull report written to {ARTIFACT_DIR / 'phase4_report.json'}")
    return report


if __name__ == "__main__":
    main()
