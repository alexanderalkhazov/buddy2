"""Research: can the SHORT selection rule be improved, rather than just
gated away? trade_mechanics_backtest.py measured the CURRENT short
mechanics (lowest-technical-composite ticker in the bottom-3-predicted
sectors) at -0.21R/trade expectancy — genuinely negative, which is why
every SHORT candidate in the report shows VERDICT: AVOID. That gate is
correct given the current selection rule; this script asks whether a
DIFFERENT selection rule, using the same sectors/mechanics/backtest
discipline, would actually work.

Hypothesis being tested: shorting the LOWEST-technical-composite (most
oversold/beaten-down) ticker in a weak sector may be selecting exactly the
names most prone to a mean-reversion bounce (short squeeze risk) — a
well-documented pattern. An alternative is to short the HIGHEST-technical-
composite ticker in a weak sector instead (still-elevated/overextended
strength rolling over in a structurally weak sector), or to exclude
already-oversold names (RSI14 < 35) from the candidate pool before
picking.

Same walk-forward folds, same point-in-time indicator truncation, same
resolution rule, same cost tiers as trade_mechanics_backtest.py — only the
SELECTION rule changes between variants tested here. Whichever variant (if
any) demonstrates genuinely positive expectancy gets proposed for
production; none of this changes production on its own.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from processing import scoring
from processing.indicators import enrich, ichimoku_cloud_position, technical_composite_score
from processing.ml.registry import ARTIFACT_DIR, log_stat_experiment
from processing.ml.sector_prediction_model import EMBARGO_DAYS, TOP_K, _build_dataset, _feature_set
from processing.ml.train import purged_walk_forward_splits
from processing.ml.trade_mechanics_backtest import COST_TIERS_BPS, MAX_HOLD_DAYS
from processing.ml.universe import EXPANDED_UNIVERSE_V2
from storage.cache import get_ohlcv

_ENRICHED_CACHE: dict[str, pd.DataFrame] = {}


def _get_enriched(ticker: str) -> pd.DataFrame | None:
    if ticker not in _ENRICHED_CACHE:
        try:
            _ENRICHED_CACHE[ticker] = enrich(get_ohlcv(ticker))
        except Exception:
            _ENRICHED_CACHE[ticker] = None
    return _ENRICHED_CACHE[ticker]


def _snapshot_asof(ticker: str, as_of: pd.Timestamp) -> dict | None:
    """Point-in-time technical_composite, close, ATR14, AND rsi14 (the extra
    field trade_mechanics_backtest.py's version doesn't expose, needed here
    to test an RSI-based oversold filter) — never a later bar."""
    enriched = _get_enriched(ticker)
    if enriched is None:
        return None
    truncated = enriched[enriched.index <= as_of]
    if len(truncated) < 60:
        return None
    valid = truncated.dropna(subset=["close"])
    if valid.empty:
        return None
    row = valid.iloc[-1]
    close = float(row["close"])
    cloud = ichimoku_cloud_position(
        close,
        None if pd.isna(row.get("ichimoku_senkou_a")) else float(row["ichimoku_senkou_a"]),
        None if pd.isna(row.get("ichimoku_senkou_b")) else float(row["ichimoku_senkou_b"]),
    )
    tc = technical_composite_score(
        rsi14=None if pd.isna(row.get("rsi14")) else float(row["rsi14"]),
        stoch_k=None if pd.isna(row.get("stoch_k")) else float(row["stoch_k"]),
        adx14=None if pd.isna(row.get("adx14")) else float(row["adx14"]),
        bb_pctb=None if pd.isna(row.get("bb_pctb")) else float(row["bb_pctb"]),
        roc10=None if pd.isna(row.get("roc10")) else float(row["roc10"]),
        macd_hist=None if pd.isna(row.get("hist")) else float(row["hist"]),
        cci20=None if pd.isna(row.get("cci20")) else float(row["cci20"]),
        ichimoku_cloud=cloud,
        mfi14=None if pd.isna(row.get("mfi14")) else float(row["mfi14"]),
        aroon_oscillator=None if pd.isna(row.get("aroon_oscillator")) else float(row["aroon_oscillator"]),
        keltner_pctk=None if pd.isna(row.get("keltner_pctk")) else float(row["keltner_pctk"]),
    )
    atr14 = None if pd.isna(row.get("atr14")) else float(row["atr14"])
    rsi14 = None if pd.isna(row.get("rsi14")) else float(row["rsi14"])
    if tc is None or atr14 is None or atr14 <= 0:
        return None
    return {"tc": tc, "close": close, "atr14": atr14, "rsi14": rsi14}


def _resolve_trade(ticker: str, entry_date: pd.Timestamp, stop: float, tp1: float, direction: str) -> dict:
    try:
        df = get_ohlcv(ticker)
    except Exception:
        return {"outcome": "NO_DATA", "days_held": None}
    future = df[df.index > entry_date].iloc[:MAX_HOLD_DAYS]
    if future.empty:
        return {"outcome": "NO_DATA", "days_held": None}
    for i, (_, bar) in enumerate(future.iterrows(), start=1):
        low, high = float(bar["low"]), float(bar["high"])
        stop_hit = high >= stop if direction == "short" else low <= stop
        tp1_hit = low <= tp1 if direction == "short" else high >= tp1
        if stop_hit:
            return {"outcome": "LOSS", "days_held": i}
        if tp1_hit:
            return {"outcome": "WIN", "days_held": i}
    return {"outcome": "TIMEOUT", "days_held": len(future)}


# --- Selection strategies (SHORT only — LONG is already positive) ---------
def _select_lowest_tc(candidates: list[dict]) -> dict | None:
    """BASELINE — current production rule: lowest technical_composite."""
    return min(candidates, key=lambda c: c["tc"]) if candidates else None


def _select_highest_tc(candidates: list[dict]) -> dict | None:
    """ALTERNATIVE A: highest technical_composite — short still-elevated
    strength in a structurally weak sector, not an already-crashed name."""
    return max(candidates, key=lambda c: c["tc"]) if candidates else None


def _select_lowest_tc_not_oversold(candidates: list[dict]) -> dict | None:
    """ALTERNATIVE B: lowest technical_composite, but excluding RSI14<35
    (already-oversold names, the ones most prone to a mean-reversion
    bounce / short squeeze) from the pool first."""
    pool = [c for c in candidates if c["rsi14"] is None or c["rsi14"] >= 35]
    pool = pool or candidates  # fall back to full pool if the filter empties it
    return min(pool, key=lambda c: c["tc"]) if pool else None


STRATEGIES = {
    "baseline_lowest_tc": _select_lowest_tc,
    "highest_tc": _select_highest_tc,
    "lowest_tc_not_oversold": _select_lowest_tc_not_oversold,
}


def run_strategy(strategy_name: str) -> dict:
    select_fn = STRATEGIES[strategy_name]
    df, extra_cols = _build_dataset()
    feature_cols = _feature_set(extra_cols)
    data = df.dropna(subset=[*feature_cols, "is_top_k"]).reset_index(drop=True)
    splits = purged_walk_forward_splits(data["date"], n_folds=4, embargo_days=EMBARGO_DAYS)

    trades = []
    for train_idx, test_idx in splits:
        X_train, y_train = data.loc[train_idx, feature_cols], data.loc[train_idx, "is_top_k"].values
        if len(set(y_train)) < 2:
            continue
        scaler = StandardScaler().fit(X_train)
        Xtr = scaler.transform(X_train)
        hgb = CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=4, max_iter=150), method="sigmoid", cv=3).fit(Xtr, y_train)
        lr = CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight="balanced"), method="sigmoid", cv=3).fit(Xtr, y_train)

        test = data.loc[test_idx].copy()
        Xte = scaler.transform(test[feature_cols])
        test["p_ensemble"] = (hgb.predict_proba(Xte)[:, 1] + lr.predict_proba(Xte)[:, 1]) / 2

        for as_of in sorted(test["date"].unique()):
            day_rows = test[test["date"] == as_of]
            ranked = day_rows.sort_values("p_ensemble", ascending=True)  # short: lowest predicted P(top-3) first
            picked_sectors = ranked.head(TOP_K)["sector"].tolist()

            for sector in picked_sectors:
                candidates = []
                for ticker in EXPANDED_UNIVERSE_V2[sector]:
                    snap = _snapshot_asof(ticker, pd.Timestamp(as_of))
                    if snap is not None:
                        candidates.append({**snap, "ticker": ticker})
                picked = select_fn(candidates)
                if picked is None:
                    continue
                levels = scoring.trade_levels(picked["close"], picked["atr14"], direction="short")
                if "error" in levels:
                    continue
                resolution = _resolve_trade(picked["ticker"], pd.Timestamp(as_of), levels["stop"], levels["take_profit_1"], "short")
                trades.append({
                    "entry": picked["close"], "stop": levels["stop"], "outcome": resolution["outcome"],
                    "days_held": resolution["days_held"],
                })

    resolved = [t for t in trades if t["outcome"] in ("WIN", "LOSS")]
    n_win = sum(1 for t in resolved if t["outcome"] == "WIN")
    win_rate = round(n_win / len(resolved), 4) if resolved else None
    tp1_r, stop_r = 1.5, 1.0
    expectancy_r = round(win_rate * tp1_r - (1 - win_rate) * stop_r, 4) if win_rate is not None else None

    cost_adjusted = {}
    for tier, bps in COST_TIERS_BPS.items():
        per_trade_r = []
        for t in resolved:
            risk_per_share = abs(t["entry"] - t["stop"])
            if risk_per_share <= 0:
                continue
            cost_r = (bps / 10_000 * t["entry"]) / risk_per_share
            per_trade_r.append((tp1_r if t["outcome"] == "WIN" else -stop_r) - cost_r)
        cost_adjusted[tier] = round(float(np.mean(per_trade_r)), 4) if per_trade_r else None

    return {
        "strategy": strategy_name,
        "n_trades_attempted": len(trades),
        "n_resolved": len(resolved),
        "n_wins": n_win,
        "n_losses": len(resolved) - n_win,
        "n_timeout": sum(1 for t in trades if t["outcome"] == "TIMEOUT"),
        "win_rate_of_resolved": win_rate,
        "expectancy_r_per_trade": expectancy_r,
        "expectancy_r_per_trade_cost_adjusted_base": cost_adjusted.get("BASE"),
    }


def run() -> dict:
    results = {name: run_strategy(name) for name in STRATEGIES}
    baseline_r = results["baseline_lowest_tc"]["expectancy_r_per_trade"]
    best_name = max(results, key=lambda n: (results[n]["expectancy_r_per_trade"] if results[n]["expectancy_r_per_trade"] is not None else -999))
    report = {
        "results": results,
        "baseline_expectancy_r": baseline_r,
        "best_strategy": best_name,
        "best_beats_baseline": results[best_name]["expectancy_r_per_trade"] > baseline_r if baseline_r is not None else None,
        "note": (
            "Tests whether the SHORT selection rule itself (not just the ATR stop/TP mechanics, already "
            "tested in trade_mechanics_backtest.py) can be improved. Only ships to production if a variant "
            "shows genuinely positive expectancy after the same scrutiny as every other feature this "
            "project has tested (not just 'better than baseline' — baseline is already negative)."
        ),
    }
    log_stat_experiment(
        "short_selection_strategy_research", "short_selection_research", report,
        notes="Tests alternative SHORT candidate selection rules against the baseline lowest-technical-composite rule.",
    )
    (ARTIFACT_DIR / "short_selection_research_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
