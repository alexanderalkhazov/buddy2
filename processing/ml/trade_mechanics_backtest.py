"""Walk-forward backtest of the actual TRADE MECHANICS this report hands a
reader — not sector-level AUC/Brier again, but "if you had taken the
specific ticker this system's ranking would have picked, on the specific
ATR-based entry/stop/TP1 it computes, did the trade actually work?"

This has never been measured anywhere else in this codebase. Every other
metric in this project (AUC, Brier, calibration) evaluates the SECTOR
probability alone. This evaluates the full chain a reader actually acts
on: sector_prediction_model's OOS sector pick -> the highest-technical-
composite member ticker in that sector (mirroring ui/market_report.py's
long_score ranking, since nudge is monotonic in technical_composite within
a fixed sector) -> processing/scoring.py's ATR-based entry/stop/TP1/TP2.

Point-in-time discipline: for each OOS test date in each walk-forward fold
(the SAME folds/embargo sector_prediction_model.py:evaluate() uses — this
reuses its trained per-fold models rather than retraining), the picked
ticker's own technical indicators are computed from an OHLCV series
TRUNCATED to that date (never later bars) — the same truncation-invariance
guarantee tests/test_ml_leakage.py checks for the indicator layer.

Resolution rule, walking forward day-by-day from entry: a day whose LOW
touches the stop is a loss (checked first — conservative: if a single day's
range could plausibly touch both stop and TP1, this assumes the worse
outcome happens first, never the better one); otherwise a day whose HIGH
touches TP1 is a win. A trade that resolves neither way within
MAX_HOLD_DAYS trading days is a TIMEOUT, not silently dropped or counted
either way — this project's report explicitly does not know this system's
horizon guarantees an outcome by any particular date.
"""

from __future__ import annotations

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
from processing.ml.universe import EXPANDED_UNIVERSE_V2
from storage.cache import get_ohlcv

MAX_HOLD_DAYS = 30  # trading days — comfortably past the 20D horizon the sector label targets, to give a real TP1/stop resolution a chance without holding forever
REPORT_PATH = ARTIFACT_DIR / "trade_mechanics_backtest_report.json"

# enrich() is computed ONCE per ticker over its FULL cached history and
# memoized here, then indexed by date for every (date, ticker) point-in-time
# lookup this module needs — hundreds of them across the backtest. This is
# still exactly point-in-time-safe (never a lookahead) because every
# indicator in indicators.py is a backward-only rolling()/ewm()/shift()
# computation: row t's value is identical whether the frame is truncated to
# t or computed on the full series (see tests/test_ml_leakage.py's
# truncation-invariance test), so truncating before calling enrich() every
# time was redundant work, not a correctness requirement. Doing it this way
# turns ~3,600 full-history indicator computations into ~100 (one per
# distinct ticker actually used).
_ENRICHED_CACHE: dict[str, pd.DataFrame] = {}


def _get_enriched(ticker: str) -> pd.DataFrame | None:
    if ticker not in _ENRICHED_CACHE:
        try:
            _ENRICHED_CACHE[ticker] = enrich(get_ohlcv(ticker))
        except Exception:
            _ENRICHED_CACHE[ticker] = None
    return _ENRICHED_CACHE[ticker]


def _technical_composite_asof(ticker: str, as_of: pd.Timestamp) -> tuple[float | None, float | None, float | None]:
    """Point-in-time technical_composite, last close, and ATR14 for `ticker`
    as of the most recent bar <= `as_of` — never a later bar. Returns
    (None, None, None) if the ticker has no usable data at that date."""
    enriched = _get_enriched(ticker)
    if enriched is None:
        return None, None, None
    truncated = enriched[enriched.index <= as_of]
    if len(truncated) < 60:
        return None, None, None
    valid = truncated.dropna(subset=["close"])
    if valid.empty:
        return None, None, None
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
    return tc, close, atr14


def _resolve_trade(ticker: str, entry_date: pd.Timestamp, entry: float, stop: float, tp1: float, direction: str) -> dict:
    """Walk forward from entry_date (exclusive) up to MAX_HOLD_DAYS trading
    days, checking the real daily high/low against stop/TP1. Returns
    {"outcome": "WIN"|"LOSS"|"TIMEOUT", "days_held": int|None}."""
    try:
        df = get_ohlcv(ticker)
    except Exception:
        return {"outcome": "NO_DATA", "days_held": None}
    future = df[df.index > entry_date].iloc[:MAX_HOLD_DAYS]
    if future.empty:
        return {"outcome": "NO_DATA", "days_held": None}

    for i, (date, bar) in enumerate(future.iterrows(), start=1):
        low, high = float(bar["low"]), float(bar["high"])
        if direction == "long":
            stop_hit = low <= stop
            tp1_hit = high >= tp1
        else:
            stop_hit = high >= stop
            tp1_hit = low <= tp1
        if stop_hit:  # checked first — conservative, see module docstring
            return {"outcome": "LOSS", "days_held": i}
        if tp1_hit:
            return {"outcome": "WIN", "days_held": i}
    return {"outcome": "TIMEOUT", "days_held": len(future)}


def run_backtest(direction: str = "long", max_dates_per_fold: int | None = None) -> dict:
    """The full walk-forward trade-mechanics backtest. direction: "long" or
    "short" — mirrors ui/market_report.py's LONG/SHORT candidate selection
    (top-3 predicted sectors for long, bottom-3 for short is NOT modeled
    here since sector_prediction_model was only trained/validated as a
    top-3 predictor — see market_report.py's own scope note on this).
    max_dates_per_fold caps the backtest for a quick smoke-test run; leave
    None for the full, honest sample."""
    df, extra_cols = _build_dataset()
    feature_cols = _feature_set(extra_cols)
    data = df.dropna(subset=[*feature_cols, "is_top_k"]).reset_index(drop=True)
    splits = purged_walk_forward_splits(data["date"], n_folds=4, embargo_days=EMBARGO_DAYS)

    trades = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
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

        unique_dates = sorted(test["date"].unique())
        if max_dates_per_fold:
            unique_dates = unique_dates[:max_dates_per_fold]

        for as_of in unique_dates:
            day_rows = test[test["date"] == as_of]
            ranked = day_rows.sort_values("p_ensemble", ascending=(direction == "short"))
            picked_sectors = ranked.head(TOP_K)["sector"].tolist()

            for sector in picked_sectors:
                candidates = []
                for ticker in EXPANDED_UNIVERSE_V2[sector]:
                    tc, close, atr14 = _technical_composite_asof(ticker, pd.Timestamp(as_of))
                    if tc is None or close is None or atr14 is None or atr14 <= 0:
                        continue
                    candidates.append((tc, ticker, close, atr14))
                if not candidates:
                    continue
                # Highest technical_composite for long, lowest for short —
                # mirrors long_score/short_score's monotonic-in-technical_
                # composite ranking within an already-fixed sector.
                candidates.sort(key=lambda c: c[0], reverse=(direction == "long"))
                tc, ticker, close, atr14 = candidates[0]

                levels = scoring.trade_levels(close, atr14, direction=direction)
                if "error" in levels:
                    continue
                resolution = _resolve_trade(ticker, pd.Timestamp(as_of), close, levels["stop"], levels["take_profit_1"], direction)
                trades.append(
                    {
                        "fold": fold_i, "date": str(pd.Timestamp(as_of).date()), "sector": sector, "ticker": ticker,
                        "entry": close, "stop": levels["stop"], "tp1": levels["take_profit_1"],
                        "technical_composite": tc, "outcome": resolution["outcome"], "days_held": resolution["days_held"],
                    }
                )

    resolved = [t for t in trades if t["outcome"] in ("WIN", "LOSS")]
    n_win = sum(1 for t in resolved if t["outcome"] == "WIN")
    n_timeout = sum(1 for t in trades if t["outcome"] == "TIMEOUT")
    n_no_data = sum(1 for t in trades if t["outcome"] == "NO_DATA")
    win_rate = round(n_win / len(resolved), 4) if resolved else None
    # TP1/stop are 1.5R/1R by construction (scoring.trade_levels() defaults) —
    # expectancy in R-multiples per resolved trade, computed here rather than
    # left as an exercise: win_rate*1.5R - (1-win_rate)*1R. Positive means the
    # mechanics alone (ignoring costs/slippage/gap risk) would have made
    # money on this sample; negative means they would have lost money even
    # before those real-world frictions are added.
    tp1_r, stop_r = 1.5, 1.0
    expectancy_r = round(win_rate * tp1_r - (1 - win_rate) * stop_r, 4) if win_rate is not None else None

    return {
        "direction": direction,
        "n_trades_attempted": len(trades),
        "n_resolved": len(resolved),
        "n_wins": n_win,
        "n_losses": len(resolved) - n_win,
        "n_timeout": n_timeout,
        "n_no_data": n_no_data,
        "win_rate_of_resolved": win_rate,
        "expectancy_r_per_trade": expectancy_r,
        "mean_days_held_resolved": round(float(np.mean([t["days_held"] for t in resolved])), 2) if resolved else None,
        "note": (
            "win_rate_of_resolved = wins / (wins + losses), EXCLUDING timeouts (a trade that never touched "
            "stop or TP1 within MAX_HOLD_DAYS trading days) and NO_DATA (ticker lacked enough history at that "
            "date). expectancy_r_per_trade = win_rate*1.5R - (1-win_rate)*1R (TP1/stop's R-multiples by "
            "construction) — POSITIVE means this system's exact entry/stop/TP1 mechanics would have made "
            "money on this sample BEFORE costs; NEGATIVE means they would have lost money even before real- "
            "world frictions are added. No transaction costs, slippage, or the possibility that a gap jumps "
            "past the stop for a worse fill than assumed here are modeled — real expectancy is worse than "
            "this number, never better."
        ),
        "trades": trades,
    }


def run_and_save() -> dict:
    """Run both directions, persist a permanent report (like the Phase 1-5
    JSON reports), and log to the never-overwritten experiment registry —
    so this result is re-derivable and citable the same way every other
    validated claim in this project is, not just printed once and lost."""
    import json

    report = {"long": run_backtest(direction="long"), "short": run_backtest(direction="short")}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    log_stat_experiment(
        "trade_mechanics_backtest", "trade_mechanics_backtest",
        {k: {kk: vv for kk, vv in v.items() if kk != "trades"} for k, v in report.items()},
        notes="Walk-forward backtest of the ranking rule's actual entry/stop/TP1 mechanics, not just sector AUC.",
    )
    return report


if __name__ == "__main__":
    import json

    result = run_and_save()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "trades"} for k, v in result.items()}, indent=2))
