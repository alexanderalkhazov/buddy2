"""Backtesting: turn a declarative strategy spec into vectorbt signals and metrics.

The LLM never emits Python. It emits a StrategySpec (entry/exit rules over named
indicator columns), which this module compiles to boolean signal arrays. That keeps
proposed strategies executable, inspectable, and safe to run.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from processing.indicators import enrich
from storage import cache

OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "crosses_above": "crosses_above",
    "crosses_below": "crosses_below",
}

# Columns a rule may reference — everything enrich() produces, plus raw OHLCV.
ALLOWED_FIELDS = {
    "open", "high", "low", "close", "volume", "rsi14", "rsi2", "macd", "signal", "hist",
    "sma20", "sma50", "sma200", "bb_upper", "bb_lower", "bb_mid", "bb_pctb", "atr14",
    "vol_avg30", "stoch_k", "stoch_d", "adx14", "donchian_upper20", "donchian_lower10",
    "roc10", "obv",
}


@dataclass
class Rule:
    left: str
    op: str
    right: Any  # a field name or a number

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.left not in ALLOWED_FIELDS:
            raise ValueError(f"unknown field {self.left!r}")
        left = df[self.left]

        if isinstance(self.right, str):
            if self.right not in ALLOWED_FIELDS:
                raise ValueError(f"unknown field {self.right!r}")
            right = df[self.right]
        else:
            right = float(self.right)

        if self.op in ("crosses_above", "crosses_below"):
            prev_left = left.shift()
            prev_right = right.shift() if isinstance(right, pd.Series) else right
            if self.op == "crosses_above":
                return (left > right) & (prev_left <= prev_right)
            return (left < right) & (prev_left >= prev_right)

        if self.op not in OPS:
            raise ValueError(f"unsupported operator {self.op!r}")
        return OPS[self.op](left, right)


@dataclass
class StrategySpec:
    name: str
    entry_rules: list[Rule] = field(default_factory=list)
    exit_rules: list[Rule] = field(default_factory=list)
    entry_logic: str = "all"  # "all" (AND) or "any" (OR)
    exit_logic: str = "any"
    stop_loss_pct: float | None = None  # fraction, e.g. 0.08 = 8% stop
    take_profit_pct: float | None = None
    trailing_stop: bool = False  # if True, stop_loss_pct trails the peak instead of the entry

    @classmethod
    def from_dict(cls, data: dict) -> "StrategySpec":
        def _float_or_none(value) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _bool(value) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes")
            return bool(value)

        return cls(
            name=data.get("name", "unnamed"),
            entry_rules=[Rule(**r) for r in data.get("entry_rules", [])],
            exit_rules=[Rule(**r) for r in data.get("exit_rules", [])],
            entry_logic=data.get("entry_logic", "all"),
            exit_logic=data.get("exit_logic", "any"),
            stop_loss_pct=_float_or_none(data.get("stop_loss_pct")),
            take_profit_pct=_float_or_none(data.get("take_profit_pct")),
            trailing_stop=_bool(data.get("trailing_stop", False)),
        )


def _combine(signals: list[pd.Series], logic: str, index: pd.Index) -> pd.Series:
    if not signals:
        return pd.Series(False, index=index)
    stacked = pd.concat(signals, axis=1).fillna(False)
    return stacked.all(axis=1) if logic == "all" else stacked.any(axis=1)


def run(
    ticker: str,
    spec: StrategySpec | dict,
    init_cash: float = 10_000,
    fees: float = 0.001,
    slippage: float = 0.0005,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Backtest a spec against cached history and report the headline risk metrics.

    start_date/end_date (inclusive, "YYYY-MM-DD") restrict which bars are actually
    traded, while indicators are still computed from the FULL cached history first —
    so a strategy using e.g. sma200 keeps a correct warm-up window even when the
    trading window starts partway through. Used by run_out_of_sample() to split one
    continuous history into an in-sample fitting period and an out-of-sample holdout
    without recomputing indicators on a truncated series (which would shift SMA/RSI
    values near the start of the holdout window).

    slippage defaults to 0.05% per trade (vectorbt's slippage param, applied as an
    adverse price move on both entry and exit) — a conservative, non-zero default so
    a strategy's edge isn't silently overstated by assuming perfect fills. Set to 0.0
    to match the old zero-slippage behavior.
    """
    if isinstance(spec, dict):
        spec = StrategySpec.from_dict(spec)
    if not spec.entry_rules:
        raise ValueError("strategy needs at least one entry rule")

    df = enrich(cache.get_ohlcv(ticker)).dropna(subset=["close"])
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    if df.empty:
        raise ValueError(f"no bars in range for {ticker!r} ({start_date} to {end_date})")

    entries = _combine([r.evaluate(df) for r in spec.entry_rules], spec.entry_logic, df.index)
    exits = _combine([r.evaluate(df) for r in spec.exit_rules], spec.exit_logic, df.index)

    risk_kwargs: dict = {}
    if spec.stop_loss_pct:
        risk_kwargs["sl_stop"] = spec.stop_loss_pct
        risk_kwargs["sl_trail"] = bool(spec.trailing_stop)
    if spec.take_profit_pct:
        risk_kwargs["tp_stop"] = spec.take_profit_pct

    portfolio = vbt.Portfolio.from_signals(
        df["close"], entries, exits, init_cash=init_cash, fees=fees, slippage=slippage, freq="1D", **risk_kwargs
    )

    trades = portfolio.trades
    n_trades = int(trades.count())

    def _safe(value) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(out) else round(out, 4)

    # A fact about the number, not a confidence judgment about what it means — "is
    # 30 trades enough to trust a 50% win rate" is a statistical question with a real
    # answer (a binomial proportion at n=30 has a wide confidence interval), but this
    # system doesn't compute that interval, so it states the count and lets the
    # standard error be seen in the spread between avg_win/avg_loss instead of
    # asserting a verdict ("large enough") that isn't actually computed.
    sample_note = f"{n_trades} trades in this backtest. No confidence interval is computed here — treat the win rate and Sharpe as a point estimate from a single historical window, not a validated statistic."

    if n_trades > 0:
        expectancy = _safe(trades.expectancy())
        profit_factor = _safe(trades.profit_factor())
        avg_win = _safe(trades.winning.pnl.mean()) if trades.winning.count() else None
        avg_loss = _safe(trades.losing.pnl.mean()) if trades.losing.count() else None
        avg_win_pct = _safe(trades.winning.returns.mean() * 100) if trades.winning.count() else None
        avg_loss_pct = _safe(trades.losing.returns.mean() * 100) if trades.losing.count() else None
    else:
        expectancy = profit_factor = avg_win = avg_loss = avg_win_pct = avg_loss_pct = None

    days = max((df.index.max() - df.index.min()).days, 1)
    strategy_cagr = _safe(portfolio.annualized_return() * 100)
    buy_hold_total = float(df["close"].iloc[-1] / df["close"].iloc[0])
    buy_hold_cagr = _safe((buy_hold_total ** (365.25 / days) - 1) * 100)
    total_return_pct = _safe(portfolio.total_return() * 100)
    buy_hold_return_pct = _safe((buy_hold_total - 1) * 100)
    return_difference_vs_buy_hold_pct = (
        round(total_return_pct - buy_hold_return_pct, 4)
        if total_return_pct is not None and buy_hold_return_pct is not None
        else None
    )

    return {
        "strategy": spec.name,
        "ticker": ticker.upper(),
        "period": f"{df.index.min().date()} to {df.index.max().date()}",
        "trades": n_trades,
        "total_return_pct": total_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        # Absolute total return is misleading across different time windows and
        # exposure levels — CAGR is the comparable figure. Positive
        # return_difference_vs_buy_hold_pct means the strategy beat buy-and-hold on
        # total return; this is NOT risk-adjusted alpha (no beta/CAPM regression is
        # computed here) — never call it "alpha".
        "strategy_cagr_pct": strategy_cagr,
        "buy_hold_cagr_pct": buy_hold_cagr,
        "return_difference_vs_buy_hold_pct": return_difference_vs_buy_hold_pct,
        "outperformed_buy_and_hold": (
            return_difference_vs_buy_hold_pct > 0 if return_difference_vs_buy_hold_pct is not None else None
        ),
        "sharpe_ratio": _safe(portfolio.sharpe_ratio()),
        "sortino_ratio": _safe(portfolio.sortino_ratio()),
        "annualized_volatility_pct": _safe(portfolio.annualized_volatility() * 100),
        "time_in_market_pct": _safe(portfolio.gross_exposure().mean() * 100),
        "max_drawdown_pct": _safe(portfolio.max_drawdown() * 100),
        "win_rate_pct": _safe(trades.win_rate() * 100) if n_trades else None,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "stop_loss_pct": _safe(spec.stop_loss_pct * 100) if spec.stop_loss_pct else None,
        "trailing_stop": spec.trailing_stop,
        "take_profit_pct": _safe(spec.take_profit_pct * 100) if spec.take_profit_pct else None,
        # The reported max_drawdown_pct is computed WITH the stop already applied in
        # the simulation (vectorbt's sl_stop) when stop_loss_pct is set — it is not a
        # raw unstopped drawdown with a hypothetical stop bolted on afterward.
        "drawdown_reflects_stop_loss": bool(spec.stop_loss_pct),
        "sample_size_note": sample_note,
        "fees_pct": round(fees * 100, 4),
        "slippage_pct": round(slippage * 100, 4),
        "caveat": (
            f"Backtest on a single historical window ({df.index.min().date()} to "
            f"{df.index.max().date()}) with {fees * 100:.2f}% fees and {slippage * 100:.2f}% "
            "slippage modeled per trade (an adverse price move on both entry and exit, "
            "not a real order book simulation — still a simplification, not a claim of "
            "exact live-fill accuracy). Not evidence of forward performance; beware "
            "overfitting to this period — see run_out_of_sample for a train/holdout split "
            "that checks whether an edge found on one window persists on unseen data. "
            "total_return_pct and CAGR both compare the SAME asset (the ticker vs. its "
            "own buy-and-hold), not a market index — neither figure is beta-adjusted alpha."
        ),
    }


def run_template(ticker: str, template: str, **overrides) -> dict:
    """Run one of the built-in named strategies, with optional field overrides
    (e.g. stop_loss_pct) merged on top of the template's defaults."""
    from processing.strategies import get_spec

    spec = {**get_spec(template), **overrides}
    return run(ticker, spec)


def run_out_of_sample(ticker: str, spec: StrategySpec | dict, split_pct: float = 0.7, **kwargs) -> dict:
    """Split the cached history into an in-sample fitting window and an out-of-
    sample holdout, and run the same spec independently on each. A strategy that
    looks strong in-sample but falls apart out-of-sample is a warning sign of
    curve-fitting to that specific historical window — this is a sanity check, not
    a guarantee the out-of-sample result predicts future performance either; it's
    still one more single historical window, just one the strategy wasn't tuned on.
    """
    if not 0.1 <= split_pct <= 0.9:
        raise ValueError("split_pct must be between 0.1 and 0.9")

    df = cache.get_ohlcv(ticker).dropna(subset=["close"])
    if df.empty:
        raise ValueError(f"no price data for {ticker!r}")

    split_idx = int(len(df) * split_pct)
    split_date = str(df.index[split_idx].date())

    in_sample = run(ticker, spec, end_date=split_date, **kwargs)
    out_of_sample = run(ticker, spec, start_date=split_date, **kwargs)

    in_sharpe = in_sample.get("sharpe_ratio")
    out_sharpe = out_of_sample.get("sharpe_ratio")
    degraded = None
    if in_sharpe is not None and out_sharpe is not None:
        degraded = out_sharpe < in_sharpe * 0.5 or (in_sharpe > 0 and out_sharpe <= 0)

    return {
        "ticker": ticker.upper(),
        "strategy": in_sample.get("strategy"),
        "split_date": split_date,
        "split_pct": split_pct,
        "in_sample": in_sample,
        "out_of_sample": out_of_sample,
        "sharpe_degraded_out_of_sample": degraded,
        "note": (
            "in_sample is the strategy's own fitting window; out_of_sample is unseen data "
            "the strategy wasn't shaped by. sharpe_degraded_out_of_sample is a rough, fixed "
            "heuristic (out-of-sample Sharpe < half the in-sample Sharpe, or flips from "
            "positive to non-positive) — not a statistical significance test. A degraded "
            "result is a real warning sign about overfitting; a preserved result is "
            "reassuring but still not proof of a persistent edge, since both windows are "
            "still just one ticker's one historical path."
        ),
    }


def scan(ticker: str) -> list[dict]:
    """Run every built-in strategy template against one ticker and rank by Sharpe.

    This is a screen, not a recommendation — it surfaces which published, well-known
    systems currently show a statistical edge on this ticker's history so the model
    can reason about *why*, rather than inventing an ad hoc rule from scratch.

    Returns a compact ranking view, not the full report per template (that would
    repeat the same caveat/period text N times and risks blowing a small model's
    per-request token limit) — call run_backtest with the chosen template for the
    full field set once a candidate is picked.
    """
    from processing.strategies import TEMPLATES

    _RANKING_FIELDS = (
        "sharpe_ratio",
        "sortino_ratio",
        "strategy_cagr_pct",
        "buy_hold_cagr_pct",
        "outperformed_buy_and_hold",
        "max_drawdown_pct",
        "win_rate_pct",
        "profit_factor",
        "trades",
    )

    results = []
    for name in TEMPLATES:
        try:
            full = run_template(ticker, name)
            results.append({"template": name, **{k: full.get(k) for k in _RANKING_FIELDS}})
        except Exception as exc:
            results.append({"template": name, "error": f"{type(exc).__name__}: {exc}"})

    results.sort(key=lambda r: (r.get("sharpe_ratio") is None, -(r.get("sharpe_ratio") or 0)))
    return results
