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


def _build_portfolio(
    ticker: str,
    spec: StrategySpec | dict,
    init_cash: float = 10_000,
    fees: float = 0.001,
    slippage: float = 0.0005,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Shared portfolio-construction logic behind run() and monte_carlo_bootstrap()
    — keeps both consistent (same signal evaluation, same risk_kwargs handling)
    without duplicating it. Returns (portfolio, spec, df)."""
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
    return portfolio, spec, df


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
    portfolio, spec, df = _build_portfolio(
        ticker, spec, init_cash=init_cash, fees=fees, slippage=slippage, start_date=start_date, end_date=end_date
    )

    trades = portfolio.trades
    n_trades = int(trades.count())

    def _safe(value) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        # inf/-inf shows up when e.g. a window has 0 trades or no losing trades
        # (profit_factor, Sharpe) — not a meaningful number, treat like NaN.
        return None if np.isnan(out) or np.isinf(out) else round(out, 4)

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


def run_regime_windows(ticker: str, spec: StrategySpec | dict, n_windows: int = 4, **kwargs) -> dict:
    """Splits the cached history into n_windows consecutive, roughly equal-length
    calendar windows and runs the same spec independently on each — instead of one
    combined 5-year backtest, this shows whether the strategy's edge held up across
    genuinely different sub-periods (e.g. a period where the stock was flat or
    falling vs. one where it ran hard), or whether the aggregate number is being
    carried by one exceptional window. Each window is labeled by its own realized
    buy-and-hold return (bull/bear/flat) and realized volatility (low/mid/high
    relative to the OTHER windows in this same run) — both computed from this
    ticker's own history, not calibrated against any external market classification.
    """
    if n_windows < 2:
        raise ValueError("n_windows must be at least 2")

    df = cache.get_ohlcv(ticker).dropna(subset=["close"])
    if df.empty:
        raise ValueError(f"no price data for {ticker!r}")

    boundaries = [df.index[min(int(len(df) * i / n_windows), len(df) - 1)] for i in range(n_windows + 1)]
    boundaries[-1] = df.index[-1]

    window_vols = []
    raw_windows = []
    for i in range(n_windows):
        start, end = boundaries[i], boundaries[i + 1]
        seg = df[(df.index >= start) & (df.index <= end)]
        if len(seg) < 5:
            continue
        daily_returns = seg["close"].pct_change().dropna()
        vol = float(daily_returns.std() * (252 ** 0.5) * 100) if len(daily_returns) > 1 else None
        bh_return = float(seg["close"].iloc[-1] / seg["close"].iloc[0] - 1) * 100
        raw_windows.append({"start": str(start.date()), "end": str(end.date()), "bh_return_pct": round(bh_return, 2), "vol": vol})
        if vol is not None:
            window_vols.append(vol)

    vol_terciles = sorted(window_vols)
    def _vol_label(v):
        if v is None or not vol_terciles:
            return "UNKNOWN"
        idx = vol_terciles.index(v)
        third = max(1, len(vol_terciles) // 3)
        if idx < third:
            return "LOW (relative to this ticker's other tested windows)"
        if idx >= len(vol_terciles) - third:
            return "HIGH (relative to this ticker's other tested windows)"
        return "MID (relative to this ticker's other tested windows)"

    results = []
    for w in raw_windows:
        try:
            r = run(ticker, spec, start_date=w["start"], end_date=w["end"], **kwargs)
        except Exception as exc:
            results.append({**w, "error": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(
            {
                **w,
                "market_regime_label": "bull" if w["bh_return_pct"] > 10 else "bear" if w["bh_return_pct"] < -10 else "flat",
                "volatility_label": _vol_label(w["vol"]),
                "trades": r.get("trades"),
                "sharpe_ratio": r.get("sharpe_ratio"),
                "strategy_cagr_pct": r.get("strategy_cagr_pct"),
                "outperformed_buy_and_hold": r.get("outperformed_buy_and_hold"),
                "max_drawdown_pct": r.get("max_drawdown_pct"),
                "win_rate_pct": r.get("win_rate_pct"),
            }
        )

    sharpes = [r["sharpe_ratio"] for r in results if r.get("sharpe_ratio") is not None]
    consistent = None
    if len(sharpes) >= 2:
        consistent = sum(1 for s in sharpes if s > 0) == len(sharpes) or sum(1 for s in sharpes if s <= 0) == len(sharpes)

    return {
        "ticker": ticker.upper(),
        "n_windows": len(results),
        "windows": results,
        "sharpe_consistent_sign_across_windows": consistent,
        "note": (
            "Each window is this ticker's OWN sub-period, not a labeled external bull/bear market — "
            "market_regime_label and volatility_label are both derived purely from this ticker's own "
            "price path in this run, not an independent market classification. "
            "sharpe_consistent_sign_across_windows is a rough flag: if Sharpe flips sign between "
            "windows, the aggregate backtest number may be dominated by one unusually strong or weak "
            "period rather than a repeatable pattern. Still not proof either way — each window remains "
            "a single historical path for a single ticker."
        ),
    }


def monte_carlo_bootstrap(ticker: str, spec: StrategySpec | dict, n_sims: int = 2000, seed: int | None = None, **kwargs) -> dict:
    """Bootstrap-resample this backtest's own trade sequence (with replacement) to
    see the range of outcomes that same set of trades could have produced in a
    different order — NOT a simulation of different trades, just different
    orderings/repetitions of the ones that actually occurred. This isolates
    path-dependency (a few early wins/losses skewing the single actual sequence)
    from the trades' underlying distribution, but it does NOT create new
    information: if the strategy only had 23 trades, this is still built from
    those same 23 trades' return distribution, resampled.
    """
    import numpy as np

    portfolio, resolved_spec, _df = _build_portfolio(ticker, spec, **kwargs)
    trades = portfolio.trades
    n_trades = int(trades.count())
    if n_trades < 5:
        return {"error": f"only {n_trades} trades — too few to bootstrap meaningfully (need at least 5)"}

    trade_returns = np.asarray(trades.returns.values, dtype=float)
    full = run(ticker, spec, **kwargs)

    rng = np.random.default_rng(seed)
    sim_totals = []
    sim_max_drawdowns = []
    for _ in range(n_sims):
        sample = rng.choice(trade_returns, size=n_trades, replace=True)
        equity = np.cumprod(1 + sample)
        sim_totals.append(equity[-1] - 1)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        sim_max_drawdowns.append(drawdown.min())

    sim_totals = np.array(sim_totals) * 100
    sim_max_drawdowns = np.array(sim_max_drawdowns) * 100
    prob_loss = float((sim_totals < 0).mean() * 100)

    return {
        "ticker": ticker.upper(),
        "strategy": full.get("strategy"),
        "n_trades": n_trades,
        "n_simulations": n_sims,
        "actual_total_return_pct": full.get("total_return_pct"),
        "bootstrap_total_return_pct": {
            "p5": round(float(np.percentile(sim_totals, 5)), 2),
            "median": round(float(np.percentile(sim_totals, 50)), 2),
            "p95": round(float(np.percentile(sim_totals, 95)), 2),
        },
        "bootstrap_max_drawdown_pct": {
            "p5_best_case": round(float(np.percentile(sim_max_drawdowns, 95)), 2),
            "median": round(float(np.percentile(sim_max_drawdowns, 50)), 2),
            "p95_worst_case": round(float(np.percentile(sim_max_drawdowns, 5)), 2),
        },
        "probability_of_loss_pct": round(prob_loss, 1),
        "note": (
            f"Resamples this strategy's own {n_trades} historical trade returns (with replacement, "
            f"{n_sims} simulations) to show the range of outcomes their ORDER/repetition alone could "
            "produce — this does NOT simulate new trades or new market conditions, only different "
            "sequences of the same trades that already happened. With under ~30 trades this range is "
            "wide and should be read as 'here is the path-dependency risk in the trades we have,' not "
            "as a validated probability distribution of future returns. probability_of_loss_pct is the "
            "share of simulated resequencings that ended net negative."
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

    n_tested = len(TEMPLATES)
    valid = [r for r in results if r.get("sharpe_ratio") is not None]
    best_sharpe = valid[0]["sharpe_ratio"] if valid else None
    return {
        "rankings": results,
        "strategies_tested": n_tested,
        "selection_bias_warning": (
            f"{n_tested} strategies were tried on this ticker and ranked by Sharpe; the top result "
            f"(Sharpe {best_sharpe}) is a MAXIMUM over {n_tested} trials, not a single a-priori test — "
            "some inflation above the strategies' true edge is expected purely from trying multiple "
            "templates and keeping the best (multiple-comparisons / selection bias), even before any "
            "parameter tuning within a template. Treat the ranking as hypothesis-GENERATING (which "
            "published systems currently show an edge worth investigating), not as a validated result. "
            "Use run_out_of_sample_backtest and run_regime_window_backtest on the top candidate before "
            "treating its edge as real — those are the closest things this system has to confirmation."
        ) if valid else f"{n_tested} strategies were tried; none produced a usable Sharpe ratio.",
    }
