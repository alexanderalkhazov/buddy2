"""Tools the model can call to pull data on demand.

Each returns a compact JSON-serializable structure from the processing layer — never
a raw provider payload. Schemas are hand-written JSON Schema so the tool contract is
explicit and stays stable across model versions.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from data import fundamentals, polymarket
from processing import valuation
from processing import bundle, compare, scoring
from processing import portfolio as portfolio_mod

# vectorbt is slow to import (numba/regex warm-up), so the backtest module is
# loaded on first use rather than at startup — chat shouldn't pay for it.

# Small models often send scalars as strings ("false", "90"). Accepting both types
# keeps the provider's strict validator from rejecting the call; coerce() below
# normalizes the value before it reaches a handler.
BOOL = ["boolean", "string"]
INT = ["integer", "string"]

TOOLS: list[dict] = [
    {
        "name": "get_context_bundle",
        "description": (
            "Fetch the pre-processed research bundle for one ticker: latest DAILY "
            "CLOSING price (never real-time — see data_freshness.as_of/data_age_days/"
            "is_stale; never call it 'the current price') with technical indicators "
            "(RSI, MACD, SMAs, ATR, relative volume, trend label), a deterministic "
            "scorecard (fundamental/technical/valuation/sentiment/overall scores and "
            "a decision label — report these verbatim, never invent your own) PLUS "
            "scorecard.risk_flags (cyclicality_risk/momentum_extension_risk/"
            "volatility_risk, each HIGH/MODERATE/LOW/UNKNOWN) and "
            "scorecard.risk_adjusted_conviction — decision/overall answer 'is this "
            "cheap and technically strong,' NOT 'is this safe'; a high overall score "
            "with HIGH risk flags is high-conviction bullish AND high-risk, never "
            "report the overall score alone as if it meant safety, always pair it "
            "with risk_flags. Also: sentiment-tagged news summaries, related "
            "prediction-market odds, ownership data (short_percent_of_float, "
            "held_percent_insiders/institutions), and fundamentals (sector, P/E, "
            "margins, growth, next earnings date). Analyst targets/ratings are under "
            "fundamentals.external_sentiment — nested deliberately; never cite that "
            "as a supporting signal, only as labeled sentiment color. Works for any "
            "yfinance ticker, including commodities/"
            "indices (gold=GC=F, oil=CL=F, S&P 500=^GSPC). Call this before answering "
            "any question about a specific asset. Data is cached; pass refresh=true "
            "only when the user explicitly asks for fresh data or the cached as_of "
            "date is too old."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL."},
                "include_news": {"type": BOOL, "description": "Default true."},
                "include_fundamentals": {"type": BOOL, "description": "Default true."},
                "include_filings": {
                    "type": BOOL,
                    "description": "Recent SEC 8-K/10-Q filings. Default false; enable when "
                    "the question concerns corporate events or disclosures.",
                },
                "refresh": {"type": BOOL, "description": "Bypass the cache. Default false."},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_price_history",
        "description": (
            "Summarize a ticker's price path over a trailing window: return, high, low, "
            "and average volume. Use when the question is about performance over a "
            "specific timeframe rather than the current snapshot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {"type": INT, "description": "Trailing calendar days. Default 90."},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "search_prediction_markets",
        "description": (
            "Search live Polymarket markets by keyword and return current odds. Use for "
            "macro or event probabilities (rates, elections, recession) that bear on a thesis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword, e.g. 'fed rate cut'."},
                "limit": {"type": INT, "description": "Max markets. Default 5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "compare_tickers",
        "description": (
            "Compare 2+ tickers side by side: forward P/E, sector, margins, revenue "
            "growth, next earnings date, and a quick technical read (RSI, trend) for "
            "each. Use this for any relative-valuation or peer question ('how does MU "
            "compare to its peers', 'is X cheap relative to Y') instead of asserting a "
            "comparison you can't support — this is the actual data source for that."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-6 ticker symbols to compare.",
                },
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "get_insider_activity",
        "description": (
            "Recent insider transactions for an equity: buy/sell counts and net "
            "shares transacted, from real SEC-filed data (not sentiment or a "
            "rumor). Equity only. Use this when the user asks whether insiders are "
            "buying or selling, or as a real signal in a conviction/risk read — "
            "net insider buying is a stronger signal than net selling (which can "
            "be routine — tax, diversification, options exercise — so don't over-"
            "read a single sale, but a cluster of sales across multiple insiders "
            "is worth noting)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_valuation_sensitivity",
        "description": (
            "Answers 'is this cheap, or does it just look cheap because of where "
            "we are in the earnings cycle' — the question to ask before trusting a "
            "low forward P/E. Returns trailing_to_forward_eps_gap_pct: the percentage "
            "gap between trailing EPS and forward consensus EPS (this part is real "
            "fetched data, zero assumptions — a forward P/E far below trailing P/E "
            "is arithmetically explained by this gap; it is NOT the same claim as "
            "'the market is pricing in X% growth,' which this number does not "
            "establish). Also returns "
            "a bear/base/bull sensitivity grid: implied price if EPS lands at some "
            "% of forward consensus and the P/E multiple compresses/expands to some "
            "% of today's — THIS PART IS A CONDITIONAL CALCULATOR ON ASSUMPTIONS "
            "YOU SUPPLY, not a forecast. Default assumptions are illustrative only "
            "(bear: EPS at 60% of consensus + multiple at 70% of current; bull: EPS "
            "at 130% + multiple at 120%) — override them with your own view of the "
            "range via the optional bear/bull EPS and PE percentage args, and always "
            "state the assumptions used alongside any implied_price you report, "
            "never the number alone. Also returns eps_revision_history (real data: "
            "how the forward EPS estimate has moved over 7/30/60/90 days and how "
            "many analysts revised up vs down — a settled year-old estimate is a "
            "different claim than one that jumped 50% in 90 days) and "
            "valuation_breakdown (multiple_attractiveness / forward_growth_dependency "
            "/ cyclicality_adjustment — descriptive labels explaining what a high "
            "numeric Valuation score can hide; report these alongside the score, "
            "never let a high score alone read as 'low risk'). Equity only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "bear_eps_pct": {"type": ["number", "string"], "description": "e.g. 60 for 60% of forward consensus. Default 60."},
                "bear_pe_pct": {"type": ["number", "string"], "description": "Default 70."},
                "bull_eps_pct": {"type": ["number", "string"], "description": "Default 130."},
                "bull_pe_pct": {"type": ["number", "string"], "description": "Default 120."},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_portfolio",
        "description": (
            "Read the user's actual tracked positions: shares, cost basis, live "
            "market value, unrealized P&L per position and in total, portfolio "
            "weight per position, and sector concentration. Call this before "
            "suggesting a new position so you can note how it affects concentration "
            "or correlates with existing holdings — e.g. 'this would push sector X "
            "concentration to 40%'. Read-only; positions are added by the user via "
            "chat commands, never by you."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "size_position",
        "description": (
            "Compute a risk-based position size deterministically — ALWAYS call this "
            "for the Position Sizing section instead of doing the arithmetic yourself. "
            "Takes entry_price (use scorecard-consistent last price), stop_price (use "
            "scorecard.suggested_stop_price unless deliberately overriding), "
            "account_size, and risk_pct (percent of account to risk, default 1.0). "
            "Returns shares (FLOORED, never rounded, so actual risk never exceeds the "
            "target), capital_invested, portfolio_allocation_pct (capital at work — "
            "different from actual_risk_pct, the capital actually at risk to the "
            "stop; report both, don't conflate them). If account_size isn't known, "
            "ask the user or use a clearly labeled example size like $100,000 — never "
            "silently assume one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entry_price": {"type": ["number", "string"]},
                "stop_price": {"type": ["number", "string"]},
                "account_size": {"type": ["number", "string"]},
                "risk_pct": {
                    "type": ["number", "string"],
                    "description": "Percent of account to risk, e.g. 1.0 for 1%. Default 1.0.",
                },
            },
            "required": ["entry_price", "stop_price", "account_size"],
        },
    },
    {
        "name": "get_trade_levels",
        "description": (
            "Deterministic entry/stop/take-profit arithmetic for a HYPOTHETICAL LONG "
            "position — NOT a recommendation to take the trade. entry is the latest "
            "available daily close (no live quote exists in this system). stop is the "
            "same ATR-multiple stop as the scorecard's suggested_stop_price. TP1/TP2 "
            "are R-multiples (default 1.5R/3R, a common but arbitrary convention) of "
            "the entry-to-stop distance. ALWAYS call this instead of computing entry/"
            "stop/TP arithmetic yourself. Also returns earnings_proximity (severity: "
            "NONE/APPROACHING/ELEVATED/IMMEDIATE/PASSED/UNKNOWN, plus "
            "calendar_days_until_earnings — CALENDAR days, not trading sessions, say so "
            "explicitly if you cite it — and a note) — an ATR-based stop assumes roughly "
            "continuous price movement, which an earnings gap can violate; if severity is "
            "ELEVATED or IMMEDIATE, you MUST surface that warning alongside the stop, not "
            "just the price levels. This warning does NOT change the computed stop/TP — "
            "it's an additive note about execution risk, never silently drop it. Whether to take the trade at "
            "all is a separate judgment this tool does not make — report the invalidation "
            "condition and note alongside any level you cite."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "stop_multiplier": {"type": ["number", "string"], "description": "ATR multiplier for the stop, default 2.0."},
                "tp1_r_multiple": {"type": ["number", "string"], "description": "Default 1.5."},
                "tp2_r_multiple": {"type": ["number", "string"], "description": "Default 3.0."},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_relative_strength",
        "description": (
            "Is this ticker's move stock-specific, or is the whole market/sector moving "
            "together? Compares the ticker's trailing return against SPY (broad market) "
            "and, when resolvable, a sector/industry ETF proxy (e.g. SOXX for "
            "semiconductors) over the same window — real computed returns, not a "
            "forecast of continued relative performance. relative_strength_label "
            "distinguishes 'outperforming both market and sector' (real stock-specific "
            "strength) from 'outperforming the market but not its own sector' (a "
            "sector-wide move, not stock-specific) — always report which case applies, "
            "don't collapse it to just 'up' or 'down.'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "days": {"type": ["integer", "string"], "description": "Lookback window in calendar days, default 90."},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_market_regime",
        "description": (
            "Broad-market context for reading a single ticker's technicals: SPY and "
            "QQQ trend (vs. their own 20/50/200 SMAs), VIX level and regime "
            "(LOW/NORMAL/ELEVATED/HIGH), overall_trend (bullish/bearish/mixed, "
            "requires SPY AND QQQ to agree), and risk_label (risk-on/risk-off/"
            "neutral). This is index-trend + VIX-level only — NOT a breadth or "
            "advance/decline model, and not a market-timing signal by itself. "
            "ALWAYS call this before characterizing a stock's move as 'strong' or "
            "'weak' in isolation — a stock's technicals mean something different in "
            "a risk-on low-vol tape than a risk-off high-vol one. Pass a ticker to "
            "also get its sector/industry ETF proxy trend."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Optional — adds that ticker's sector/industry ETF trend to the response."}},
        },
    },
    {
        "name": "get_macro_context",
        "description": (
            "Broader macro backdrop, kept as three EXPLICITLY SEPARATE groups — never "
            "blend them into one score: (1) market_proxies — real daily-close pricing "
            "for 10Y/13W/30Y Treasury yields, US Dollar Index, WTI crude, and gold, plus "
            "a computed yield_curve_10y_minus_13w_pct_points (inversion is a real, "
            "widely-cited recession precursor but NOT a short-term timing signal — say "
            "so if you cite it). Yield values are yfinance's x10 convention. "
            "(2) economic_indicators — OFFICIAL government/Fed statistics from FRED "
            "(CPI, Fed funds rate, unemployment rate, real GDP growth): LAGGED (a "
            "release describes a past month/quarter, not today) and SUBJECT TO "
            "REVISION — never call these 'current.' (3) prediction_markets — Polymarket "
            "odds on Fed cuts/recession/inflation/unemployment/GDP, which is sentiment/"
            "betting-market pricing, not a measured fact. Cite the specific group a "
            "number came from — 'the market is pricing X' (proxies/prediction markets) "
            "is a different claim than 'the government measured X' (economic "
            "indicators)."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_strategy_templates",
        "description": (
            "List the built-in library of well-known, published algorithmic trading "
            "strategies (Donchian breakout, Bollinger mean reversion, MACD trend, "
            "MA+ADX, RSI(2) mean reversion, stochastic momentum), each with the market "
            "regime it fits and its known failure mode. Prefer these — or scan_strategies "
            "to screen all of them at once — over inventing an ad hoc rule from scratch."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "scan_strategies",
        "description": (
            "Run every built-in strategy template against one ticker's ~5 years of "
            "history and return a compact ranking by Sharpe: strategy/buy-hold CAGR, "
            "outperformed_buy_and_hold, max drawdown, win rate, Sortino, trade count. "
            "This is a screen for which published systems currently show a statistical "
            "edge — not a recommendation on its own, and not the full report (call "
            "run_backtest with the chosen template afterward for the complete field set, "
            "including whether the drawdown already reflects a stop-loss). Use this "
            "before proposing a technical strategy so the choice is grounded in what "
            "actually back-tests well, and explain *why* the top strategy's regime "
            "(trend vs. mean-reversion) matches or conflicts with the current technical "
            "picture."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "run_backtest",
        "description": (
            "Backtest a strategy against ~5 years of daily history. Returns, per "
            "strategy AND per buy-and-hold on the same stock: total_return_pct, "
            "strategy_cagr_pct/buy_hold_cagr_pct (the properly comparable figures — "
            "compare these, not just absolute total return, since exposure and time "
            "horizon differ), return_difference_vs_buy_hold_pct (percentage POINTS, "
            "strategy minus buy-hold — call it a 'return difference,' not 'excess "
            "return' or 'alpha,' those imply a risk-adjustment this doesn't do) and "
            "outperformed_buy_and_hold (explicit, unambiguous — a strategy can have a "
            "better Sharpe while still underperforming buy-and-hold on total return; "
            "report both, and interpret the gap: a lower-return, lower-exposure "
            "strategy is trading upside for less time at risk, not simply 'worse'). "
            "Also: sharpe_ratio, sortino_ratio, annualized_volatility_pct, "
            "time_in_market_pct, max_drawdown_pct, win_rate_pct, trade count, "
            "expectancy (average $ P&L per trade at the $10k backtest sizing), "
            "profit_factor (gross profit / gross loss — >1 means winners outweighed "
            "losers in dollar terms), avg_win_pct/avg_loss_pct. "
            "drawdown_reflects_stop_loss is true when stop_loss_pct was set — in that "
            "case max_drawdown_pct is already the WITH-stop simulated drawdown, not a "
            "raw number with the stop asserted afterward. None of these figures is "
            "beta-adjusted alpha — do not call them that. "
            "Either pass 'template' to run one of the built-in named strategies (see "
            "list_strategy_templates), optionally with overrides, or pass custom "
            "'entry_rules'/'exit_rules' comparing indicator fields to numbers or to each "
            "other. Available fields: open, high, low, close, volume, rsi14, rsi2, macd, "
            "signal, hist, sma20, sma50, sma200, bb_upper, bb_lower, bb_mid, bb_pctb, "
            "atr14, vol_avg30, stoch_k, stoch_d, adx14, donchian_upper20, "
            "donchian_lower10, roc10, obv. Operators: <, <=, >, >=, crosses_above, "
            "crosses_below. stop_loss_pct/take_profit_pct are fractions (0.08 = 8%); "
            "trailing_stop makes the stop trail the peak instead of the entry price. "
            "ALWAYS call this (or scan_strategies) before presenting any strategy as "
            "actionable, and report the metrics — including underperformance vs. "
            "buy-and-hold or a large drawdown — even when they undercut the thesis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "template": {
                    "type": "string",
                    "description": "Name of a built-in template from list_strategy_templates. "
                    "If set, 'strategy' is only needed for overrides (e.g. stop_loss_pct).",
                },
                "strategy": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "entry_rules": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "left": {"type": "string"},
                                    "op": {"type": "string"},
                                    "right": {},
                                },
                                "required": ["left", "op", "right"],
                            },
                        },
                        "exit_rules": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "left": {"type": "string"},
                                    "op": {"type": "string"},
                                    "right": {},
                                },
                                "required": ["left", "op", "right"],
                            },
                        },
                        "entry_logic": {"type": "string", "enum": ["all", "any"]},
                        "exit_logic": {"type": "string", "enum": ["all", "any"]},
                        "stop_loss_pct": {"type": "number"},
                        "take_profit_pct": {"type": "number"},
                        "trailing_stop": {"type": BOOL},
                    },
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "run_out_of_sample_backtest",
        "description": (
            "Checks whether a strategy's backtest edge survives on data it wasn't fit "
            "on — splits the cached history into an in-sample window (default: first "
            "70%) and an out-of-sample holdout (the remaining, later window), and runs "
            "the SAME strategy independently on each via run_backtest's engine. Returns "
            "in_sample and out_of_sample (each shaped like a run_backtest result) plus "
            "sharpe_degraded_out_of_sample (true if the out-of-sample Sharpe collapsed "
            "to under half the in-sample Sharpe, or flipped from positive to "
            "non-positive) — a real warning sign of curve-fitting to the in-sample "
            "window, though NOT a statistical significance test. A preserved "
            "out-of-sample result is reassuring but still not proof of a persistent "
            "edge — both windows remain a single ticker's single historical path. "
            "Use this before calling any scan_strategies/run_backtest result a "
            "'reliable' edge, not just a strong in-sample number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "template": {"type": "string", "description": "Name of a built-in template from list_strategy_templates."},
                "split_pct": {"type": "number", "description": "Fraction of history used as the in-sample window, default 0.7."},
            },
            "required": ["ticker", "template"],
        },
    },
    {
        "name": "run_regime_window_backtest",
        "description": (
            "Splits the cached history into several consecutive sub-periods of this "
            "ticker's OWN price path (default 4) and runs the same strategy "
            "independently on each — checks whether an edge holds across genuinely "
            "different conditions (a window where the stock was flat/falling vs. one "
            "where it ran hard) rather than being carried by one exceptional window. "
            "market_regime_label/volatility_label per window are derived purely from "
            "THIS ticker's own return/volatility in that window, not an external "
            "market classification. sharpe_consistent_sign_across_windows flips to "
            "false when Sharpe changes sign between windows — a warning the aggregate "
            "backtest number may not reflect a repeatable pattern. Use this alongside "
            "run_out_of_sample_backtest, not instead of it — they check different things."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "template": {"type": "string", "description": "Name of a built-in template from list_strategy_templates."},
                "n_windows": {"type": ["integer", "string"], "description": "Number of sub-periods, default 4."},
            },
            "required": ["ticker", "template"],
        },
    },
    {
        "name": "run_monte_carlo_backtest",
        "description": (
            "Bootstrap-resamples a strategy's own historical trade returns (with "
            "replacement, thousands of simulations) to show the range of outcomes "
            "those SAME trades' order/repetition alone could produce. This does NOT "
            "simulate new trades or new market conditions — it isolates path-"
            "dependency risk in the actual trade sequence from the underlying return "
            "distribution. Returns bootstrap_total_return_pct and "
            "bootstrap_max_drawdown_pct as {p5, median, p95} plus "
            "probability_of_loss_pct. With under ~30 trades (typical for this "
            "system's backtests) this range is WIDE — report it as 'the path-"
            "dependency risk in the trades we have,' never as a validated forward-"
            "looking probability distribution. Errors if the strategy has fewer than "
            "5 trades."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "template": {"type": "string", "description": "Name of a built-in template from list_strategy_templates."},
                "n_sims": {"type": ["integer", "string"], "description": "Number of bootstrap simulations, default 2000."},
            },
            "required": ["ticker", "template"],
        },
    },
    {
        "name": "get_earnings_reaction_history",
        "description": (
            "How did the stock ACTUALLY move in the 1/5/10 days after each of its "
            "last several reported earnings — real historical price action anchored "
            "to yfinance's exact earnings-announcement timestamps (not the fiscal-"
            "quarter-end dates in earnings_history, which are NOT the day the market "
            "reacted). A company can beat EPS estimates and still sell off (weak "
            "guidance, 'sell the news') — eps_surprise_pct and return_1d_pct are "
            "reported per-event for exactly this reason; never assume they move "
            "together. Sample size is small (typically under 10 events) and each "
            "report's market conditions differ — this is NOT a forecast of the next "
            "earnings reaction, only a record of what actually happened before."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "limit": {"type": ["integer", "string"], "description": "Max number of past earnings events, default 8."},
            },
            "required": ["ticker"],
        },
    },
]


def _run_backtest(**kw) -> dict:
    from processing import backtest  # deferred: pulls in vectorbt

    if kw.get("template"):
        return backtest.run_template(kw["ticker"], kw["template"], **(kw.get("strategy") or {}))
    if not kw.get("strategy") or not kw["strategy"].get("entry_rules"):
        raise ValueError("pass either 'template' or a 'strategy' with entry_rules")
    return backtest.run(kw["ticker"], kw["strategy"])


def _run_out_of_sample_backtest(**kw) -> dict:
    from processing import backtest  # deferred: pulls in vectorbt
    from processing.strategies import get_spec

    return backtest.run_out_of_sample(
        kw["ticker"], get_spec(kw["template"]), split_pct=float(kw.get("split_pct", 0.7))
    )


def _run_regime_window_backtest(**kw) -> dict:
    from processing import backtest  # deferred: pulls in vectorbt
    from processing.strategies import get_spec

    return backtest.run_regime_windows(
        kw["ticker"], get_spec(kw["template"]), n_windows=int(kw.get("n_windows", 4))
    )


def _run_monte_carlo_backtest(**kw) -> dict:
    from processing import backtest  # deferred: pulls in vectorbt
    from processing.strategies import get_spec

    return backtest.monte_carlo_bootstrap(
        kw["ticker"], get_spec(kw["template"]), n_sims=int(kw.get("n_sims", 2000))
    )


def _get_earnings_reaction_history(**kw) -> dict:
    return fundamentals.earnings_reaction_history(kw["ticker"], limit=int(kw.get("limit", 8)))


def _scan_strategies(**kw) -> list[dict]:
    from processing import backtest  # deferred: pulls in vectorbt

    return backtest.scan(kw["ticker"])


def _list_strategy_templates(**_kw) -> list[dict]:
    from processing.strategies import list_templates

    return list_templates()


def _as_list(value) -> list[str]:
    """Accept a real list or a comma-separated string — small models send both."""
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def _get_valuation_sensitivity(**kw) -> dict:
    ticker = kw["ticker"]
    fund = fundamentals.fetch_fundamentals(ticker)
    if fund.get("asset_class") != "EQUITY":
        return {"error": f"valuation sensitivity requires EPS/P/E data, not available for asset_class={fund.get('asset_class')}"}

    from storage.cache import get_ohlcv
    from processing.indicators import snapshot

    last_price = snapshot(ticker, get_ohlcv(ticker)).last

    scenarios = {
        "bear": {
            "eps_pct_of_forward": float(kw.get("bear_eps_pct", 60)) / 100,
            "pe_pct_of_current": float(kw.get("bear_pe_pct", 70)) / 100,
        },
        "base": {"eps_pct_of_forward": 1.0, "pe_pct_of_current": 1.0},
        "bull": {
            "eps_pct_of_forward": float(kw.get("bull_eps_pct", 130)) / 100,
            "pe_pct_of_current": float(kw.get("bull_pe_pct", 120)) / 100,
        },
    }

    from processing.scoring import cyclicality_risk

    ig = valuation.implied_growth(fund.get("trailing_eps"), fund.get("forward_eps"))
    cyc = cyclicality_risk(fund.get("sector"), fund.get("industry"))

    return {
        "ticker": ticker.upper(),
        "implied_growth": ig,
        "eps_revision_history": fundamentals.eps_revision_history(ticker),
        "valuation_breakdown": valuation.valuation_breakdown(
            fund.get("forward_pe"), ig.get("trailing_to_forward_eps_gap_pct"), cyc
        ),
        "sensitivity_grid": valuation.sensitivity_grid(
            last_price, fund.get("forward_eps"), fund.get("forward_pe"), scenarios
        ),
    }


def _get_trade_levels(**kw) -> dict:
    from storage.cache import get_ohlcv
    from processing.indicators import snapshot

    ticker = kw["ticker"]
    snap = snapshot(ticker, get_ohlcv(ticker))
    fund = fundamentals.fetch_fundamentals(ticker)
    return scoring.trade_levels(
        snap.last,
        snap.atr14,
        stop_multiplier=float(kw.get("stop_multiplier", 2.0)),
        tp1_r_multiple=float(kw.get("tp1_r_multiple", 1.5)),
        tp2_r_multiple=float(kw.get("tp2_r_multiple", 3.0)),
        as_of=snap.as_of,
        next_earnings_date=fund.get("next_earnings_date"),
    )


def _get_market_regime(**kw) -> dict:
    from processing import regime, relative_strength

    out = regime.market_regime()
    ticker = kw.get("ticker")
    if ticker:
        fund = fundamentals.fetch_fundamentals(ticker)
        etf = relative_strength.sector_etf(fund.get("sector"), fund.get("industry"))
        out["sector_regime"] = regime.sector_regime(etf)
    return out


def _get_macro_context(**_kw) -> dict:
    from processing import macro

    return macro.full_macro_context()


def _get_relative_strength(**kw) -> dict:
    from processing import relative_strength

    ticker = kw["ticker"]
    fund = fundamentals.fetch_fundamentals(ticker)
    return relative_strength.relative_strength(
        ticker, fund.get("sector"), fund.get("industry"), days=int(kw.get("days", 90))
    )


def _get_context_bundle(**kw) -> dict:
    return bundle.build(
        kw["ticker"],
        include_news=kw.get("include_news", True),
        include_fundamentals=kw.get("include_fundamentals", True),
        include_filings=kw.get("include_filings", False),
        refresh=kw.get("refresh", False),
    )


HANDLERS: dict[str, Callable[..., Any]] = {
    "get_context_bundle": _get_context_bundle,
    "get_price_history": lambda **kw: bundle.history_window(kw["ticker"], kw.get("days", 90)),
    "search_prediction_markets": lambda **kw: polymarket.search_markets(
        kw["query"], kw.get("limit", 5)
    ),
    "run_backtest": _run_backtest,
    "run_out_of_sample_backtest": _run_out_of_sample_backtest,
    "run_regime_window_backtest": _run_regime_window_backtest,
    "run_monte_carlo_backtest": _run_monte_carlo_backtest,
    "get_earnings_reaction_history": _get_earnings_reaction_history,
    "scan_strategies": _scan_strategies,
    "list_strategy_templates": _list_strategy_templates,
    "compare_tickers": lambda **kw: compare.compare_tickers(_as_list(kw["tickers"])),
    "get_portfolio": lambda **_kw: portfolio_mod.portfolio_summary(),
    "get_insider_activity": lambda **kw: fundamentals.insider_activity(kw["ticker"]),
    "get_valuation_sensitivity": _get_valuation_sensitivity,
    "size_position": lambda **kw: scoring.position_size(
        entry_price=float(kw["entry_price"]),
        stop_price=float(kw["stop_price"]),
        account_size=float(kw["account_size"]),
        risk_pct=float(kw.get("risk_pct", 1.0)),
    ),
    "get_trade_levels": _get_trade_levels,
    "get_relative_strength": _get_relative_strength,
    "get_market_regime": _get_market_regime,
    "get_macro_context": _get_macro_context,
}


_BOOL_FIELDS = {"include_news", "include_fundamentals", "include_filings", "refresh"}
_INT_FIELDS = {"days", "limit"}


def coerce(tool_input: dict) -> dict:
    """Normalize loosely-typed arguments.

    Smaller models routinely send "false" or "5" as strings. Rather than fail the
    turn on a type mismatch, coerce the known scalar fields to what the schema wants.
    """
    out = dict(tool_input or {})
    for key in _BOOL_FIELDS & out.keys():
        value = out[key]
        if isinstance(value, str):
            out[key] = value.strip().lower() in ("true", "1", "yes")
    for key in _INT_FIELDS & out.keys():
        try:
            out[key] = int(out[key])
        except (TypeError, ValueError):
            del out[key]  # fall back to the handler's default
    return out


def execute(name: str, tool_input: dict) -> tuple[str, bool]:
    """Run a tool; return (serialized result, is_error)."""
    handler = HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}", True
    try:
        return json.dumps(handler(**coerce(tool_input)), default=str), False
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}", True
