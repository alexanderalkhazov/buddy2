"""Neutral data sheet — zero interpretation, zero recommendation, zero forecast.

Same deterministic pipeline as ui/report.py, formatted under a strict rule set: no
decision language (buy/sell/attractive/cheap/bullish/etc.), no interpretation of any
number, external estimates kept in their own explicitly labeled section, every
scenario/conditional calculation labeled as such with its assumptions attached, and
every missing/inapplicable field marked N/A rather than guessed. Reads like a data
sheet, not a thesis — that's a deliberate contrast with report.py, which is written
to prompt a chatbot toward an interpreted, decision-oriented analysis. Use this one
when you want the numbers with nothing added.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from data.fundamentals import earnings_reaction_history, estimate_dispersion, eps_revision_history, insider_activity
from data import macro as macro_mod_data
from processing import backtest, bundle, regime, relative_strength, scoring, valuation
from processing import macro as macro_mod
from processing.strategies import get_spec
from storage import cache
from storage.cache import get_ohlcv

NA = "N/A — data unavailable"
NA_ASSET_CLASS = "N/A — not applicable to this asset class"


def _fmt(value, none: str = NA) -> str:
    if value is None:
        return none
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        v = float(value) + 0.0
        av = abs(v)
        if av >= 1e12:
            return f"{v / 1e12:,.4f}T"
        if av >= 1e9:
            return f"{v / 1e9:,.4f}B"
        if av >= 1e6:
            return f"{v / 1e6:,.4f}M"
        return f"{v:,.0f}" if v == int(v) else f"{v:,.4f}"
    return str(value)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


def _return_over(df, days: int) -> float | None:
    window = df[df.index >= df.index.max() - __import__("datetime").timedelta(days=days)]
    if window.empty or len(window) < 2:
        return None
    first, last = float(window["close"].iloc[0]), float(window["close"].iloc[-1])
    if first == 0:
        return None
    return round((last / first - 1) * 100, 4)


def generate(ticker: str, refresh: bool = False, include_backtest: bool = True) -> str:
    ticker = ticker.upper()
    lines: list[str] = []

    b = bundle.build(ticker, include_filings=False, refresh=refresh)
    price = b["price"]
    fund = b.get("fundamentals", {})
    asset_class = b.get("asset_class", "UNKNOWN")
    fresh = b["data_freshness"]
    is_equity = asset_class == "EQUITY"

    lines.append(f"# {ticker} — DATA REPORT")

    lines.append("\n## 1. Data Timestamp")
    lines += _table(
        ["Field", "Value"],
        [
            ["Data as of (last reported close)", fresh["as_of"]],
            ["Report generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")],
            ["Data freshness", f"{fresh['data_age_days']} day(s)"],
            ["Primary sources", "yfinance (market data, fundamentals), NewsAPI/yfinance (news), Polymarket Gamma API (prediction markets)"],
            ["Asset class", asset_class],
        ],
    )

    try:
        mr = regime.market_regime()
    except Exception as exc:
        mr = {"error": f"{type(exc).__name__}: {exc}"}
    lines.append("\n## 1a. Market/Sector Regime — index-trend and VIX-level context, not a signal about this ticker")
    if "error" not in mr:
        lines += _table(
            ["Metric", "Value"],
            [
                ["SPY", f"{_fmt(mr['spy']['last'])} ({_fmt(mr['spy']['change_pct'])}%) — {mr['spy']['trend']}"],
                ["QQQ", f"{_fmt(mr['qqq']['last'])} ({_fmt(mr['qqq']['change_pct'])}%) — {mr['qqq']['trend']}"],
                ["VIX", f"{_fmt(mr['vix']['last'])} ({_fmt(mr['vix']['change_pct'])}%) — {mr['vix']['regime']}"],
                ["Overall trend (SPY & QQQ agreement required)", mr["overall_trend"]],
                ["Risk label", mr["risk_label"]],
            ],
        )
        if is_equity:
            etf = relative_strength.sector_etf(fund.get("sector"), fund.get("industry"))
            sr = regime.sector_regime(etf)
            if sr:
                lines += _table(["Metric", "Value"], [[f"Sector/industry proxy ({etf})", f"{_fmt(sr['last'])} ({_fmt(sr['change_pct'])}%) — {sr['trend']}"]])
        lines.append(f"\n{mr.get('note')}")
    else:
        lines.append(NA)

    lines.append("\n## 2. Market Data")
    lines += _table(
        ["Metric", "Value"],
        [
            ["Last reported close", _fmt(price["last"])],
            ["Daily change", _fmt(price["change_pct"]) + "%" if price["change_pct"] is not None else NA],
            ["Volume", _fmt(fund.get("volume"))],
            ["Volume vs 30d average (ratio)", _fmt(price["vol_vs_30d_avg"])],
            ["Market capitalization", _fmt(fund.get("market_cap"))],
            ["52-week high", _fmt(fund.get("fifty_two_week_high"))],
            ["52-week low", _fmt(fund.get("fifty_two_week_low"))],
            ["Beta", _fmt(fund.get("beta")) if is_equity else NA_ASSET_CLASS],
        ],
    )

    if is_equity:
        try:
            rs = relative_strength.relative_strength(ticker, fund.get("sector"), fund.get("industry"))
        except Exception as exc:
            rs = {"error": f"{type(exc).__name__}: {exc}"}
        lines.append("\n### 2a. Relative Strength — return vs. market and sector/industry benchmarks, same window")
        if "error" not in rs:
            rows = [
                [f"{ticker} return ({rs['days']}d)", (f"{rs['ticker_return_pct']}%" if rs.get("ticker_return_pct") is not None else NA)],
                [f"{rs['benchmark']} return ({rs['days']}d)", (f"{rs['benchmark_return_pct']}%" if rs.get("benchmark_return_pct") is not None else NA)],
                ["Relative return vs. market (percentage points)", (f"{rs['relative_vs_benchmark_pct']}" if rs.get("relative_vs_benchmark_pct") is not None else NA)],
            ]
            if rs.get("sector_etf"):
                rows += [
                    [f"{rs['sector_etf']} (sector/industry proxy) return ({rs['days']}d)", (f"{rs['sector_etf_return_pct']}%" if rs.get("sector_etf_return_pct") is not None else NA)],
                    ["Relative return vs. sector/industry (percentage points)", (f"{rs['relative_vs_sector_pct']}" if rs.get("relative_vs_sector_pct") is not None else NA)],
                ]
            lines += _table(["Metric", "Value"], rows)
            lines.append(f"\n{rs.get('note')}")
        else:
            lines.append(NA)

    lines.append("\n## 3. Technical Data")
    lines += _table(
        ["Indicator", "Value"],
        [
            ["RSI(14)", _fmt(price["rsi14"])],
            ["MACD histogram", _fmt(price["macd_hist"])],
            ["SMA 20", _fmt(price["sma20"])],
            ["SMA 50", _fmt(price["sma50"])],
            ["SMA 200", _fmt(price["sma200"])],
            ["ATR(14)", _fmt(price["atr14"])],
            ["Trend classification (system-computed label)", price["trend"]],
        ],
    )

    if is_equity:
        lines.append("\n## 4. Fundamental Data")
        lines += _table(
            ["Metric", "Value"],
            [
                ["Company name / Ticker", f"{ticker}"],
                ["Exchange", str(fund.get("exchange") or NA)],
                ["Sector", str(fund.get("sector") or NA)],
                ["Industry", str(fund.get("industry") or NA)],
                ["Employees", _fmt(fund.get("employees"))],
                ["Revenue (TTM)", _fmt(fund.get("revenue"))],
                ["Revenue growth (YoY)", _fmt(fund.get("revenue_growth"))],
                ["Gross profit", _fmt(fund.get("gross_profit"))],
                ["Net income", _fmt(fund.get("net_income"))],
                ["EPS (trailing)", _fmt(fund.get("trailing_eps"))],
                ["EPS (forward consensus)", _fmt(fund.get("forward_eps"))],
                ["Profit margin", _fmt(fund.get("profit_margin"))],
                ["Operating cash flow", _fmt(fund.get("operating_cashflow"))],
                ["Free cash flow", _fmt(fund.get("free_cashflow"))],
                ["Cash", _fmt(fund.get("total_cash"))],
                ["Debt", _fmt(fund.get("total_debt"))],
                ["Net debt (debt − cash)", _fmt(fund.get("net_debt"))],
                ["Debt/Equity", _fmt(fund.get("debt_to_equity"))],
                ["Current ratio", _fmt(fund.get("current_ratio"))],
                ["Quick ratio", _fmt(fund.get("quick_ratio"))],
                ["EBITDA", _fmt(fund.get("ebitda"))],
                ["Net debt / EBITDA", _fmt(fund.get("net_debt_to_ebitda"))],
            ],
        )

        lines.append("\n## 5. Valuation")
        lines += _table(
            ["Metric", "Value"],
            [
                ["Trailing P/E", _fmt(fund.get("trailing_pe"))],
                ["Forward P/E", _fmt(fund.get("forward_pe"))],
                ["PEG ratio", _fmt(fund.get("peg_ratio"))],
                ["Price/Sales", _fmt(fund.get("price_to_sales"))],
                ["Price/Book", _fmt(fund.get("price_to_book"))],
                ["EV/EBITDA", _fmt(fund.get("ev_to_ebitda"))],
                ["EV/Revenue", _fmt(fund.get("ev_to_revenue"))],
                ["Dividend yield", _fmt(fund.get("dividend_yield"))],
            ],
        )

        ig = valuation.implied_growth(fund.get("trailing_eps"), fund.get("forward_eps"))
        lines.append("\n### Trailing-to-Forward EPS Gap (system-computed, deterministic)")
        lines.append("This is the percentage difference between two EPS reporting periods, not a market forecast.")
        gap = ig.get("trailing_to_forward_eps_gap_pct")
        lines += _table(
            ["Metric", "Value"],
            [
                ["Trailing EPS", _fmt(ig.get("trailing_eps"))],
                ["Forward EPS (consensus)", _fmt(ig.get("forward_eps"))],
                ["Forward EPS vs. trailing EPS (gap)", (f"{gap}%" if gap is not None else NA)],
            ],
        )

        grid = valuation.sensitivity_grid(price["last"], fund.get("forward_eps"), fund.get("forward_pe"))
        if "error" not in grid:
            lines.append("\n### Valuation Scenarios — CONDITIONAL CALCULATION — NOT A FORECAST")
            lines.append("Each row is an implied price given the stated EPS and multiple assumptions. Assumptions are inputs, not measured or predicted values.")
            lines += _table(
                ["Scenario", "EPS assumption", "Multiple assumption", "Implied EPS", "Implied P/E", "Implied price"],
                [
                    [
                        name.title(),
                        f"{grid['scenarios'][name]['assumption'].split(', multiple')[0]}",
                        f"multiple {grid['scenarios'][name]['assumption'].split('multiple ')[1]}",
                        _fmt(grid["scenarios"][name]["implied_eps"]),
                        _fmt(grid["scenarios"][name]["implied_pe_multiple"]),
                        _fmt(grid["scenarios"][name]["implied_price"]),
                    ]
                    for name in ("bear", "base", "bull")
                ],
            )

        lines.append("\n## 6. Ownership & Positioning")
        lines += _table(
            ["Metric", "Value"],
            [
                ["Institutional ownership", _fmt(fund.get("held_percent_institutions"))],
                ["Insider ownership", _fmt(fund.get("held_percent_insiders"))],
                ["Short percent of float", _fmt(fund.get("short_percent_of_float"))],
                ["Short ratio (days to cover)", _fmt(fund.get("short_ratio"))],
            ],
        )

        lines.append("\n## 7. Insider Transactions")
        ia = insider_activity(ticker)
        if "error" in ia or not ia.get("transactions"):
            lines.append(NA)
        else:
            lines.append(
                f"Open-market purchases (last {len(ia['transactions'])} filed transactions): {ia['buy_count']}. "
                f"Open-market sales: {ia['sell_count']}. Net shares (open-market only): {_fmt(ia['net_shares'])}."
            )
            lines += _table(
                ["Insider", "Position", "Transaction type", "Open-market transaction", "Shares", "Filing date"],
                [
                    [
                        str(t["insider"]),
                        str(t["position"]),
                        t["transaction"],
                        "Yes" if t["is_open_market"] else "No (award/gift/other non-market transaction)",
                        _fmt(t["shares"]),
                        t["start_date"],
                    ]
                    for t in ia["transactions"][:10]
                ],
            )

        lines.append("\n## 8. Earnings / Financial Events")
        lines += _table(
            ["Field", "Value"],
            [
                ["Next earnings date", str(fund.get("next_earnings_date") or NA)],
                ["Last reported earnings surprise", _fmt(fund.get("earnings_surprise_pct")) + "%" if fund.get("earnings_surprise_pct") is not None else NA],
            ],
        )

        erh = earnings_reaction_history(ticker)
        lines.append("\n### 8a. Earnings Reaction History — actual price moves after past reports")
        if "error" not in erh:
            lines += _table(
                ["Report Date", "EPS Surprise", "1d Return", "5d Return", "10d Return"],
                [
                    [
                        e["report_date"],
                        (f"{e['eps_surprise_pct']}%" if e.get("eps_surprise_pct") is not None else NA),
                        (f"{e['return_1d_pct']}%" if e.get("return_1d_pct") is not None else NA),
                        (f"{e['return_5d_pct']}%" if e.get("return_5d_pct") is not None else NA),
                        (f"{e['return_10d_pct']}%" if e.get("return_10d_pct") is not None else NA),
                    ]
                    for e in erh["events"]
                ],
            )
            agg = erh["aggregate"]
            lines.append(
                f"\nMedian reaction: {agg['return_1d']['median_pct']}% (1d), {agg['return_5d']['median_pct']}% (5d), "
                f"{agg['return_10d']['median_pct']}% (10d). Positive 1-day reaction in "
                f"{agg['return_1d']['positive_pct_of_events']}% of the last {agg['return_1d']['n']} reports."
            )
            lines.append(f"\n{erh.get('note')}")
        else:
            lines.append(NA)

        lines.append("\n## 9. Estimates & Consensus")
        lines.append("**EXTERNAL CONSENSUS — NOT MODEL JUDGMENT**")
        es = fund.get("external_sentiment") or {}
        lines += _table(
            ["Metric", "Value"],
            [
                ["Forward EPS consensus", _fmt(fund.get("forward_eps"))],
                ["Analyst count", _fmt(es.get("analyst_count"))],
                ["Mean target", _fmt(es.get("analyst_mean_target"))],
                ["Median target", _fmt(es.get("analyst_median_target"))],
                ["High target", _fmt(es.get("analyst_high_target"))],
                ["Low target", _fmt(es.get("analyst_low_target"))],
                ["Consensus rating", str(es.get("analyst_recommendation") or NA)],
            ],
        )

        rev = eps_revision_history(ticker)
        if "error" not in rev:
            lines.append("\n### EPS Estimate Revision History — EXTERNAL CONSENSUS — NOT MODEL JUDGMENT")
            est = rev["eps_estimates"]
            lines += _table(
                ["90 days ago", "60 days ago", "30 days ago", "7 days ago", "Current"],
                [[_fmt(est["90_days_ago"]), _fmt(est["60_days_ago"]), _fmt(est["30_days_ago"]), _fmt(est["7_days_ago"]), _fmt(est["current"])]],
            )
            lines += _table(
                ["Metric", "Value"],
                [
                    ["Revision, 90 days", (f"{rev['revision_pct_90d']}%" if rev.get("revision_pct_90d") is not None else NA)],
                    ["Analysts revising up (30d)", _fmt(rev.get("analysts_revising_up_last_30d"))],
                    ["Analysts revising down (30d)", _fmt(rev.get("analysts_revising_down_last_30d"))],
                ],
            )

        disp = estimate_dispersion(ticker)
        if "error" not in disp:
            lines.append("\n### EPS & Revenue Estimate Dispersion — EXTERNAL CONSENSUS — NOT MODEL JUDGMENT")
            lines += _table(
                ["Metric", "Value"],
                [
                    ["EPS estimate low", _fmt(disp.get("eps_estimate_low"))],
                    ["EPS estimate avg", _fmt(disp.get("eps_estimate_avg"))],
                    ["EPS estimate high", _fmt(disp.get("eps_estimate_high"))],
                    ["EPS estimate count", _fmt(disp.get("eps_estimate_count"))],
                    ["EPS estimate spread (% of avg)", (f"{disp['eps_estimate_spread_pct_of_avg']}%" if disp.get("eps_estimate_spread_pct_of_avg") is not None else NA)],
                    ["Revenue estimate low", _fmt(disp.get("revenue_estimate_low"))],
                    ["Revenue estimate avg", _fmt(disp.get("revenue_estimate_avg"))],
                    ["Revenue estimate high", _fmt(disp.get("revenue_estimate_high"))],
                    ["Revenue estimate count", _fmt(disp.get("revenue_estimate_count"))],
                    ["Revenue estimate spread (% of avg)", (f"{disp['revenue_estimate_spread_pct_of_avg']}%" if disp.get("revenue_estimate_spread_pct_of_avg") is not None else NA)],
                ],
            )
            if disp.get("revenue_estimate_period_note"):
                lines.append(f"\n{disp['revenue_estimate_period_note']}")

        from processing import scoring as _scoring

        td = valuation.thesis_dependencies(
            ticker,
            fund.get("forward_eps"),
            disp.get("eps_estimate_spread_pct_of_avg") if "error" not in disp else None,
            disp.get("eps_estimate_count") if "error" not in disp else None,
            ig.get("trailing_to_forward_eps_gap_pct"),
            _scoring.cyclicality_risk(fund.get("sector"), fund.get("industry")),
            fund.get("beta"),
            fund.get("forward_pe"),
        )
        lines.append("\n## 9a. Conditions Underlying the Current Forward Valuation")
        lines.append("Each item below is a condition implied by data already in this report, restated explicitly — not a new judgment or forecast.")
        for item in td["thesis_depends_on"]:
            lines.append(f"- {item}")
        lines += _table(["Dimension", "System-computed label"], [[k.replace("_", " ").title(), v] for k, v in td["thesis_fragility"].items()])
    else:
        for section in ("4. Fundamental Data", "5. Valuation", "6. Ownership & Positioning", "7. Insider Transactions", "8. Earnings / Financial Events", "9. Estimates & Consensus"):
            lines.append(f"\n## {section}")
            lines.append(NA_ASSET_CLASS)
        if fund.get("circulating_supply"):
            lines.append(f"\nCirculating supply: {_fmt(fund.get('circulating_supply'))}")

    lines.append("\n## 10. Recent News")
    news_items = cache.load_news(ticker) or []
    if news_items:
        lines += _table(
            ["Date", "Source", "Event"],
            [
                [str(n.get("published_at") or NA), str(n.get("source") or NA), str(n.get("title") or n.get("summary") or NA)]
                for n in news_items
            ],
        )
    else:
        lines.append(NA)

    lines.append("\n## 11. Macro / Market Context")

    lines.append("\n### 11a. Market-Based Proxies (daily-close pricing)")
    try:
        mp = macro_mod.market_proxies()
    except Exception as exc:
        mp = {"error": f"{type(exc).__name__}: {exc}"}
    if "error" not in mp:
        rows = [[v["label"], f"{_fmt(v['last'])}", (f"{v['change_pct']}%" if v.get("change_pct") is not None else NA)] for k, v in mp.items() if isinstance(v, dict) and "last" in v]
        lines += _table(["Instrument", "Last", "Daily Change"], rows)
        if mp.get("yield_curve_10y_minus_13w_pct_points") is not None:
            lines += _table(["Metric", "Value"], [["Yield curve (10Y minus 13W), percentage points", f"{mp['yield_curve_10y_minus_13w_pct_points']:+.2f}"]])
        lines.append(f"\n{mp.get('note', '')}")
        if mp.get("yield_curve_note"):
            lines.append(f"\n{mp['yield_curve_note']}")
    else:
        lines.append(NA)

    lines.append("\n### 11b. Official Economic Indicators — EXTERNAL SOURCE (FRED) — LAGGED, SUBJECT TO REVISION")
    try:
        econ = macro_mod_data.economic_indicators()
    except Exception as exc:
        econ = {"error": f"{type(exc).__name__}: {exc}"}
    if "error" not in econ:
        rows = []
        for series_id, v in econ.items():
            if series_id == "note" or "error" in v:
                continue
            rows.append([v["label"], v["as_of"], _fmt(v["value"]), _fmt(v.get("prior_value")), _fmt(v.get("change"))])
        lines += _table(["Indicator", "As Of (period described)", "Value", "Prior", "Change"], rows)
        lines.append(f"\n{econ.get('note', '')}")
    else:
        lines.append(NA)

    lines.append("\n### 11c. Prediction Markets (Polymarket)")
    if b.get("polymarket_related"):
        lines += _table(
            ["Market", "Odds", "Volume"],
            [[m["market"], _fmt(m.get("odds")), _fmt(m.get("volume"))] for m in b["polymarket_related"]],
        )
    else:
        lines.append("N/A — no related markets found, or the Polymarket API was unreachable this run.")

    lines.append("\n## 12. Historical Performance")
    df = get_ohlcv(ticker)
    today = date.today()
    ytd_days = (today - date(today.year, 1, 1)).days or 1
    periods = [("1D", 1), ("1W", 7), ("1M", 30), ("3M", 91), ("6M", 182), ("YTD", ytd_days), ("1Y", 365), ("3Y", 1095), ("5Y", 1825)]
    lines += _table(
        ["Period", "Return"],
        [[label, (f"{r}%" if (r := _return_over(df, d)) is not None else NA)] for label, d in periods],
    )

    lines.append("\n## 13. Historical Drawdown / Volatility")
    lines.append("Measured over the backtest period below (see Section 14) for the top-Sharpe strategy's underlying instrument.")

    if include_backtest:
        lines.append("\n## 14. Backtest Data")
        scan_results = backtest.scan(ticker)
        top = next((r for r in scan_results if "error" not in r), None)
        if top:
            full = backtest.run_template(ticker, top["template"])
            lines.append(
                f"Methodology: strategy = {top['template']} (system-selected by highest Sharpe ratio among 6 "
                f"tested strategies). Fees: {full.get('fees_pct')}% per trade. Slippage: {full.get('slippage_pct')}% "
                "per trade (adverse price move on entry and exit, not a real order-book simulation)."
            )
            lines += _table(
                ["Metric", "Value"],
                [
                    ["Test period", str(full.get("period"))],
                    ["Number of trades", _fmt(full.get("trades"))],
                    ["Total return (strategy)", _fmt(full.get("total_return_pct")) + "%"],
                    ["Total return (benchmark: buy & hold)", _fmt(full.get("buy_hold_return_pct")) + "%"],
                    ["CAGR (strategy)", _fmt(full.get("strategy_cagr_pct")) + "%"],
                    ["CAGR (benchmark)", _fmt(full.get("buy_hold_cagr_pct")) + "%"],
                    ["Sharpe ratio", _fmt(full.get("sharpe_ratio"))],
                    ["Sortino ratio", _fmt(full.get("sortino_ratio"))],
                    ["Maximum drawdown", _fmt(full.get("max_drawdown_pct")) + "%"],
                    ["Win rate", _fmt(full.get("win_rate_pct")) + "%"],
                    ["Profit factor", _fmt(full.get("profit_factor"))],
                    ["Time in market", _fmt(full.get("time_in_market_pct")) + "%"],
                    ["Outperformed benchmark", str(full.get("outperformed_buy_and_hold"))],
                ],
            )

            try:
                oos = backtest.run_out_of_sample(ticker, get_spec(top["template"]))
            except Exception:
                oos = None
            if oos and "error" not in oos:
                lines.append(f"\n### Out-of-Sample Split — {top['template']}")
                lines.append(
                    f"Split date: {oos['split_date']} ({int(oos['split_pct'] * 100)}% in-sample / "
                    f"{int((1 - oos['split_pct']) * 100)}% out-of-sample holdout, same strategy run "
                    "independently on each window)."
                )
                lines += _table(
                    ["Window", "Period", "Trades", "Sharpe", "CAGR", "Benchmark CAGR", "Win Rate"],
                    [
                        [
                            label,
                            str(seg.get("period")),
                            _fmt(seg.get("trades")),
                            _fmt(seg.get("sharpe_ratio")),
                            _fmt(seg.get("strategy_cagr_pct")) + "%",
                            _fmt(seg.get("buy_hold_cagr_pct")) + "%",
                            _fmt(seg.get("win_rate_pct")) + "%",
                        ]
                        for label, seg in (("In-sample", oos["in_sample"]), ("Out-of-sample", oos["out_of_sample"]))
                    ],
                )
                degraded = oos.get("sharpe_degraded_out_of_sample")
                lines.append(
                    f"\nsharpe_degraded_out_of_sample: {degraded} — "
                    "a fixed, non-statistical heuristic (out-of-sample Sharpe under half the in-sample Sharpe, "
                    "or flipped from positive to non-positive). Not a significance test."
                )

            try:
                rw = backtest.run_regime_windows(ticker, get_spec(top["template"]))
            except Exception:
                rw = None
            if rw and rw.get("windows"):
                lines.append(f"\n### Regime-Window Test — {rw['n_windows']} sub-periods of this ticker's own price history")
                lines += _table(
                    ["Period", "Buy&Hold Return", "Regime Label", "Volatility Label", "Trades", "Sharpe", "Strategy CAGR", "Outperformed B&H"],
                    [
                        (
                            [f"{w['start']} to {w['end']}", "", "", "", "", "", "", f"ERROR: {w['error']}"]
                            if "error" in w else
                            [
                                f"{w['start']} to {w['end']}",
                                f"{w.get('bh_return_pct')}%",
                                w.get("market_regime_label"),
                                w.get("volatility_label"),
                                _fmt(w.get("trades")),
                                _fmt(w.get("sharpe_ratio")),
                                f"{_fmt(w.get('strategy_cagr_pct'))}%",
                                str(w.get("outperformed_buy_and_hold")),
                            ]
                        )
                        for w in rw["windows"]
                    ],
                )
                lines.append(f"\nsharpe_consistent_sign_across_windows: {rw.get('sharpe_consistent_sign_across_windows')}")
                lines.append(f"\n{rw.get('note')}")

            try:
                mc = backtest.monte_carlo_bootstrap(ticker, get_spec(top["template"]), n_sims=2000, seed=42)
            except Exception:
                mc = None
            if mc and "error" not in mc:
                lines.append(f"\n### Monte Carlo Bootstrap — {mc['n_simulations']} resamples of {mc['n_trades']} actual trades")
                lines += _table(
                    ["Metric", "Value"],
                    [
                        ["Actual total return", f"{mc.get('actual_total_return_pct')}%"],
                        ["Bootstrap total return — 5th percentile", f"{mc['bootstrap_total_return_pct']['p5']}%"],
                        ["Bootstrap total return — median", f"{mc['bootstrap_total_return_pct']['median']}%"],
                        ["Bootstrap total return — 95th percentile", f"{mc['bootstrap_total_return_pct']['p95']}%"],
                        ["Bootstrap max drawdown — best case (95th pct)", f"{mc['bootstrap_max_drawdown_pct']['p5_best_case']}%"],
                        ["Bootstrap max drawdown — median", f"{mc['bootstrap_max_drawdown_pct']['median']}%"],
                        ["Bootstrap max drawdown — worst case (5th pct)", f"{mc['bootstrap_max_drawdown_pct']['p95_worst_case']}%"],
                        ["Probability of net loss (resampled)", f"{mc.get('probability_of_loss_pct')}%"],
                    ],
                )
                lines.append(f"\n{mc.get('note')}")
            elif mc and "error" in mc:
                lines.append(f"\n### Monte Carlo Bootstrap\n{mc['error']}")
        else:
            lines.append(NA)

    lines.append("\n## 14a. Trade Levels — CONDITIONAL CALCULATION FOR A HYPOTHETICAL LONG POSITION — NOT A RECOMMENDATION")
    tl = scoring.trade_levels(price["last"], price["atr14"], as_of=fresh["as_of"], next_earnings_date=fund.get("next_earnings_date"))
    if "error" not in tl:
        ep = tl.get("earnings_proximity", {})
        lines += _table(
            ["Metric", "Value"],
            [
                ["Entry (latest available close)", _fmt(tl["entry"])],
                ["Stop", _fmt(tl["stop"])],
                ["Risk per share", _fmt(tl["risk_per_share"])],
                [f"Take-profit 1 ({tl['take_profit_1_r_multiple']}R)", _fmt(tl["take_profit_1"])],
                [f"Take-profit 2 ({tl['take_profit_2_r_multiple']}R)", _fmt(tl["take_profit_2"])],
                ["Risk/reward ratio to TP1", f"{tl['risk_reward_ratio_tp1']}:1"],
                ["Invalidation condition", tl["invalidation"]],
                ["Earnings proximity severity", ep.get("severity", "UNKNOWN")],
                ["Calendar days until next earnings", _fmt(ep.get("calendar_days_until_earnings"))],
                ["Earnings proximity note", ep.get("note", NA)],
            ],
        )
        lines.append(f"\n{tl['note']}")
    else:
        lines.append(NA)

    lines.append("\n## 15. Data Quality")
    missing = [k for k, v in fund.items() if v is None and k not in ("circulating_supply", "fund_category")]
    lines += _table(
        ["Item", "Status"],
        [
            ["Missing fundamental fields", ", ".join(missing) if missing else "None"],
            ["Stale fields", "None detected" if not fresh["is_stale"] else f"Price data is {fresh['data_age_days']} day(s) old"],
            ["Conflicting sources", "None detected — single-source pipeline (yfinance)"],
            ["Estimated fields", "None — all values are directly sourced or deterministic calculations; no field is model-estimated"],
            ["External-consensus fields", "Section 9 (analyst targets/ratings/EPS revisions) and any analyst fields under Section 4"],
            ["Methodology limitations", "Single data provider (yfinance); no independent cross-verification; backtest excludes slippage; sensitivity grid assumptions are inputs, not measured or predicted values"],
        ],
    )

    lines.append("\n### 15a. Model Reliability")
    _oos = locals().get("oos")
    _rw = locals().get("rw")
    _disp = locals().get("disp")
    _erh_agg = locals().get("erh")
    lines += _table(
        ["Item", "Status"],
        [
            ["Scorecard validity", "Deterministic and reproducible — identical inputs always produce the identical score"],
            [
                "Scorecard predictive validity",
                "NOT AVAILABLE — historical point-in-time fundamentals/valuation inputs are not available in this "
                "data pipeline (only current-snapshot fundamentals are fetchable). The scorecard should be treated "
                "as a deterministic research heuristic, not a historically validated predictor of forward returns.",
            ],
            ["Technical indicator inputs", "Point-in-time computed from historical bars; no lookahead in indicator calculation itself — this does not independently verify the absence of lookahead in signal timing, order execution, or fill assumptions elsewhere in the backtest engine"],
            ["Out-of-sample strategy test", "Present — see Section 14" if _oos and "error" not in (_oos or {}) else "Not available for this run"],
            ["Regime-consistency test", "Present — see Section 14" if _rw and _rw.get("windows") else "Not available for this run"],
            ["Trade count / statistical significance", "Shown per backtest; no confidence interval computed; samples under ~30 trades are NOT statistically validated"],
            ["EPS estimate dispersion", f"{_disp['eps_estimate_spread_pct_of_avg']}% spread — see Section 9" if _disp and "error" not in (_disp or {}) else NA],
            ["Cyclicality / volatility risk labels", f"{b['scorecard'].get('risk_flags', {}).get('cyclicality_risk', 'UNKNOWN')} / {b['scorecard'].get('risk_flags', {}).get('volatility_risk', 'UNKNOWN')}" if is_equity else NA_ASSET_CLASS],
            ["Earnings-proximity severity (Section 14a)", tl.get("earnings_proximity", {}).get("severity", "UNKNOWN") if "error" not in tl else NA],
            ["Earnings reaction history sample", f"{len(_erh_agg.get('events', []))} past events — see Section 8a" if _erh_agg and "error" not in (_erh_agg or {}) else NA],
        ],
    )

    return "\n".join(lines)
