"""Zero-LLM report generator.

Runs the exact same deterministic pipeline the chat agent's tools use — bundle,
scorecard, strategy scan, backtest, position sizing, insider activity — and formats
it as Markdown, tailored for pasting straight into a chatbot's text box. No API
calls, no cost, no rate limits: this is Layer 1 (fetched facts) and Layer 2
(deterministic calculations) with the Layer 3 LLM step skipped entirely, left for
you to run yourself wherever you like.
"""

from __future__ import annotations

from datetime import datetime, timezone

from data.fundamentals import estimate_dispersion, eps_revision_history, insider_activity
from processing import backtest, bundle, compare, portfolio, regime, relative_strength, scoring, valuation
from processing.strategies import get_spec

SYSTEM_BLOCK = """\
You are acting as a trading research analyst. Everything in the DATA section below \
was fetched by real market-data APIs and computed by deterministic code — no AI wrote \
any of these numbers. Your job is interpretation only: explain what the data means, \
never invent a number, and never silently override a value already computed below.

Ground rules:
1. `scorecard.decision` / `.overall` / `.fundamental` / `.technical` / `.valuation` / \
`.sentiment` are fixed-formula outputs — report them verbatim, don't recompute or \
override them.
2. `scorecard.risk_flags` and `risk_adjusted_conviction` answer a DIFFERENT question \
than `.overall` — `.overall` says "is this cheap and technically strong," risk_flags \
say "how extended / cyclical / volatile is this." A high `.overall` score with HIGH \
risk flags means high-conviction bullish AND high-risk — never present a high overall \
score alone as if it meant "safe."
3. If a fundamental/valuation field reads "n/a (asset_class)", that's a category error \
to force (e.g. gold has no P/E) — not a missing data point. Say so plainly.
4. Analyst targets/ratings are labeled EXTERNAL SENTIMENT — never cite them as a \
supporting signal, only as color, and note analysts are frequently wrong.
5. `return_difference_vs_buy_hold_pct` is percentage POINTS, not "excess return" or \
"alpha" (no beta/CAPM adjustment is computed here).
6. `max_drawdown_pct` already includes any stop-loss shown when \
`drawdown_reflects_stop_loss` is true — don't describe the stop as separate, untested \
risk management layered on afterward.
7. `sample_size_note` is a fact about trade count, not a statistical verdict — quote \
it, don't upgrade "N trades" into "statistically significant."
8. Position-sizing numbers are pre-computed: `portfolio_allocation_pct` (capital \
deployed) and `actual_risk_pct` (capital actually at risk to the stop) are different \
numbers — never conflate them.
9. The price is a DAILY CLOSE, not real-time — check `data_freshness` before calling \
anything "current."
10. No model reliably predicts short-term price direction. Treat the backtest as a \
sanity check on one historical window, not proof a strategy works.
11. `trailing_to_forward_eps_gap_pct` (Valuation Sensitivity section) IS real, computed \
data — the percentage gap between trailing EPS and forward-consensus EPS. It is NOT \
"the market is pricing in X% growth" — that phrasing claims the market has an explicit \
growth target, which this number doesn't establish. Say "forward-consensus EPS is X% \
above trailing EPS" instead. The bear/base/bull `implied_price` grid below it is ALSO \
NOT real data — it's a conditional calculator on stated assumptions (the % of consensus \
EPS, the % multiple change). Always quote the assumption alongside any implied_price you \
cite ("if EPS lands at 60% of consensus and the multiple compresses to 70%, implied price \
is $X") — never state an implied_price as if it were a prediction.

Please respond with: **Decision** / **Confidence breakdown** (report the scores AND \
the risk flags together) / **Why** / **Why not** / **Scenarios** (bull/base/bear) / \
**Key uncertainty** / **Suggested action** / **Position sizing** / **Invalidation** / \
**Backtest summary**.
"""


def _fmt(value, suffix: str = "", none: str = "n/a") -> str:
    """Human-readable numbers — never scientific notation (1.1e+12 reads badly in
    a chatbot's text box). Large magnitudes get T/B/M suffixes like typical
    financial reporting; everything else is plain comma-grouped."""
    if value is None:
        return none
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        v = float(value) + 0.0  # normalize -0.0 to 0.0 so it never prints as "-0"
        av = abs(v)
        if av >= 1e12:
            out = f"{v / 1e12:,.2f}T"
        elif av >= 1e9:
            out = f"{v / 1e9:,.2f}B"
        elif av >= 1e6:
            out = f"{v / 1e6:,.2f}M"
        elif av >= 1000:
            out = f"{v:,.0f}"
        elif av >= 1:
            out = f"{v:,.0f}" if v == int(v) else f"{v:,.2f}"
        else:
            out = f"{v:.4g}"  # small fractions (0.0265, 0.5591) stay plain, not scientific
        return out + suffix
    return f"{value}{suffix}"


def _kv_table(rows: list[tuple[str, str]]) -> list[str]:
    """A compact two-column Markdown table — renders cleanly in every major
    chatbot's UI, unlike hand-aligned plain-text columns which break as soon as a
    value is a different width than expected."""
    out = ["| Field | Value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return out


def generate(
    ticker: str,
    account_size: float | None = None,
    risk_pct: float = 1.0,
    peers: list[str] | None = None,
    refresh: bool = False,
    include_insiders: bool = True,
) -> str:
    ticker = ticker.upper()
    lines: list[str] = []

    b = bundle.build(ticker, include_filings=False, refresh=refresh)
    price = b["price"]
    fund = b.get("fundamentals", {})
    sc = b["scorecard"]
    asset_class = b.get("asset_class", "UNKNOWN")

    lines.append(SYSTEM_BLOCK)
    lines.append("---\n# DATA\n")
    lines.append(f"# {ticker} — Research Data Package")
    lines.append(
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"zero LLM calls used · asset class: **{asset_class}***"
    )
    fresh = b["data_freshness"]
    freshness_flag = "⚠️ STALE" if fresh["is_stale"] else "current"
    lines.append(f"\n**Data freshness:** as of {fresh['as_of']} ({fresh['data_age_days']}d old) — {freshness_flag}")

    try:
        mr = regime.market_regime()
    except Exception as exc:
        mr = {"error": f"{type(exc).__name__}: {exc}"}
    if "error" not in mr:
        lines.append(f"\n## Market Regime — context for reading {ticker}'s own technicals")
        lines += _kv_table(
            [
                ("SPY", f"{_fmt(mr['spy']['last'])} ({_fmt(mr['spy']['change_pct'], '%')}) — {mr['spy']['trend']}"),
                ("QQQ", f"{_fmt(mr['qqq']['last'])} ({_fmt(mr['qqq']['change_pct'], '%')}) — {mr['qqq']['trend']}"),
                ("VIX", f"{_fmt(mr['vix']['last'])} ({_fmt(mr['vix']['change_pct'], '%')}) — {mr['vix']['regime']}"),
                ("Overall trend (SPY & QQQ agreement)", mr["overall_trend"]),
                ("Risk label", mr["risk_label"]),
            ]
        )
        if asset_class == "EQUITY":
            etf = relative_strength.sector_etf(fund.get("sector"), fund.get("industry"))
            sr = regime.sector_regime(etf)
            if sr:
                lines.append(f"\n**Sector/industry proxy ({etf}):** {_fmt(sr['last'])} ({_fmt(sr['change_pct'], '%')}) — {sr['trend']}")
        lines.append(f"\n*{mr.get('note')}*")

    ig = None
    if asset_class == "EQUITY":
        ig = valuation.implied_growth(fund.get("trailing_eps"), fund.get("forward_eps"))
        gap = ig.get("trailing_to_forward_eps_gap_pct")
        if gap is not None and abs(gap) >= 30:
            direction = "above" if gap > 0 else "below"
            lines.append(
                f"\n> ⚠️ **LARGE TRAILING/FORWARD EPS GAP:** forward-consensus EPS is "
                f"{abs(gap):.1f}% {direction} trailing EPS. Not a forecast or a "
                f"bearish/bullish signal by itself — a dependency warning: the "
                f"current forward multiple leans heavily on that estimate holding; "
                f"if it's revised, the apparent multiple can change quickly. See "
                f"Valuation Sensitivity below."
            )

    lines.append("\n## Price & Technicals")
    lines += _kv_table(
        [
            ("Last close", _fmt(price["last"])),
            ("Change", _fmt(price["change_pct"], "%")),
            ("RSI(14)", _fmt(price["rsi14"])),
            ("MACD histogram", _fmt(price["macd_hist"])),
            ("SMA 20 / 50 / 200", f"{_fmt(price['sma20'])} / {_fmt(price['sma50'])} / {_fmt(price['sma200'])}"),
            ("ATR(14)", _fmt(price["atr14"])),
            ("Volume vs 30d avg", _fmt(price["vol_vs_30d_avg"], "x")),
            ("Trend", price["trend"]),
            ("52-week range", f"{_fmt(fund.get('fifty_two_week_low'))} – {_fmt(fund.get('fifty_two_week_high'))}"),
            ("Market cap", _fmt(fund.get("market_cap"))),
        ]
    )

    if asset_class == "EQUITY":
        try:
            rs = relative_strength.relative_strength(ticker, fund.get("sector"), fund.get("industry"))
        except Exception as exc:
            rs = {"error": f"{type(exc).__name__}: {exc}"}
        if "error" not in rs:
            lines.append("\n### Relative Strength — is this move stock-specific, or the whole market/sector moving?")
            rows = [
                (f"{ticker} return ({rs['days']}d)", _fmt(rs.get("ticker_return_pct"), "%")),
                (f"{rs['benchmark']} return ({rs['days']}d)", _fmt(rs.get("benchmark_return_pct"), "%")),
                ("Relative vs. market", _fmt(rs.get("relative_vs_benchmark_pct"), " pts")),
            ]
            if rs.get("sector_etf"):
                rows += [
                    (f"{rs['sector_etf']} (sector/industry) return ({rs['days']}d)", _fmt(rs.get("sector_etf_return_pct"), "%")),
                    ("Relative vs. sector/industry", _fmt(rs.get("relative_vs_sector_pct"), " pts")),
                ]
            lines += _kv_table(rows)
            if rs.get("relative_strength_label"):
                lines.append(f"\n**{rs['relative_strength_label']}**")
            lines.append(f"\n*{rs.get('note')}*")

    if asset_class == "EQUITY":
        lines.append("\n## Fundamentals")
        lines += _kv_table(
            [
                ("Sector / Industry", f"{fund.get('sector')} / {fund.get('industry')}"),
                ("Trailing / Forward P/E", f"{_fmt(fund.get('trailing_pe'))} / {_fmt(fund.get('forward_pe'))}"),
                ("Profit margin", _fmt(fund.get("profit_margin"))),
                ("Revenue growth", _fmt(fund.get("revenue_growth"))),
                ("Beta", _fmt(fund.get("beta"))),
                ("Dividend yield", _fmt(fund.get("dividend_yield"))),
                ("Last EPS actual / estimate", f"{_fmt(fund.get('last_eps_actual'))} / {_fmt(fund.get('last_eps_estimate'))}"),
                ("Earnings surprise (last)", _fmt(fund.get("earnings_surprise_pct"), "%")),
                ("Next earnings date", _fmt(fund.get("next_earnings_date"))),
                ("Most recent quarter end", _fmt(fund.get("most_recent_quarter"))),
                ("Last fiscal year end", _fmt(fund.get("last_fiscal_year_end"))),
                ("Next fiscal year end", _fmt(fund.get("next_fiscal_year_end"))),
            ]
        )

        lines.append("\n## Cash Flow & Balance Sheet")
        lines += _kv_table(
            [
                ("Revenue (TTM)", _fmt(fund.get("revenue"))),
                ("Gross profit", _fmt(fund.get("gross_profit"))),
                ("Operating cash flow", _fmt(fund.get("operating_cashflow"))),
                ("Free cash flow", _fmt(fund.get("free_cashflow"))),
                ("Cash", _fmt(fund.get("total_cash"))),
                ("Total debt", _fmt(fund.get("total_debt"))),
                ("Net debt (debt − cash)", _fmt(fund.get("net_debt"))),
                ("Debt / Equity", _fmt(fund.get("debt_to_equity"))),
                ("Current ratio", _fmt(fund.get("current_ratio"))),
                ("Quick ratio", _fmt(fund.get("quick_ratio"))),
                ("EBITDA", _fmt(fund.get("ebitda"))),
                ("Net debt / EBITDA", _fmt(fund.get("net_debt_to_ebitda"))),
            ]
        )

        lines.append("\n## Valuation Multiples (beyond P/E)")
        lines += _kv_table(
            [
                ("Price / Sales", _fmt(fund.get("price_to_sales"))),
                ("Price / Book", _fmt(fund.get("price_to_book"))),
                ("PEG ratio", _fmt(fund.get("peg_ratio"))),
                ("EV / EBITDA", _fmt(fund.get("ev_to_ebitda"))),
                ("EV / Revenue", _fmt(fund.get("ev_to_revenue"))),
            ]
        )

        lines.append("\n## Ownership & Positioning")
        lines += _kv_table(
            [
                ("Short % of float", _fmt(fund.get("short_percent_of_float"), "%")),
                ("Short ratio (days to cover)", _fmt(fund.get("short_ratio"))),
                ("Held by insiders", _fmt(fund.get("held_percent_insiders"), "%")),
                ("Held by institutions", _fmt(fund.get("held_percent_institutions"), "%")),
            ]
        )

        if include_insiders:
            ia = insider_activity(ticker)
            if "error" not in ia and ia.get("transactions"):
                lines.append("\n### Recent Insider Transactions")
                lines.append(f"Buys: {ia['buy_count']}  |  Sells: {ia['sell_count']}  |  Net shares: {_fmt(ia['net_shares'])}")
                lines.append(f"*{ia['note']}*\n")
                lines.append("| Insider | Position | Transaction | Open market? | Shares | Date |")
                lines.append("|---|---|---|---|---|---|")
                for t in ia["transactions"][:8]:
                    lines.append(
                        f"| {t['insider']} | {t['position']} | {t['transaction']} | "
                        f"{'Yes' if t['is_open_market'] else 'no (award/gift)'} | "
                        f"{_fmt(t['shares'])} | {t['start_date']} |"
                    )

        es = fund.get("external_sentiment") or {}
        lines.append(
            f"\n> **External sentiment (not evidence):** analyst mean target "
            f"{_fmt(es.get('analyst_mean_target'))}, rating {_fmt(es.get('analyst_recommendation'))}, "
            f"n={_fmt(es.get('analyst_count'))} analysts."
        )

        lines.append("\n## Valuation Sensitivity — trailing/forward EPS gap, and what if the estimate is wrong")
        lines += _kv_table(
            [
                ("Trailing EPS", _fmt(ig.get("trailing_eps"))),
                ("Forward EPS (consensus)", _fmt(ig.get("forward_eps"))),
                ("**Forward EPS vs. trailing EPS (gap)**", f"**{_fmt(ig.get('trailing_to_forward_eps_gap_pct'), '%')}**"),
            ]
        )
        lines.append(f"\n*{ig.get('note')}*")

        vb = valuation.valuation_breakdown(
            fund.get("forward_pe"),
            ig.get("trailing_to_forward_eps_gap_pct"),
            scoring.cyclicality_risk(fund.get("sector"), fund.get("industry")),
        )
        lines.append("\n### What the Valuation score doesn't show on its own")
        lines += _kv_table(
            [
                ("Multiple attractiveness (cheap in isolation?)", vb["multiple_attractiveness"]),
                ("Forward growth dependency", vb["forward_growth_dependency"]),
                ("Cyclicality adjustment", vb["cyclicality_adjustment"]),
            ]
        )
        lines.append(f"\n*{vb['note']}*")

        try:
            rev = eps_revision_history(ticker)
        except Exception:
            rev = None
        if rev and "error" not in rev:
            lines.append("\n### EPS Estimate Revision History (real data — not a scenario assumption)")
            est = rev["eps_estimates"]
            lines.append("| 90d ago | 60d ago | 30d ago | 7d ago | Current |")
            lines.append("|---|---|---|---|---|")
            lines.append(
                f"| {_fmt(est['90_days_ago'])} | {_fmt(est['60_days_ago'])} | "
                f"{_fmt(est['30_days_ago'])} | {_fmt(est['7_days_ago'])} | **{_fmt(est['current'])}** |"
            )
            lines.append(
                f"\nRevision: {_fmt(rev.get('revision_pct_90d'), '%')} over 90 days · "
                f"{rev.get('analysts_revising_up_last_30d', 0)} analysts revised up, "
                f"{rev.get('analysts_revising_down_last_30d', 0)} revised down (last 30 days)."
            )
            lines.append(f"\n*{rev.get('stability_note')}*")

        try:
            disp = estimate_dispersion(ticker)
        except Exception:
            disp = None
        if disp and "error" not in disp:
            lines.append("\n### EPS & Revenue Estimate Dispersion (real data — shows analyst disagreement, not just the average)")
            lines += _kv_table(
                [
                    ("EPS estimate range (low / avg / high)", f"{_fmt(disp.get('eps_estimate_low'))} / {_fmt(disp.get('eps_estimate_avg'))} / {_fmt(disp.get('eps_estimate_high'))}"),
                    ("EPS estimate count", _fmt(disp.get("eps_estimate_count"))),
                    ("EPS estimate spread (% of avg)", _fmt(disp.get("eps_estimate_spread_pct_of_avg"), "%")),
                    ("Revenue estimate range (low / avg / high)", f"{_fmt(disp.get('revenue_estimate_low'))} / {_fmt(disp.get('revenue_estimate_avg'))} / {_fmt(disp.get('revenue_estimate_high'))}"),
                    ("Revenue estimate count", _fmt(disp.get("revenue_estimate_count"))),
                    ("Revenue estimate spread (% of avg)", _fmt(disp.get("revenue_estimate_spread_pct_of_avg"), "%")),
                ]
            )
            if disp.get("dispersion_note"):
                lines.append(f"\n*{disp['dispersion_note']}*")
            if disp.get("revenue_estimate_period_note"):
                lines.append(f"\n*{disp['revenue_estimate_period_note']}*")

        td = valuation.thesis_dependencies(
            ticker,
            fund.get("forward_eps"),
            (disp or {}).get("eps_estimate_spread_pct_of_avg"),
            (disp or {}).get("eps_estimate_count"),
            ig.get("trailing_to_forward_eps_gap_pct"),
            scoring.cyclicality_risk(fund.get("sector"), fund.get("industry")),
            fund.get("beta"),
            fund.get("forward_pe"),
        )
        lines.append("\n## Thesis Dependencies — what must stay true for the current valuation to hold")
        for item in td["thesis_depends_on"]:
            lines.append(f"- {item}")
        lines.append("\n**Thesis fragility:**")
        lines += _kv_table([(k.replace("_", " ").title(), v) for k, v in td["thesis_fragility"].items()])
        lines.append(f"\n*{td['note']}*")

        grid = valuation.sensitivity_grid(price["last"], fund.get("forward_eps"), fund.get("forward_pe"))
        if "error" not in grid:
            lines.append("\n### Bear / Base / Bull — implied price on stated assumptions")
            lines.append("| Scenario | Assumption | Implied EPS | Implied P/E | Implied Price | Implied Return |")
            lines.append("|---|---|---|---|---|---|")
            for name in ("bear", "base", "bull"):
                row = grid["scenarios"][name]
                lines.append(
                    f"| **{name.title()}** | {row['assumption']} | {_fmt(row['implied_eps'])} | "
                    f"{_fmt(row['implied_pe_multiple'])} | {_fmt(row['implied_price'])} | "
                    f"{_fmt(row['implied_return_pct'], '%')} |"
                )
            lines.append(f"\n*{grid.get('caveat')}*")

        heatmap = valuation.sensitivity_heatmap(price["last"], fund.get("forward_eps"), fund.get("forward_pe"))
        if "error" not in heatmap:
            lines.append("\n### Full Sensitivity Heatmap — implied price by EPS% x Multiple%")
            header = "| EPS% ↓ / P/E% → | " + " | ".join(f"{p}%" for p in heatmap["pe_pcts_of_current"]) + " |"
            lines.append(header)
            lines.append("|" + "---|" * (len(heatmap["pe_pcts_of_current"]) + 1))
            for row in heatmap["rows"]:
                cells = " | ".join(_fmt(row["prices"][p]) for p in heatmap["pe_pcts_of_current"])
                lines.append(f"| {row['eps_pct_of_forward']}% | {cells} |")
            lines.append(f"\n*Current price for reference: {_fmt(heatmap['current_price'])}. {heatmap.get('caveat')}*")
    else:
        lines.append("\n## Fundamentals")
        lines.append(f"n/a — {sc.get('asset_class_note', 'not an operating equity')}")
        extra = []
        if fund.get("circulating_supply"):
            extra.append(("Circulating supply", _fmt(fund.get("circulating_supply"))))
        if fund.get("fund_category"):
            extra.append(("Fund category", fund.get("fund_category")))
        if extra:
            lines += _kv_table(extra)

    if peers:
        lines.append(f"\n## Peer Comparison: {ticker} vs {', '.join(p.upper() for p in peers)}")
        lines.append("| Ticker | Last | Fwd P/E | Margin | Growth | Trend |")
        lines.append("|---|---|---|---|---|---|")
        for row in compare.compare_tickers([ticker, *peers]):
            lines.append(
                f"| {row['ticker']} | {_fmt(row.get('last'))} | {_fmt(row.get('forward_pe'))} | "
                f"{_fmt(row.get('profit_margin'))} | {_fmt(row.get('revenue_growth'))} | "
                f"{row.get('trend', 'n/a')} |"
            )

    lines.append("\n## News")
    if b.get("news_summary"):
        lines.append(f"**Aggregate sentiment:** {b.get('sentiment')}\n")
        for line in b["news_summary"]:
            lines.append(f"- {line}")
    else:
        lines.append("No news available." + (f" ({b['news_error']})" if b.get("news_error") else ""))

    lines.append("\n## Macro Context (Polymarket)")
    if b.get("polymarket_related"):
        lines.append("| Market | Odds | Volume |")
        lines.append("|---|---|---|")
        for m in b["polymarket_related"]:
            lines.append(f"| {m['market']} | {_fmt(m.get('odds'))} | {_fmt(m.get('volume'))} |")
    else:
        lines.append("No related markets found.")

    lines.append("\n## Deterministic Scorecard *(code-computed — report verbatim)*")
    lines += _kv_table(
        [
            ("Fundamental", _fmt(sc["fundamental"])),
            ("Technical", _fmt(sc["technical"])),
            ("Valuation", _fmt(sc["valuation"])),
            ("Sentiment", _fmt(sc["sentiment"])),
            ("**Overall**", f"**{_fmt(sc['overall'])}**"),
            ("**Decision**", f"**{sc['decision']}**"),
            ("Suggested stop (entry − 2×ATR)", _fmt(sc["suggested_stop_price"])),
        ]
    )
    if sc.get("risk_flags"):
        lines.append("\n### Risk / Fragility Layer *(a different question than the score above)*")
        rf = sc["risk_flags"]
        lines += _kv_table(
            [
                ("Cyclicality risk", rf.get("cyclicality_risk", "UNKNOWN")),
                ("Momentum extension risk", rf.get("momentum_extension_risk", "UNKNOWN")),
                ("Volatility risk", rf.get("volatility_risk", "UNKNOWN")),
                ("**Risk-adjusted conviction**", f"**{sc.get('risk_adjusted_conviction', 'UNKNOWN')}**"),
            ]
        )
    if sc.get("asset_class_note"):
        lines.append(f"\n> {sc['asset_class_note']}")
    lines.append(f"\n*Methodology: {sc['methodology']}*")

    lines.append("\n## Strategy Screen — 6 published systems, ranked by Sharpe")
    scan_results = backtest.scan(ticker)
    lines.append("| Strategy | Sharpe | CAGR | vs BuyHold CAGR | Max DD | Win % | Profit Factor | Trades |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in scan_results:
        if "error" in r:
            lines.append(f"| {r['template']} | ERROR: {r['error']} | | | | | | |")
            continue
        lines.append(
            f"| {r['template']} | {_fmt(r['sharpe_ratio'])} | {_fmt(r['strategy_cagr_pct'])}% | "
            f"{_fmt(r['buy_hold_cagr_pct'])}% | {_fmt(r['max_drawdown_pct'])}% | "
            f"{_fmt(r['win_rate_pct'])}% | {_fmt(r['profit_factor'])} | {r['trades']} |"
        )

    top = next((r for r in scan_results if "error" not in r), None)
    if top:
        lines.append(f"\n### Full Backtest Detail — top-ranked: {top['template']}")
        full = backtest.run_template(ticker, top["template"])
        lines += _kv_table(
            [
                ("Period", full.get("period")),
                ("Trades", full.get("trades")),
                ("Total return (strategy)", _fmt(full.get("total_return_pct"), "%")),
                ("Total return (buy & hold)", _fmt(full.get("buy_hold_return_pct"), "%")),
                ("CAGR (strategy)", _fmt(full.get("strategy_cagr_pct"), "%")),
                ("CAGR (buy & hold)", _fmt(full.get("buy_hold_cagr_pct"), "%")),
                ("Return difference vs buy&hold (pts)", _fmt(full.get("return_difference_vs_buy_hold_pct"))),
                ("Outperformed buy & hold?", _fmt(full.get("outperformed_buy_and_hold"))),
                ("Sharpe / Sortino", f"{_fmt(full.get('sharpe_ratio'))} / {_fmt(full.get('sortino_ratio'))}"),
                ("Annualized volatility", _fmt(full.get("annualized_volatility_pct"), "%")),
                ("Time in market", _fmt(full.get("time_in_market_pct"), "%")),
                ("Max drawdown", _fmt(full.get("max_drawdown_pct"), "%")),
                ("Drawdown reflects stop-loss?", _fmt(full.get("drawdown_reflects_stop_loss"))),
                ("Win rate", _fmt(full.get("win_rate_pct"), "%")),
                ("Expectancy ($/trade)", _fmt(full.get("expectancy"))),
                ("Profit factor", _fmt(full.get("profit_factor"))),
                ("Avg win / avg loss", f"{_fmt(full.get('avg_win_pct'))}% / {_fmt(full.get('avg_loss_pct'))}%"),
                ("Stop-loss used", _fmt(full.get("stop_loss_pct"), "%")),
            ]
        )
        lines.append(f"\n*{full.get('sample_size_note')}*")
        lines.append(f"\n*{full.get('caveat')}*")

        try:
            oos = backtest.run_out_of_sample(ticker, get_spec(top["template"]))
        except Exception:
            oos = None
        if oos and "error" not in oos:
            lines.append(f"\n### Out-of-Sample Check — {top['template']} (does the edge hold on unseen data?)")
            lines.append(f"Split date: {oos['split_date']} ({int(oos['split_pct'] * 100)}% in-sample / {int((1 - oos['split_pct']) * 100)}% holdout)")
            lines.append("| Window | Period | Trades | Sharpe | CAGR | vs BuyHold CAGR | Win % |")
            lines.append("|---|---|---|---|---|---|---|")
            for label, seg in (("In-sample", oos["in_sample"]), ("Out-of-sample", oos["out_of_sample"])):
                lines.append(
                    f"| {label} | {seg.get('period')} | {seg.get('trades')} | {_fmt(seg.get('sharpe_ratio'))} | "
                    f"{_fmt(seg.get('strategy_cagr_pct'))}% | {_fmt(seg.get('buy_hold_cagr_pct'))}% | {_fmt(seg.get('win_rate_pct'))}% |"
                )
            degraded = oos.get("sharpe_degraded_out_of_sample")
            flag = "⚠️ **DEGRADED** — possible overfitting to the in-sample window." if degraded else (
                "✅ Held up — not proof of a persistent edge, but not an overfitting red flag either." if degraded is False else ""
            )
            if flag:
                lines.append(f"\n{flag}")
            lines.append(f"\n*{oos.get('note')}*")

    tl = scoring.trade_levels(price["last"], price["atr14"])
    if "error" not in tl:
        lines.append("\n## Trade Levels — hypothetical LONG entry, stop, and take-profit (not a recommendation to enter)")
        lines += _kv_table(
            [
                ("Entry (latest close)", _fmt(tl["entry"])),
                ("Stop", _fmt(tl["stop"])),
                ("Risk per share", _fmt(tl["risk_per_share"])),
                (f"Take-profit 1 ({tl['take_profit_1_r_multiple']}R)", _fmt(tl["take_profit_1"])),
                (f"Take-profit 2 ({tl['take_profit_2_r_multiple']}R)", _fmt(tl["take_profit_2"])),
                ("Risk/reward to TP1", f"{tl['risk_reward_ratio_tp1']}:1"),
            ]
        )
        lines.append(f"\n**Invalidation:** {tl['invalidation']}")
        lines.append(f"\n*{tl['note']}*")

    if account_size:
        entry = price["last"]
        stop = sc["suggested_stop_price"]
        lines.append(f"\n## Position Sizing — ${account_size:,.0f} account, {risk_pct}% risk")
        if entry and stop:
            sizing = scoring.position_size(entry, stop, account_size, risk_pct)
            rows = [(k.replace("_", " ").title(), _fmt(v)) for k, v in sizing.items() if v is not None and k not in ("note", "warning")]
            lines += _kv_table(rows)
            if sizing.get("note"):
                lines.append(f"\n*{sizing['note']}*")
            if sizing.get("warning"):
                lines.append(f"\n⚠️ {sizing['warning']}")
        else:
            lines.append("Cannot size — missing entry price or stop.")

    positions = portfolio.list_positions()
    if positions:
        lines.append("\n## Your Tracked Portfolio")
        summary = portfolio.portfolio_summary()
        lines.append("| Ticker | Shares | Cost | P&L | P&L % | Weight % |")
        lines.append("|---|---|---|---|---|---|")
        for p in summary["positions"]:
            lines.append(
                f"| {p['ticker']} | {p['shares']} | {_fmt(p['cost_basis'])} | "
                f"{_fmt(p['unrealized_pl'])} | {_fmt(p['unrealized_pl_pct'])}% | "
                f"{_fmt(p['portfolio_weight_pct'])}% |"
            )
        lines.append(f"\n**Total P&L:** {_fmt(summary['total_pl'])} ({_fmt(summary['total_pl_pct'])}%)")
        if summary.get("sector_concentration_pct"):
            conc = ", ".join(f"{k} {v:.0f}%" for k, v in summary["sector_concentration_pct"].items())
            lines.append(f"\n**Sector concentration:** {conc}")

    lines.append(f"\n---\n*End of data package for {ticker}. Now respond using the structure requested above.*")

    return "\n".join(lines)
