"""Fundamentals via yfinance: valuation, earnings, and asset-class-appropriate stats.

Works across asset classes (equity, ETF, index, futures/commodities, forex, crypto),
not just stocks. quoteType drives which fields are even meaningful — a forward P/E on
gold futures is a category error, not a missing data point, so equity-only fields are
None (not fetched) for non-equity assets rather than silently attempted and failing.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

# yfinance's quoteType values. EQUITY is the only class with P/E, margins, earnings,
# and analyst targets — attempting those calls on other classes wastes a request and
# yfinance logs a noisy error for each, so we skip them entirely instead of catching
# a predictable failure after the fact.
EQUITY = "EQUITY"


def _epoch_to_date(value) -> str | None:
    if value is None:
        return None
    try:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(int(value), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return None


def _num(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else round(out, 4)  # drop NaN


def fetch_fundamentals(ticker: str) -> dict:
    """Compact snapshot, shaped by asset class. Missing fields come back as None
    rather than absent, so callers can always index the same keys."""
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:  # yfinance raises assorted network/parse errors
        info = {}

    asset_class = info.get("quoteType") or "UNKNOWN"

    out = {
        "asset_class": asset_class,
        # Universal — meaningful for every asset class yfinance covers.
        "market_cap": _num(info.get("marketCap")),
        "volume": _num(info.get("volume") or info.get("regularMarketVolume")),
        "fifty_two_week_low": _num(info.get("fiftyTwoWeekLow")),
        "fifty_two_week_high": _num(info.get("fiftyTwoWeekHigh")),
        # Equity-only — an operating company's business fundamentals.
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "trailing_pe": _num(info.get("trailingPE")),
        "forward_pe": _num(info.get("forwardPE")),
        "trailing_eps": _num(info.get("trailingEps")),
        "forward_eps": _num(info.get("forwardEps")),
        "profit_margin": _num(info.get("profitMargins")),
        "revenue_growth": _num(info.get("revenueGrowth")),
        "beta": _num(info.get("beta")),
        "dividend_yield": _num(info.get("dividendYield")),
        "earnings_surprise_pct": None,
        "next_earnings_date": None,
        "last_fiscal_year_end": _epoch_to_date(info.get("lastFiscalYearEnd")),
        "next_fiscal_year_end": _epoch_to_date(info.get("nextFiscalYearEnd")),
        "most_recent_quarter": _epoch_to_date(info.get("mostRecentQuarter")),
        # Income statement / cash flow — real fields from the same .info payload,
        # no extra API call.
        "revenue": _num(info.get("totalRevenue")),
        "gross_profit": _num(info.get("grossProfits")),
        "net_income": _num(info.get("netIncomeToCommon")),
        "operating_cashflow": _num(info.get("operatingCashflow")),
        "free_cashflow": _num(info.get("freeCashflow")),
        # Balance sheet
        "total_cash": _num(info.get("totalCash")),
        "total_debt": _num(info.get("totalDebt")),
        "debt_to_equity": _num(info.get("debtToEquity")),
        # Valuation, beyond P/E
        "price_to_sales": _num(info.get("priceToSalesTrailing12Months")),
        "price_to_book": _num(info.get("priceToBook")),
        "peg_ratio": _num(info.get("pegRatio") or info.get("trailingPegRatio")),
        "ev_to_ebitda": _num(info.get("enterpriseToEbitda")),
        "ev_to_revenue": _num(info.get("enterpriseToRevenue")),
        "ebitda": _num(info.get("ebitda")),
        "current_ratio": _num(info.get("currentRatio")),
        "quick_ratio": _num(info.get("quickRatio")),
        "earnings_growth": _num(info.get("earningsGrowth")),
        # Company identity
        "employees": _num(info.get("fullTimeEmployees")),
        "exchange": info.get("fullExchangeName") or info.get("exchange"),
        # Ownership/positioning — real signals for "how crowded/extended is this
        # trade," cheap to include since they're already in the same .info payload.
        "short_percent_of_float": _num(info.get("shortPercentOfFloat")),
        "short_ratio": _num(info.get("shortRatio")),
        "held_percent_insiders": _num(info.get("heldPercentInsiders")),
        "held_percent_institutions": _num(info.get("heldPercentInstitutions")),
        # Crypto-only.
        "circulating_supply": _num(info.get("circulatingSupply")),
        # ETF/mutual-fund-only.
        "fund_category": info.get("category"),
        # Deliberately its own nested object, not flat alongside the valuation
        # fields above — analyst targets/ratings are external sentiment, not
        # evidence of intrinsic value. They must never be cited under "Supporting
        # signals" in the LLM's output; the nesting exists to make that mistake
        # structurally awkward, not just prompt-discouraged. Equity-only — analyst
        # coverage doesn't exist for commodities/forex/crypto/indices.
        "external_sentiment": {
            "analyst_mean_target": _num(info.get("targetMeanPrice")),
            "analyst_median_target": _num(info.get("targetMedianPrice")),
            "analyst_high_target": _num(info.get("targetHighPrice")),
            "analyst_low_target": _num(info.get("targetLowPrice")),
            "analyst_recommendation": info.get("recommendationKey"),
            "analyst_count": info.get("numberOfAnalystOpinions"),
        }
        if asset_class == EQUITY
        else {},
    }

    # yfinance's priceToSalesTrailing12Months is sometimes None even when both
    # inputs it needs are present (observed on MU) — derive it ourselves rather
    # than pass through a false "not available" when market_cap/revenue exist.
    if out["price_to_sales"] is None and out["market_cap"] and out["revenue"]:
        out["price_to_sales"] = round(out["market_cap"] / out["revenue"], 4)

    net_debt = None
    if out["total_debt"] is not None and out["total_cash"] is not None:
        net_debt = round(out["total_debt"] - out["total_cash"], 4)
    out["net_debt"] = net_debt

    net_debt_to_ebitda = None
    if net_debt is not None and out["ebitda"]:
        net_debt_to_ebitda = round(net_debt / out["ebitda"], 4)
    out["net_debt_to_ebitda"] = net_debt_to_ebitda

    if asset_class != EQUITY:
        return out

    try:
        hist = t.earnings_history
        if hist is not None and not hist.empty and "surprisePercent" in hist:
            last = hist.dropna(subset=["surprisePercent"]).iloc[-1]
            # yfinance's surprisePercent is a fraction (0.0554 == 5.54%), not
            # already a percent — multiply here so every caller can safely
            # append "%" without a silent 10x-off display.
            out["earnings_surprise_pct"] = _num(last["surprisePercent"] * 100)
            out["last_eps_actual"] = _num(last.get("epsActual"))
            out["last_eps_estimate"] = _num(last.get("epsEstimate"))
    except Exception:
        pass

    try:
        calendar = t.calendar
        dates = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
        if dates:
            out["next_earnings_date"] = str(dates[0])
    except Exception:
        pass

    return out


def eps_revision_history(ticker: str, period: str = "+1y") -> dict:
    """How much (and how recently) analysts have moved their EPS estimate — the
    difference between "the market has believed $155 for a year" and "the market
    revised up from $102 to $155 in the last 90 days" that a single current-forward-
    EPS number can't show on its own.

    period: yfinance's own period codes — '0q' (current quarter), '+1q' (next
    quarter), '0y' (current fiscal year), '+1y' (next fiscal year, the period that
    corresponds to fetch_fundamentals' forward_eps for most tickers).
    """
    t = yf.Ticker(ticker)
    try:
        trend = t.eps_trend
        revisions = t.eps_revisions
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    if trend is None or trend.empty or period not in trend.index:
        return {"error": f"No EPS trend data for period {period!r}"}

    row = trend.loc[period]
    current = _num(row.get("current"))
    snapshots = {
        "current": current,
        "7_days_ago": _num(row.get("7daysAgo")),
        "30_days_ago": _num(row.get("30daysAgo")),
        "60_days_ago": _num(row.get("60daysAgo")),
        "90_days_ago": _num(row.get("90daysAgo")),
    }

    def _pct_change(old: float | None) -> float | None:
        if not current or not old or old == 0:
            return None
        return round((current / old - 1) * 100, 2)

    out = {
        "period": period,
        "eps_estimates": snapshots,
        "revision_pct_7d": _pct_change(snapshots["7_days_ago"]),
        "revision_pct_30d": _pct_change(snapshots["30_days_ago"]),
        "revision_pct_90d": _pct_change(snapshots["90_days_ago"]),
    }

    if revisions is not None and not revisions.empty and period in revisions.index:
        rrow = revisions.loc[period]
        out["analysts_revising_up_last_30d"] = int(rrow.get("upLast30days") or 0)
        out["analysts_revising_down_last_30d"] = int(rrow.get("downLast30days") or 0)

    rev_90d = out["revision_pct_90d"]
    if rev_90d is not None:
        if abs(rev_90d) < 5:
            out["stability_note"] = "Estimate has been stable over the last 90 days — this is a settled consensus, not a moving target."
        elif rev_90d > 0:
            out["stability_note"] = f"Estimate has risen {rev_90d:+.1f}% in 90 days — a live upward revision cycle, not a long-settled number. Recent, fast-moving revisions are inherently less proven than a stable estimate."
        else:
            out["stability_note"] = f"Estimate has fallen {rev_90d:+.1f}% in 90 days — analysts are cutting expectations, worth weighing against any bullish valuation read."

    return out


def estimate_dispersion(ticker: str, period: str = "+1y") -> dict:
    """How much analysts disagree, not just what they average to. A 155.64 forward
    EPS consensus built from estimates ranging 106.89-221.27 is a very different
    claim than one where every analyst is within a few points of the average — the
    average alone can't show that. period: yfinance period codes, see
    eps_revision_history."""
    t = yf.Ticker(ticker)
    try:
        eps_est = t.earnings_estimate
        rev_est = t.revenue_estimate
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    if eps_est is None or eps_est.empty or period not in eps_est.index:
        return {"error": f"No estimate data for period {period!r}"}

    try:
        fund_dates = fetch_fundamentals(ticker)
    except Exception:
        fund_dates = {}

    e = eps_est.loc[period]
    out = {
        "period": period,
        "last_fiscal_year_end": fund_dates.get("last_fiscal_year_end"),
        "next_fiscal_year_end": fund_dates.get("next_fiscal_year_end"),
        "most_recent_quarter": fund_dates.get("most_recent_quarter"),
        "eps_estimate_low": _num(e.get("low")),
        "eps_estimate_avg": _num(e.get("avg")),
        "eps_estimate_high": _num(e.get("high")),
        "eps_estimate_count": int(e.get("numberOfAnalysts")) if e.get("numberOfAnalysts") is not None else None,
        "eps_estimate_growth_pct": _num(e.get("growth") * 100 if e.get("growth") is not None else None),
    }

    if rev_est is not None and not rev_est.empty and period in rev_est.index:
        r = rev_est.loc[period]
        out["revenue_estimate_low"] = _num(r.get("low"))
        out["revenue_estimate_avg"] = _num(r.get("avg"))
        out["revenue_estimate_high"] = _num(r.get("high"))
        out["revenue_estimate_count"] = int(r.get("numberOfAnalysts")) if r.get("numberOfAnalysts") is not None else None
        if out["revenue_estimate_low"] and out["revenue_estimate_high"] and out["revenue_estimate_avg"]:
            out["revenue_estimate_spread_pct_of_avg"] = round(
                (out["revenue_estimate_high"] - out["revenue_estimate_low"]) / abs(out["revenue_estimate_avg"]) * 100, 1
            )
        fy_note = (
            f" This company's next fiscal year ends {out['next_fiscal_year_end']}"
            f" (last fiscal year end: {out['last_fiscal_year_end']}, most recent "
            f"reported quarter: {out['most_recent_quarter']})."
            if out.get("next_fiscal_year_end")
            else ""
        )
        out["revenue_estimate_period_note"] = (
            f"'{period}' is yfinance's next-fiscal-year estimate, a different window than "
            "trailing-twelve-month revenue reported elsewhere in this system — for a cyclical "
            "business mid-upswing, a large gap between TTM revenue and this estimate can be "
            "real (analysts expecting the current run-rate to continue/grow through the next "
            f"fiscal year), not a unit or period mismatch bug.{fy_note}"
        )

    if out["eps_estimate_low"] and out["eps_estimate_high"] and out["eps_estimate_avg"]:
        spread_pct = round((out["eps_estimate_high"] - out["eps_estimate_low"]) / abs(out["eps_estimate_avg"]) * 100, 1)
        out["eps_estimate_spread_pct_of_avg"] = spread_pct
        out["dispersion_note"] = (
            f"EPS estimates for {period} range {out['eps_estimate_low']}-{out['eps_estimate_high']} "
            f"({out['eps_estimate_count']} analysts) — a spread of {spread_pct}% of the average estimate. "
            + ("Wide dispersion — the consensus figure hides real disagreement about outcomes."
               if spread_pct >= 30 else
               "Relatively tight dispersion — analysts broadly agree on the range.")
        )

    return out


def earnings_reaction_history(ticker: str, limit: int = 8) -> dict:
    """How did the stock actually move in the days after each of its last N
    reported earnings? Distinct from earnings_surprise_pct alone — a company can
    beat estimates and still sell off (e.g. on weak guidance), so this checks the
    ACTUAL subsequent price action against the CACHED daily-close history already
    in this system, not just the EPS-beat number. Uses yfinance's earnings_dates
    (exact announcement timestamps), not earnings_history's fiscal-quarter-end
    dates, which are not the same as the day the market actually reacted."""
    t = yf.Ticker(ticker)
    try:
        dates_df = t.earnings_dates
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    if dates_df is None or dates_df.empty:
        return {"error": "no earnings date history available"}

    reported = dates_df.dropna(subset=["Reported EPS"]).sort_index(ascending=False).head(limit)
    if reported.empty:
        return {"error": "no past reported earnings found"}

    from storage.cache import get_ohlcv

    try:
        prices = get_ohlcv(ticker)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    events = []
    for ts, row in reported.iterrows():
        report_date = ts.tz_localize(None) if ts.tzinfo else ts
        # Earnings announced after market close (>=16:00) react on the NEXT trading
        # day; before/during market hours react same day. This matters — using the
        # wrong anchor day silently shifts every subsequent return by one session.
        anchor_date = report_date.normalize() + pd.Timedelta(days=1 if ts.hour >= 16 else 0)
        after = prices[prices.index >= anchor_date]
        before = prices[prices.index < anchor_date]
        if after.empty or before.empty:
            continue
        base_close = float(before["close"].iloc[-1])

        reaction = {"report_date": str(report_date.date()), "eps_surprise_pct": _num(row.get("Surprise(%)"))}
        for label, n_days in (("1d", 1), ("5d", 5), ("10d", 10)):
            window = after.iloc[: n_days]
            if len(window) < n_days or base_close == 0:
                reaction[f"return_{label}_pct"] = None
                continue
            reaction[f"return_{label}_pct"] = round((float(window["close"].iloc[-1]) / base_close - 1) * 100, 2)
        events.append(reaction)

    if not events:
        return {"error": "no earnings events overlap with cached price history"}

    def _agg(field: str) -> dict:
        vals = [e[field] for e in events if e.get(field) is not None]
        if not vals:
            return {"median_pct": None, "positive_pct_of_events": None, "n": 0}
        return {
            "median_pct": round(sorted(vals)[len(vals) // 2], 2),
            "positive_pct_of_events": round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1),
            "n": len(vals),
        }

    return {
        "ticker": ticker.upper(),
        "events": events,
        "aggregate": {
            "return_1d": _agg("return_1d_pct"),
            "return_5d": _agg("return_5d_pct"),
            "return_10d": _agg("return_10d_pct"),
        },
        "note": (
            f"Actual price reaction over the {len(events)} most recent reported earnings dates "
            "found in cached price history, anchored the trading day after an after-close report "
            "(or same-day for a before-close/premarket report). This is real historical price "
            "action, not a forecast of how the NEXT earnings reaction will go — sample size is "
            "small (typically under 10 events) and each company's guidance/market conditions "
            "differ event to event. A stock can beat EPS estimates and still sell off (weak "
            "guidance, 'sell the news') — eps_surprise_pct and return_1d_pct are reported "
            "separately for exactly this reason; do not assume they move together."
        ),
    }


def insider_activity(ticker: str, limit: int = 10) -> dict:
    """Recent insider buy/sell activity — a real, code-computed signal for whether
    the people who know the business best are net buying or selling. Separate from
    fetch_fundamentals since it's its own API call; call on demand, not by default."""
    t = yf.Ticker(ticker)
    try:
        df = t.insider_transactions
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    if df is None or df.empty:
        return {"transactions": [], "net_shares": None, "buy_count": 0, "sell_count": 0}

    buys = sells = 0
    net_shares = 0.0
    transactions = []
    for _, row in df.head(limit).iterrows():
        shares = row.get("Shares")
        # yfinance's "Transaction" column is empty in this feed for most tickers —
        # the actual description ("Sale at price...", "Stock Award(Grant)...") is
        # in "Text" instead. Fall back to "Transaction" in case a different feed
        # populates it.
        description = str(row.get("Text") or row.get("Transaction") or "").strip()
        label = description.split(" at price")[0].strip() or "Unknown"
        desc_lower = description.lower()
        # Only real open-market transactions count as buy/sell signal — an RSU
        # award or a gift isn't the insider expressing a market view the way an
        # open-market purchase or sale is, so those are shown but not counted.
        is_buy = "purchase" in desc_lower or ("buy" in desc_lower and "buyback" not in desc_lower)
        is_sell = "sale" in desc_lower or "sold" in desc_lower
        if is_buy:
            buys += 1
        elif is_sell:
            sells += 1
        try:
            shares_num = float(shares) if shares is not None else 0.0
        except (TypeError, ValueError):
            shares_num = 0.0
        net_shares += shares_num if is_buy else (-shares_num if is_sell else 0.0)
        transactions.append(
            {
                "insider": row.get("Insider"),
                "position": row.get("Position"),
                "transaction": label,
                "is_open_market": is_buy or is_sell,
                "shares": _num(shares),
                "start_date": str(row.get("Start Date")),
            }
        )

    return {
        "transactions": transactions,
        "buy_count": buys,
        "sell_count": sells,
        "net_shares": round(net_shares, 2),
        "note": (
            f"Most recent {len(transactions)} filed transactions — not necessarily "
            "all within a fixed recent window; check start_date on each. "
            "buy_count/sell_count/net_shares only count open-market purchases/sales "
            "(is_open_market=true) — RSU vests, option-grant awards, and gifts are "
            "shown but excluded, since they aren't the insider expressing a market "
            "view the way a real purchase or sale is."
        ),
    }


def compare(tickers: list[str]) -> list[dict]:
    """Side-by-side fundamentals for relative/peer valuation questions."""
    return [{"ticker": t.upper(), **fetch_fundamentals(t)} for t in tickers]
