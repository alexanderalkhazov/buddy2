"""Multi-ticker screening and peer/sector aggregation.

Reuses the exact same deterministic bundle/scorecard pipeline as a single-ticker
report — this is not a separate scoring system, just the same one run N times and
laid out side by side. News is skipped by default (use_news=False) purely for
speed across a list; sentiment_score falls back to its neutral default when no
news is passed, same as any other missing-input case in scoring.py.
"""

from __future__ import annotations

from processing import bundle as bundle_mod


def screen(tickers: list[str], use_news: bool = False, refresh: bool = False) -> list[dict]:
    """Run the scorecard pipeline across a list of tickers and rank by overall score.
    Not a recommendation to buy the top-ranked name — it's "which of these currently
    scores highest on this system's fixed formula," nothing more."""
    rows = []
    for ticker in tickers:
        try:
            b = bundle_mod.build(ticker, include_news=use_news, include_polymarket=False, refresh=refresh)
        except Exception as exc:
            rows.append({"ticker": ticker.upper(), "error": f"{type(exc).__name__}: {exc}"})
            continue
        sc = b["scorecard"]
        fund = b.get("fundamentals", {})
        price = b["price"]
        rows.append(
            {
                "ticker": ticker.upper(),
                "asset_class": b.get("asset_class"),
                "last_price": price.get("last"),
                "change_pct": price.get("change_pct"),
                "overall_score": sc.get("overall"),
                "decision": sc.get("decision"),
                "fundamental_score": sc.get("fundamental"),
                "technical_score": sc.get("technical"),
                "valuation_score": sc.get("valuation"),
                "sentiment_score": sc.get("sentiment"),
                "risk_adjusted_conviction": sc.get("risk_adjusted_conviction"),
                "forward_pe": fund.get("forward_pe"),
                "revenue_growth": fund.get("revenue_growth"),
                "profit_margin": fund.get("profit_margin"),
                "beta": fund.get("beta"),
            }
        )

    ranked = sorted(
        (r for r in rows if "error" not in r),
        key=lambda r: r["overall_score"] if r["overall_score"] is not None else -1,
        reverse=True,
    )
    return ranked + [r for r in rows if "error" in r]


def _avg(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def sector_summary(tickers: list[str], refresh: bool = False) -> dict:
    """Peer-group averages a single ticker's numbers can be read against, instead of
    judged in isolation. Not weighted by market cap — a simple mean across the
    supplied list, so the caller controls what "peer group" means."""
    fetched = []
    for ticker in tickers:
        try:
            from data import fundamentals

            fund = fundamentals.fetch_fundamentals(ticker)
            fetched.append({"ticker": ticker.upper(), **fund})
        except Exception as exc:
            fetched.append({"ticker": ticker.upper(), "error": f"{type(exc).__name__}: {exc}"})

    valid = [f for f in fetched if "error" not in f]
    return {
        "tickers": [f["ticker"] for f in valid],
        "peer_count": len(valid),
        "avg_forward_pe": _avg([f.get("forward_pe") for f in valid]),
        "avg_trailing_pe": _avg([f.get("trailing_pe") for f in valid]),
        "avg_profit_margin": _avg([f.get("profit_margin") for f in valid]),
        "avg_revenue_growth": _avg([f.get("revenue_growth") for f in valid]),
        "avg_beta": _avg([f.get("beta") for f in valid]),
        "avg_ev_to_ebitda": _avg([f.get("ev_to_ebitda") for f in valid]),
        "avg_price_to_sales": _avg([f.get("price_to_sales") for f in valid]),
        "avg_debt_to_equity": _avg([f.get("debt_to_equity") for f in valid]),
        "per_ticker": fetched,
        "note": (
            "Unweighted mean across the supplied tickers — you choose what 'peer "
            "group' means by which tickers you pass in. Use to read a single "
            "ticker's multiple against this group, not as an index-level truth."
        ),
    }
