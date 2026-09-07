"""Context bundle assembly.

This is the ONLY structure the LLM layer is allowed to see. Every data source is
reduced to a handful of scalars or one-line summaries here; raw frames, article
bodies, and API payloads stop at this boundary.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from data import edgar, fundamentals, news, polymarket
from processing import news_proc, scoring
from processing.indicators import snapshot
from storage import cache


def build(
    ticker: str,
    include_news: bool = True,
    include_polymarket: bool = True,
    include_fundamentals: bool = True,
    include_filings: bool = False,
    refresh: bool = False,
) -> dict:
    """Assemble the compact per-ticker bundle. Failures degrade to None, never raise."""
    ticker = ticker.upper()

    df = cache.get_ohlcv(ticker, force=refresh)
    snap = snapshot(ticker, df)

    as_of_date = datetime.strptime(snap.as_of, "%Y-%m-%d").date()
    data_age_days = (date.today() - as_of_date).days

    bundle: dict = {
        "ticker": ticker,
        "as_of": snap.as_of,
        # This system has daily closing bars only, never real-time intraday quotes —
        # even refresh=true only pulls the latest COMPLETED close, not a live price.
        # data_age_days counts calendar days since that close (weekends inflate it;
        # 2-3 is normal after a weekend, not necessarily stale).
        "data_freshness": {
            "as_of": snap.as_of,
            "data_age_days": data_age_days,
            "is_stale": data_age_days > 1,
            "note": (
                "Daily closing price only, not a real-time quote. Never call this "
                "'the current price' — say 'latest available close, as of {date}'."
            ),
        },
        "price": {
            "last": snap.last,
            "change_pct": snap.change_pct,
            "rsi14": snap.rsi14,
            "macd_hist": snap.macd_hist,
            "sma20": snap.sma20,
            "sma50": snap.sma50,
            "sma200": snap.sma200,
            "atr14": snap.atr14,
            "vol_vs_30d_avg": snap.vol_vs_30d_avg,
            "trend": snap.trend,
        },
    }

    processed_news: list[dict] = []
    fund: dict = {}

    if include_news:
        try:
            if refresh or cache.is_stale(ticker, "news", cache.NEWS_TTL):
                processed_news = news_proc.process(news.fetch_news(ticker))
                cache.store_news(ticker, processed_news)
            else:
                processed_news = cache.load_news(ticker)
            bundle["news_summary"] = [n["summary"] for n in processed_news]
            bundle["sentiment"] = news_proc.aggregate_sentiment(processed_news)
        except Exception as exc:
            bundle["news_summary"] = []
            bundle["sentiment"] = "unknown"
            bundle["news_error"] = str(exc)[:120]

    if include_polymarket:
        try:
            bundle["polymarket_related"] = polymarket.macro_context()
        except Exception as exc:
            bundle["polymarket_related"] = []
            bundle["polymarket_error"] = str(exc)[:120]

    if include_fundamentals:
        try:
            fund = fundamentals.fetch_fundamentals(ticker)
            bundle["fundamentals"] = fund
        except Exception as exc:
            bundle["fundamentals"] = {}
            bundle["fundamentals_error"] = str(exc)[:120]

    if include_filings:
        try:
            bundle["recent_filings"] = edgar.recent_filings(ticker)
        except Exception as exc:
            bundle["recent_filings"] = []
            bundle["filings_error"] = str(exc)[:120]

    asset_class = fund.get("asset_class", "UNKNOWN")
    bundle["asset_class"] = asset_class

    # Deterministic — computed by code, not the model. The model reports these
    # numbers; it does not invent its own confidence scores. valuation uses the
    # sector-agnostic P/E fallback here (no peer list in a single-ticker bundle) —
    # call compare_tickers for a true peer-relative valuation score instead.
    scorecard = scoring.compute_scorecard(
        rsi14=snap.rsi14,
        macd_hist=snap.macd_hist,
        trend=snap.trend,
        profit_margin=fund.get("profit_margin"),
        revenue_growth=fund.get("revenue_growth"),
        forward_pe=fund.get("forward_pe"),
        peer_forward_pes=None,
        news_items=processed_news,
        last_price=snap.last,
        atr14=snap.atr14,
        sector=fund.get("sector"),
        industry=fund.get("industry"),
        beta=fund.get("beta"),
        fifty_two_week_high=fund.get("fifty_two_week_high"),
        sma20=snap.sma20,
        sma50=snap.sma50,
    ).to_dict()

    if asset_class not in ("EQUITY", "UNKNOWN"):
        # fundamental/valuation are already None (no P/E or margins exist for a
        # commodity, index, currency, or crypto asset) — say so explicitly rather
        # than leaving a silent null the model has to guess the reason for.
        scorecard["asset_class_note"] = (
            f"'{asset_class}' assets have no earnings or P/E — fundamental and "
            "valuation scores are not applicable, not missing data. The overall "
            "score is technical + sentiment only, reweighted across just those two."
        )

    bundle["scorecard"] = scorecard

    return bundle


def history_window(ticker: str, days: int = 90) -> dict:
    """Coarse price-path summary for questions about a specific timeframe."""
    df = cache.get_ohlcv(ticker)
    window = df[df.index >= df.index.max() - timedelta(days=days)]
    if window.empty:
        return {"ticker": ticker.upper(), "days": days, "error": "no data in window"}

    first, last = float(window["close"].iloc[0]), float(window["close"].iloc[-1])
    return {
        "ticker": ticker.upper(),
        "days": days,
        "start": str(window.index.min().date()),
        "end": str(window.index.max().date()),
        "return_pct": round((last / first - 1) * 100, 2),
        "high": round(float(window["high"].max()), 2),
        "low": round(float(window["low"].min()), 2),
        "avg_volume": round(float(window["volume"].mean())),
    }
