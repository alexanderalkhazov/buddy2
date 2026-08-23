"""Market/volatility regime — is the broad tape bullish, bearish, or choppy right
now, and is volatility calm or elevated? A stock's own technicals mean something
different in a risk-on, low-vol tape than in a risk-off, high-vol one; this module
answers that question independently of any single ticker.

Everything here reuses processing.indicators.snapshot() on benchmark tickers
(SPY, QQQ, ^VIX) already fetchable through the same cached daily-OHLCV path as
any other ticker — no new data source, no intraday data, no options data. VIX
level/trend is a real, standard volatility-regime proxy; there is no proprietary
"breadth" or advance/decline data in this system, so regime here means index-level
trend + VIX only, not full market internals.
"""

from __future__ import annotations

from processing.indicators import snapshot
from storage.cache import get_ohlcv

_TREND_LABELS = {
    "above_all_smas": "bullish",
    "below_all_smas": "bearish",
}


def _trend_bucket(trend_label: str) -> str:
    return _TREND_LABELS.get(trend_label, "mixed")


def _index_regime(ticker: str) -> dict:
    snap = snapshot(ticker, get_ohlcv(ticker))
    return {
        "ticker": ticker,
        "last": snap.last,
        "change_pct": snap.change_pct,
        "trend": snap.trend,
        "trend_bucket": _trend_bucket(snap.trend),
        "above_20sma": (snap.last is not None and snap.sma20 is not None and snap.last > snap.sma20),
        "above_50sma": (snap.last is not None and snap.sma50 is not None and snap.last > snap.sma50),
        "above_200sma": (snap.last is not None and snap.sma200 is not None and snap.last > snap.sma200),
        "rsi14": snap.rsi14,
    }


def _vix_regime(vix_last: float | None) -> str:
    if vix_last is None:
        return "UNKNOWN"
    if vix_last < 15:
        return "LOW"
    if vix_last < 25:
        return "NORMAL"
    if vix_last < 35:
        return "ELEVATED"
    return "HIGH"


def market_regime() -> dict:
    """Snapshot of SPY, QQQ, and VIX — the broad-market context a single ticker's
    technicals should be read against. This is index-trend + VIX-level only, not a
    breadth/advance-decline model; treat 'regime' loosely, not as a formal state
    classifier."""
    spy = _index_regime("SPY")
    qqq = _index_regime("QQQ")

    vix_snap = snapshot("^VIX", get_ohlcv("^VIX"))
    vix = {
        "last": vix_snap.last,
        "change_pct": vix_snap.change_pct,
        "regime": _vix_regime(vix_snap.last),
    }

    trend_buckets = [spy["trend_bucket"], qqq["trend_bucket"]]
    if trend_buckets.count("bullish") == 2:
        overall_trend = "bullish"
    elif trend_buckets.count("bearish") == 2:
        overall_trend = "bearish"
    else:
        overall_trend = "mixed"

    risk_label = (
        "risk-on" if overall_trend == "bullish" and vix["regime"] in ("LOW", "NORMAL")
        else "risk-off" if overall_trend == "bearish" or vix["regime"] in ("ELEVATED", "HIGH")
        else "neutral"
    )

    return {
        "spy": spy,
        "qqq": qqq,
        "vix": vix,
        "overall_trend": overall_trend,
        "risk_label": risk_label,
        "note": (
            "Index-level trend (SPY/QQQ vs. their own 20/50/200 SMAs) plus VIX level — not a "
            "breadth, advance/decline, or intraday regime model. overall_trend requires SPY AND "
            "QQQ to agree to call 'bullish'/'bearish'; otherwise 'mixed'. This is context for "
            "reading a single ticker's technicals, not a market-timing signal on its own — a "
            "stock can look strong in a weak regime (real relative strength) or weak in a strong "
            "regime (real relative weakness); see get_relative_strength for that ticker-specific "
            "comparison."
        ),
    }


def sector_regime(sector_etf: str | None) -> dict | None:
    """Same trend snapshot, applied to a ticker's own sector/industry ETF proxy
    (see relative_strength.sector_etf) — separate from the broad-market regime
    above, since a stock's own sector can diverge from SPY/QQQ."""
    if not sector_etf:
        return None
    return _index_regime(sector_etf)
