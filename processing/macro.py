"""Macro context: market-based proxies (yields, dollar, oil, gold) plus official
economic indicators (via data.macro) plus prediction-market odds (via
data.polymarket) — three different kinds of macro data, kept distinct rather than
blended into one score, since they update on different schedules and answer
different questions (what the bond/commodity market is pricing right now, what
government statistics actually measured last month, what bettors think will
happen).
"""

from __future__ import annotations

from processing.indicators import snapshot
from storage.cache import get_ohlcv

_PROXY_TICKERS = {
    "^TNX": "10-Year Treasury yield (x10, e.g. 46.6 = 4.66%)",
    "^IRX": "13-Week Treasury bill yield (x10) — short-rate proxy",
    "^TYX": "30-Year Treasury yield (x10)",
    "DX-Y.NYB": "US Dollar Index",
    "CL=F": "WTI crude oil futures",
    "GC=F": "Gold futures",
}


def market_proxies() -> dict:
    """Real-time-ish (daily close) market pricing for rates/dollar/commodities —
    what the market is pricing right now, not a lagged government statistic. Yield
    figures from ^TNX/^IRX/^TYX are already x10 (yfinance convention) — divide by
    10 for the actual percentage yield."""
    out = {}
    for ticker, label in _PROXY_TICKERS.items():
        try:
            snap = snapshot(ticker, get_ohlcv(ticker))
            out[ticker] = {"label": label, "last": snap.last, "change_pct": snap.change_pct, "as_of": snap.as_of}
        except Exception as exc:
            out[ticker] = {"label": label, "error": f"{type(exc).__name__}: {exc}"}

    tnx = out.get("^TNX", {}).get("last")
    irx = out.get("^IRX", {}).get("last")
    yield_curve_spread = None
    yield_curve_note = None
    if tnx is not None and irx is not None:
        yield_curve_spread = round((tnx - irx) / 10, 4)  # convert from x10 units to actual pct points
        yield_curve_note = (
            "10Y minus 13-week yield, in percentage points. Negative (inverted) has historically "
            "preceded recessions with a long and variable lag — this is a real, widely-cited "
            "indicator, but not a short-term timing signal, and past inversions have had false "
            "positives too."
        )

    out["yield_curve_10y_minus_13w_pct_points"] = yield_curve_spread
    out["yield_curve_note"] = yield_curve_note
    out["note"] = (
        "Daily-close market pricing (same data source/freshness as the rest of this system, not a "
        "live intraday feed). ^TNX/^IRX/^TYX values are yfinance's x10 convention — a value of 46.6 "
        "means a 4.66% yield."
    )
    return out


def full_macro_context(include_economic: bool = True, include_polymarket: bool = True) -> dict:
    """Everything macro this system can fetch, kept in three explicitly labeled
    groups rather than merged into one blob or one score."""
    out = {"market_proxies": market_proxies()}

    if include_economic:
        try:
            from data.macro import economic_indicators

            out["economic_indicators"] = economic_indicators()
        except Exception as exc:
            out["economic_indicators"] = {"error": f"{type(exc).__name__}: {exc}"}

    if include_polymarket:
        try:
            from data.polymarket import macro_context

            out["prediction_markets"] = macro_context()
        except Exception as exc:
            out["prediction_markets"] = {"error": f"{type(exc).__name__}: {exc}"}

    return out
