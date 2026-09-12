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
    "^TNX": "10-Year Treasury yield",
    "^IRX": "13-Week Treasury bill yield — short-rate proxy",
    "^TYX": "30-Year Treasury yield",
    "DX-Y.NYB": "US Dollar Index",
    "CL=F": "WTI crude oil futures",
    "GC=F": "Gold futures",
}

_YIELD_TICKERS = {"^TNX", "^IRX", "^TYX"}

# yfinance has historically reported ^TNX/^IRX/^TYX in an x10 convention (46.6 =
# 4.66%), then at some point switched to plain percent (4.97 = 4.97%) with no
# announcement — this is exactly the kind of silent upstream format drift that
# produced a real bug here once already (a stored /10 assumption turned a 1.06pp
# yield-curve spread into a wrong 0.11pp). Rather than hardcode either
# convention, detect it: no US Treasury yield of any maturity has been above
# ~15% in modern history, so a raw value that large is unambiguously still in
# the x10 convention and needs dividing; anything else is already plain percent.
_X10_THRESHOLD = 15.0


def _normalize_yield_pct(raw: float | None) -> float | None:
    if raw is None:
        return None
    return round(raw / 10, 4) if raw > _X10_THRESHOLD else round(raw, 4)


def market_proxies() -> dict:
    """Real-time-ish (daily close) market pricing for rates/dollar/commodities —
    what the market is pricing right now, not a lagged government statistic."""
    out = {}
    for ticker, label in _PROXY_TICKERS.items():
        try:
            snap = snapshot(ticker, get_ohlcv(ticker))
            last = _normalize_yield_pct(snap.last) if ticker in _YIELD_TICKERS else snap.last
            out[ticker] = {"label": label, "last": last, "change_pct": snap.change_pct, "as_of": snap.as_of}
        except Exception as exc:
            out[ticker] = {"label": label, "error": f"{type(exc).__name__}: {exc}"}

    tnx = out.get("^TNX", {}).get("last")
    irx = out.get("^IRX", {}).get("last")
    yield_curve_spread = None
    yield_curve_note = None
    if tnx is not None and irx is not None:
        yield_curve_spread = round(tnx - irx, 4)  # both already normalized to plain percentage points above
        # Self-check: this is exactly the kind of derived value that must never
        # silently drift again — assert the stored spread actually equals a
        # fresh recomputation from the same normalized inputs before trusting it.
        assert abs(yield_curve_spread - round(tnx - irx, 4)) < 1e-9, (
            f"yield curve spread mismatch: stored {yield_curve_spread} vs recomputed {tnx - irx}"
        )
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
        "live intraday feed). Treasury yields are normalized to plain percentage points regardless "
        "of which convention yfinance reports them in that day (see _normalize_yield_pct)."
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
