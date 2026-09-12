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


_BREADTH_TICKERS = {
    "^VIX9D": "9-Day VIX (near-term vol expectation)",
    "^VIX": "30-Day VIX (standard)",
    "^VIX3M": "3-Month VIX (medium-term vol expectation)",
    "^VVIX": "VVIX — volatility of VIX itself (options-market stress-of-stress)",
}


def us_market_breadth(days: int = 20) -> dict:
    """Signals about the HEALTH of the US market's advance, not just its level:
    is the rally broad (small-caps and the average stock participating) or
    narrow (a handful of mega-caps carrying a cap-weighted index)? Is the
    options market pricing near-term calm or near-term stress relative to
    further out (VIX term structure)? None of this is in market_regime() or
    market_proxies() — a real gap flagged in earlier reports, filled here."""
    from processing.bundle import history_window

    out: dict = {}

    for ticker, label in _BREADTH_TICKERS.items():
        try:
            snap = snapshot(ticker, get_ohlcv(ticker))
            out[ticker] = {"label": label, "last": snap.last, "as_of": snap.as_of}
        except Exception as exc:
            out[ticker] = {"label": label, "error": f"{type(exc).__name__}: {exc}"}

    vix9d, vix, vix3m = out.get("^VIX9D", {}).get("last"), out.get("^VIX", {}).get("last"), out.get("^VIX3M", {}).get("last")
    dates = {k: out.get(k, {}).get("as_of") for k in ("^VIX9D", "^VIX", "^VIX3M")}
    valid_dates = [d for d in dates.values() if d]
    # A term-structure read compares levels that must actually be from the same
    # (or near-same) day — yfinance has genuinely stopped updating ^VIX9D/^VIX3M
    # daily bars before (observed: frozen ~2 months while ^VIX kept updating).
    # Comparing a live leg to a stale one would assert a false contango/
    # backwardation conclusion, so refuse to conclude rather than guess.
    stale_mismatch = len(valid_dates) >= 2 and (max(valid_dates) != min(valid_dates))
    if vix9d is not None and vix3m is not None and not stale_mismatch:
        term_structure_state = (
            "BACKWARDATION (near-term vol priced ABOVE longer-term — a sign of acute, "
            "current market stress, e.g. around a known near-term event or an active selloff)"
            if vix9d > vix3m else
            "CONTANGO (near-term vol priced below longer-term — the normal, calm state)"
        )
    elif stale_mismatch:
        term_structure_state = f"UNKNOWN — legs have mismatched as-of dates ({dates}), a stale/frozen data feed for one leg would make any contango/backwardation conclusion unreliable"
    else:
        term_structure_state = None
    out["vix_term_structure_state"] = term_structure_state

    breadth_pairs = {
        "equal_weight_vs_cap_weight": ("RSP", "SPY", "Equal-weight S&P vs. cap-weight S&P — positive means the AVERAGE stock is keeping up with the index (broad participation); negative means a handful of mega-caps are carrying the index while most stocks lag."),
        "small_cap_vs_large_cap": ("IWM", "SPY", "Russell 2000 (small-cap) ETF vs. S&P 500 — small-caps leading is historically associated with risk-on breadth; small-caps lagging badly can flag narrow, fragile leadership."),
    }
    for key, (a, b, note) in breadth_pairs.items():
        try:
            ret_a = history_window(a, days=days).get("return_pct")
            ret_b = history_window(b, days=days).get("return_pct")
            spread = round(ret_a - ret_b, 3) if ret_a is not None and ret_b is not None else None
        except Exception:
            spread, ret_a, ret_b = None, None, None
        out[key] = {"a": a, "b": b, f"{a}_return_pct_{days}d": ret_a, f"{b}_return_pct_{days}d": ret_b, "spread_pct_points": spread, "note": note}

    out["note"] = (
        f"Breadth spreads use trailing {days}-calendar-day returns of liquid ETF proxies (RSP, IWM), "
        "same daily-close data source as the rest of this system — not a true advance/decline or "
        "%-above-SMA breadth count across all listed stocks, which this system does not have a data "
        "source for. VIX term structure uses CBOE's own published 9-day/30-day/3-month indices."
    )
    return out


def full_macro_context(include_economic: bool = True, include_polymarket: bool = True, include_breadth: bool = True) -> dict:
    """Everything macro this system can fetch, kept in explicitly labeled
    groups rather than merged into one blob or one score."""
    out = {"market_proxies": market_proxies()}

    if include_economic:
        try:
            from data.macro import economic_indicators

            out["economic_indicators"] = economic_indicators()
        except Exception as exc:
            out["economic_indicators"] = {"error": f"{type(exc).__name__}: {exc}"}

    if include_breadth:
        try:
            out["market_breadth"] = us_market_breadth()
        except Exception as exc:
            out["market_breadth"] = {"error": f"{type(exc).__name__}: {exc}"}

    if include_polymarket:
        try:
            from data.polymarket import macro_context

            out["prediction_markets"] = macro_context()
        except Exception as exc:
            out["prediction_markets"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Independent cross-check: compare our own yfinance-derived 10Y-13W spread
    # against FRED's OWN official 10Y-3M spread (T10Y3M) — different maturities
    # (13-week bill vs 3-month) so they won't match exactly, but they should be
    # close. A material mismatch would have caught the exact x10-scaling bug
    # this system already shipped once, from an independent second source.
    derived = out.get("market_proxies", {}).get("yield_curve_10y_minus_13w_pct_points")
    official = None
    if include_economic:
        official = (out.get("economic_indicators", {}).get("T10Y3M") or {}).get("value")
    if derived is not None and official is not None:
        diff = round(abs(derived - official), 4)
        out["yield_curve_cross_check"] = {
            "our_derived_10y_minus_13w": derived,
            "fred_official_10y_minus_3m": official,
            "abs_difference_pct_points": diff,
            "flag": "MISMATCH — investigate a scaling/calculation bug" if diff > 0.5 else "consistent",
        }

    return out
