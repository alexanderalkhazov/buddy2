"""Point-in-time market-regime/macro features — VIX, the Treasury yield-curve
slope, and high-yield credit spreads — joined into the sector dataset as
genuine model inputs, not just descriptive context.

Research basis: sector rotation is well-documented to behave differently in
risk-on vs. risk-off regimes (defensives lead in stress, cyclicals lead in
calm markets); VIX level, credit spreads, and the yield curve are the
standard, freely-available regime proxies for that. These do not predict
direction on their own — they condition which of the existing momentum/vol
features are informative, which is exactly what a GBM can exploit via
interaction splits.

All series here are DAILY, MARKET-based (not revised after publication like
CPI/GDP/unemployment), so joining them by date carries none of the point-in-
time fundamentals problem documented in features.py's module docstring —
today's YIELD CURVE VALUE really was known on today's date, unlike a monthly
CPI print. merge_asof(direction="backward") is still used for the join
(rather than a plain reindex) so that a sector-dataset row never picks up a
macro value from a date that hadn't happened yet, even across the small
calendar misalignments between equity trading days, Treasury settlement days,
and FRED's own publication calendar.
"""

from __future__ import annotations

import pandas as pd

from data.macro import fetch_series_history
from storage.cache import get_ohlcv

MACRO_SERIES = {
    "T10Y2Y": "yield_curve_10y2y",
    "BAMLH0A0HYM2": "credit_spread_hy",
}


def fetch_macro_history(refresh: bool = False) -> pd.DataFrame:
    """A frame, indexed by date on each series' OWN publication calendar (not
    forced onto any ticker's trading calendar), of vix_level, vix_chg_5d,
    yield_curve_10y2y, credit_spread_hy, credit_spread_hy_chg_5d. Forward-
    filled across gaps between different series' publication dates (weekends,
    holidays) so every date has the latest value that was actually known as
    of that date — never a future one. Callers align this onto their own
    frame with merge_macro_asof(), which re-applies the backward-only
    guarantee at the final join regardless of this frame's own date grid.
    refresh is accepted for interface symmetry with the rest of the ML
    pipeline but ignored — this hits FRED/yfinance directly each call, which
    is cheap (a handful of small HTTP requests, not per-ticker)."""
    vix_df = get_ohlcv("^VIX", force=refresh)
    series = {
        "vix_level": vix_df["close"],
        "vix_chg_5d": vix_df["close"].pct_change(5) * 100,
    }
    for series_id, col in MACRO_SERIES.items():
        try:
            points = fetch_series_history(series_id)
        except Exception:
            continue
        if points:
            series[col] = pd.Series({pd.Timestamp(d): v for d, v in points})

    out = pd.DataFrame(series).sort_index().ffill()
    out.index.name = "date"
    if "credit_spread_hy" in out.columns:
        out["credit_spread_hy_chg_5d"] = out["credit_spread_hy"].diff(5)
    return out


def merge_macro_asof(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time (backward-looking only) join of macro/regime columns onto
    df, matched by date. Uses merge_asof rather than a plain reindex/join so a
    row can never pick up a macro value dated after its own date, even when
    the two frames' trading calendars don't line up exactly."""
    left = df.reset_index().rename(columns={df.index.name or "index": "_date"})
    right = macro.reset_index().rename(columns={macro.index.name or "index": "_date"})
    merged = pd.merge_asof(
        left.sort_values("_date"), right.sort_values("_date"), on="_date", direction="backward"
    )
    merged = merged.set_index("_date")
    merged.index.name = df.index.name
    return merged.loc[df.index]
