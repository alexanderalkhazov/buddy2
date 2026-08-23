"""Relative strength vs. the broad market and sector — is this stock's move
happening in isolation, or moving with (or against) its group?

Reuses processing.bundle.history_window, so this needs no new data source: SPY and
a sector ETF are just additional tickers fetched through the same cached OHLCV path
as any other ticker. The sector->ETF map is a coarse GICS-level lookup, not a
precise index membership check — good enough to answer "is this a semis rally or
an AMD-specific rally," not to reproduce SOXX's exact holdings.
"""

from __future__ import annotations

from processing import bundle

BENCHMARK = "SPY"

_SECTOR_ETF = {
    "technology": "XLK",
    "healthcare": "XLV",
    "financial services": "XLF",
    "consumer cyclical": "XLY",
    "consumer defensive": "XLP",
    "energy": "XLE",
    "industrials": "XLI",
    "basic materials": "XLB",
    "real estate": "XLRE",
    "utilities": "XLU",
    "communication services": "XLC",
}

# Finer-grained than sector alone for a few industries where the broad sector ETF
# is a poor proxy (e.g. semis vs. "Technology" generally, which also holds AAPL/MSFT).
_INDUSTRY_ETF = {
    "semiconductor": "SOXX",
    "semiconductors": "SOXX",
    "biotechnology": "XBI",
    "regional banks": "KRE",
    "oil & gas": "XOP",
    "gold": "GDX",
    "homebuilding": "XHB",
}


def sector_etf(sector: str | None, industry: str | None) -> str | None:
    industry_l = (industry or "").lower()
    for keyword, etf in _INDUSTRY_ETF.items():
        if keyword in industry_l:
            return etf
    return _SECTOR_ETF.get((sector or "").lower())


def relative_strength(ticker: str, sector: str | None, industry: str | None, days: int = 90) -> dict:
    """Ticker return vs. SPY and (if resolvable) its sector/industry ETF, over the
    same window. Positive relative_vs_* means the ticker outperformed that
    benchmark over the window — a real, computed comparison, not a forecast of
    continued outperformance."""
    ticker_window = bundle.history_window(ticker, days=days)
    if "error" in ticker_window:
        return {"error": ticker_window["error"]}

    spy_window = bundle.history_window(BENCHMARK, days=days)
    etf = sector_etf(sector, industry)
    etf_window = bundle.history_window(etf, days=days) if etf else None

    ticker_return = ticker_window.get("return_pct")
    spy_return = spy_window.get("return_pct")
    etf_return = etf_window.get("return_pct") if etf_window and "error" not in etf_window else None

    out = {
        "ticker": ticker.upper(),
        "days": days,
        "ticker_return_pct": ticker_return,
        "benchmark": BENCHMARK,
        "benchmark_return_pct": spy_return,
        "relative_vs_benchmark_pct": (
            round(ticker_return - spy_return, 4) if ticker_return is not None and spy_return is not None else None
        ),
        "sector_etf": etf,
        "sector_etf_return_pct": etf_return,
        "relative_vs_sector_pct": (
            round(ticker_return - etf_return, 4) if ticker_return is not None and etf_return is not None else None
        ),
    }

    rel_bench = out["relative_vs_benchmark_pct"]
    rel_sector = out["relative_vs_sector_pct"]
    if rel_bench is not None:
        if rel_sector is not None:
            if rel_bench > 0 and rel_sector > 0:
                label = "OUTPERFORMING both the broad market and its sector/industry"
            elif rel_bench < 0 and rel_sector < 0:
                label = "UNDERPERFORMING both the broad market and its sector/industry"
            elif rel_bench > 0:
                label = "outperforming the broad market but not its own sector/industry — a sector-wide move, not stock-specific strength"
            else:
                label = "outperforming its sector/industry but lagging the broad market"
        else:
            label = "OUTPERFORMING the broad market" if rel_bench > 0 else "UNDERPERFORMING the broad market"
        out["relative_strength_label"] = label

    out["note"] = (
        f"Return comparison over the trailing {days} calendar days, computed from the same cached "
        "daily-close data as the rest of this system — not a forecast of continued relative performance. "
        "sector_etf is a coarse GICS-sector/industry-keyword lookup (e.g. SOXX for semiconductors), not an "
        "exact index-membership match."
    )
    return out
