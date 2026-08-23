"""Side-by-side peer comparison: fundamentals + a quick technical read per ticker.

Exists so relative-valuation questions ("how does MU compare to its peers?") have a
real data source instead of the model asserting a comparison it can't support.
"""

from __future__ import annotations

from data.fundamentals import fetch_fundamentals
from processing.indicators import snapshot
from storage.cache import get_ohlcv


def compare_tickers(tickers: list[str]) -> list[dict]:
    out = []
    for ticker in tickers:
        ticker = ticker.upper()
        row: dict = {"ticker": ticker}
        try:
            snap = snapshot(ticker, get_ohlcv(ticker))
            row["last"] = snap.last
            row["rsi14"] = snap.rsi14
            row["trend"] = snap.trend
        except Exception as exc:
            row["price_error"] = str(exc)[:120]

        try:
            row.update(
                {
                    k: v
                    for k, v in fetch_fundamentals(ticker).items()
                    if k
                    in (
                        "asset_class",
                        "sector",
                        "industry",
                        "market_cap",
                        "forward_pe",
                        "profit_margin",
                        "revenue_growth",
                        "next_earnings_date",
                        "circulating_supply",
                        "fund_category",
                    )
                }
            )
        except Exception as exc:
            row["fundamentals_error"] = str(exc)[:120]

        out.append(row)
    return out
