"""Coarse price-path summaries over an arbitrary window — used by processing/macro.py."""

from __future__ import annotations

from datetime import timedelta

from storage import cache


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
