"""Price data fetching. Milestone 1 uses yfinance (no auth required)."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Return an OHLCV frame with columns open/high/low/close/volume, indexed by date."""
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"no price data returned for {ticker!r}")

    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    return df
