"""Caching layer: price history and news in DuckDB, refreshed on a freshness window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from data.prices import fetch_ohlcv
from storage.db import connect

PRICE_TTL = timedelta(hours=12)
NEWS_TTL = timedelta(hours=6)


def _last_fetch(con, ticker: str, kind: str) -> datetime | None:
    row = con.execute(
        "SELECT fetched_at FROM fetch_log WHERE ticker = ? AND kind = ?", [ticker, kind]
    ).fetchone()
    return row[0] if row else None


def _mark_fetch(con, ticker: str, kind: str) -> None:
    con.execute(
        """INSERT INTO fetch_log VALUES (?, ?, ?)
           ON CONFLICT (ticker, kind) DO UPDATE SET fetched_at = excluded.fetched_at""",
        [ticker, kind, datetime.now(timezone.utc)],
    )


def is_stale(ticker: str, kind: str, ttl: timedelta) -> bool:
    with connect() as con:
        last = _last_fetch(con, ticker.upper(), kind)
    if last is None:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - last > ttl


def _latest_close_incomplete(con, ticker: str) -> bool:
    """True if the most recently cached bar has a NULL close — yfinance sometimes
    returns open/high/low/volume for a bar before the close is finalized (a
    partial-day or provider-lag artifact). A TTL-based staleness check alone won't
    catch this: the fetch itself succeeded and is timestamped fresh, but the data is
    incomplete and will silently stay that way until the TTL expires on its own."""
    row = con.execute(
        "SELECT close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 1", [ticker]
    ).fetchone()
    return row is not None and row[0] is None


def get_ohlcv(ticker: str, period: str = "5y", force: bool = False) -> pd.DataFrame:
    """Return OHLCV from DuckDB, refetching from the provider when the cache is stale
    or the most recent cached bar is incomplete."""
    ticker = ticker.upper()

    with connect() as con:
        incomplete = _latest_close_incomplete(con, ticker)

    if force or incomplete or is_stale(ticker, "prices", PRICE_TTL):
        df = fetch_ohlcv(ticker, period=period)
        rows = df.reset_index()
        rows.insert(0, "ticker", ticker)
        rows["date"] = pd.to_datetime(rows["date"]).dt.tz_localize(None)
        with connect() as con:
            con.execute("DELETE FROM prices WHERE ticker = ?", [ticker])
            con.register("incoming", rows[["ticker", "date", "open", "high", "low", "close", "volume"]])
            con.execute("INSERT INTO prices SELECT * FROM incoming")
            con.unregister("incoming")
            _mark_fetch(con, ticker, "prices")

    with connect() as con:
        cached = con.execute(
            "SELECT date, open, high, low, close, volume FROM prices WHERE ticker = ? ORDER BY date",
            [ticker],
        ).df()

    if cached.empty:
        raise ValueError(f"no cached price data for {ticker}")

    return cached.set_index("date")


def store_news(ticker: str, articles: list[dict]) -> None:
    ticker = ticker.upper()
    with connect() as con:
        for a in articles:
            con.execute(
                """INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (ticker, url) DO UPDATE SET
                       summary = excluded.summary, sentiment = excluded.sentiment""",
                [
                    ticker,
                    a["url"],
                    a["title"],
                    a.get("source"),
                    a.get("published_at"),
                    a.get("summary"),
                    a.get("sentiment"),
                ],
            )
        _mark_fetch(con, ticker, "news")


def load_news(ticker: str, limit: int = 8) -> list[dict]:
    with connect() as con:
        rows = con.execute(
            """SELECT title, source, published_at, summary, sentiment, url
               FROM news WHERE ticker = ? ORDER BY published_at DESC LIMIT ?""",
            [ticker.upper(), limit],
        ).fetchall()
    keys = ("title", "source", "published_at", "summary", "sentiment", "url")
    return [dict(zip(keys, r)) for r in rows]
