"""A persisted ticker watchlist — user-managed, drives the default set for dash/watch."""

from __future__ import annotations

from datetime import datetime, timezone

from storage.db import connect


def add(ticker: str) -> None:
    ticker = ticker.upper()
    with connect() as con:
        con.execute(
            "INSERT INTO watchlist VALUES (?, ?) ON CONFLICT (ticker) DO NOTHING",
            [ticker, datetime.now(timezone.utc)],
        )


def remove(ticker: str) -> bool:
    ticker = ticker.upper()
    with connect() as con:
        before = con.execute("SELECT count(*) FROM watchlist").fetchone()[0]
        con.execute("DELETE FROM watchlist WHERE ticker = ?", [ticker])
        after = con.execute("SELECT count(*) FROM watchlist").fetchone()[0]
    return after < before


def list_all() -> list[str]:
    with connect() as con:
        rows = con.execute("SELECT ticker FROM watchlist ORDER BY added_at").fetchall()
    return [r[0] for r in rows]
