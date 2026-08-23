"""Position tracking: real cost basis, live P&L, and concentration/sector exposure.

Positions are only mutated through explicit user commands (ui/chat.py), never by the
LLM — a "buy" the model didn't actually execute has no business becoming a logged
position. Reading the portfolio is a normal read-only tool, so the model can factor
existing holdings into new suggestions (concentration risk, correlated bets).
"""

from __future__ import annotations

from datetime import datetime, timezone

from data.fundamentals import fetch_fundamentals
from storage.cache import get_ohlcv
from storage.db import connect


def add_position(
    ticker: str, shares: float, cost_basis: float, note: str | None = None
) -> int:
    ticker = ticker.upper()
    with connect() as con:
        row = con.execute(
            """INSERT INTO positions
               SELECT nextval('positions_id_seq'), ?, ?, ?, ?, ?
               RETURNING id""",
            [ticker, shares, cost_basis, datetime.now(timezone.utc), note],
        ).fetchone()
    return row[0]


def remove_position(position_id: int) -> bool:
    with connect() as con:
        before = con.execute("SELECT count(*) FROM positions").fetchone()[0]
        con.execute("DELETE FROM positions WHERE id = ?", [position_id])
        after = con.execute("SELECT count(*) FROM positions").fetchone()[0]
    return after < before


def list_positions() -> list[dict]:
    with connect() as con:
        rows = con.execute(
            "SELECT id, ticker, shares, cost_basis, opened_at, note FROM positions ORDER BY ticker"
        ).fetchall()
    keys = ("id", "ticker", "shares", "cost_basis", "opened_at", "note")
    return [dict(zip(keys, r)) for r in rows]


def compact_summary() -> str:
    """One-line-per-lot text for the system prompt — no live price lookups, so it's
    cheap enough to include on every turn without slowing the model down."""
    positions = list_positions()
    if not positions:
        return ""
    lines = [f"{p['ticker']}: {p['shares']:g}sh @ ${p['cost_basis']:g}" for p in positions]
    return "; ".join(lines)


def _last_price(ticker: str) -> float | None:
    try:
        return float(get_ohlcv(ticker)["close"].iloc[-1])
    except Exception:
        return None


def portfolio_summary() -> dict:
    """Full P&L, weights, and sector concentration — uses cached prices/fundamentals,
    so repeated calls in one session are fast after the first."""
    positions = list_positions()
    if not positions:
        return {"positions": [], "total_value": 0, "total_cost": 0, "total_pl_pct": None}

    rows = []
    sector_value: dict[str, float] = {}
    total_value = 0.0
    total_cost = 0.0

    for p in positions:
        last = _last_price(p["ticker"])
        cost = p["shares"] * p["cost_basis"]
        value = p["shares"] * last if last is not None else None
        sector = None
        try:
            sector = fetch_fundamentals(p["ticker"]).get("sector")
        except Exception:
            pass

        if value is not None:
            total_value += value
            sector_value[sector or "Unknown"] = sector_value.get(sector or "Unknown", 0) + value
        total_cost += cost

        rows.append(
            {
                "id": p["id"],
                "ticker": p["ticker"],
                "shares": p["shares"],
                "cost_basis": p["cost_basis"],
                "last_price": round(last, 4) if last is not None else None,
                "market_value": round(value, 2) if value is not None else None,
                "unrealized_pl": round(value - cost, 2) if value is not None else None,
                "unrealized_pl_pct": round((value / cost - 1) * 100, 2)
                if value is not None and cost
                else None,
                "sector": sector,
                "note": p["note"],
            }
        )

    for row in rows:
        row["portfolio_weight_pct"] = (
            round(row["market_value"] / total_value * 100, 2)
            if row["market_value"] is not None and total_value
            else None
        )

    return {
        "positions": rows,
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_pl": round(total_value - total_cost, 2),
        "total_pl_pct": round((total_value / total_cost - 1) * 100, 2) if total_cost else None,
        "sector_concentration_pct": {
            k: round(v / total_value * 100, 2) for k, v in sector_value.items()
        }
        if total_value
        else {},
    }
