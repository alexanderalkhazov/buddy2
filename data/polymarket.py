"""Polymarket Gamma API — public prediction market odds, no auth required."""

from __future__ import annotations

import json

import requests

SEARCH_URL = "https://gamma-api.polymarket.com/public-search"


def _parse_prices(market: dict) -> list[float]:
    raw = market.get("outcomePrices")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    try:
        return [float(p) for p in (raw or [])]
    except (TypeError, ValueError):
        return []


def search_markets(query: str, limit: int = 5) -> list[dict]:
    """Return live markets matching a keyword, each with its current yes-odds."""
    resp = requests.get(
        SEARCH_URL,
        params={"q": query, "limit_per_type": 10, "events_status": "active"},
        timeout=15,
    )
    resp.raise_for_status()

    out: list[dict] = []
    for event in resp.json().get("events", []):
        for market in event.get("markets") or []:
            if market.get("closed"):
                continue
            prices = _parse_prices(market)
            if not prices:
                continue
            out.append(
                {
                    "market": market.get("question") or event.get("title"),
                    "odds": round(prices[0], 3),
                    "volume": round(float(market.get("volumeNum") or 0)),
                    "ends": market.get("endDate") or event.get("endDate"),
                }
            )
            if len(out) >= limit:
                return out
    return out


def macro_context(limit_per_topic: int = 2) -> list[dict]:
    """A standing set of macro markets useful as background for any ticker."""
    out: list[dict] = []
    for topic in ("fed rate cut", "recession", "inflation", "unemployment", "GDP growth"):
        try:
            out.extend(search_markets(topic, limit=limit_per_topic))
        except requests.RequestException:
            continue
    return out
