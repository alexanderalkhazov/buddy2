"""SEC EDGAR full-text search for recent filings (8-K, 10-Q, 10-K)."""

from __future__ import annotations

import requests

FTS_URL = "https://efts.sec.gov/LATEST/search-index"

# EDGAR requires a descriptive User-Agent with contact info.
HEADERS = {"User-Agent": "buddy2-trading-assistant (contact: alexander410dev@gmail.com)"}


def recent_filings(
    ticker: str, forms: tuple[str, ...] = ("8-K", "10-Q"), limit: int = 5
) -> list[dict]:
    """Return recent filings mentioning a ticker, newest-scoring first."""
    try:
        resp = requests.get(
            FTS_URL,
            params={"q": f'"{ticker}"', "forms": ",".join(forms)},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
    except (requests.RequestException, ValueError):
        return []

    out = []
    for h in hits[:limit]:
        src = h.get("_source", {})
        out.append(
            {
                "form": src.get("root_form") or src.get("file_type"),
                "filed": src.get("file_date"),
                "company": (src.get("display_names") or [None])[0],
            }
        )
    return out
