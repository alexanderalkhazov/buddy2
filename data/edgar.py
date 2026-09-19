"""SEC EDGAR — recent filings actually filed BY a given ticker's company.

Uses the real per-company submissions API (data.sec.gov/submissions/), not
EDGAR's full-text SEARCH endpoint — full-text search is relevance-ranked
keyword matching across every filer since ~2001, not "this company's own
recent filings," and testing it directly turned up filings from unrelated
companies (a keyword false-positive) dated years in the past despite the
function claiming "recent" — the submissions API used here returns this
company's ACTUAL filing history, correctly attributed by CIK, genuinely
sorted newest-first.
"""

from __future__ import annotations

import requests

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# EDGAR requires a descriptive User-Agent with contact info on every request.
HEADERS = {"User-Agent": "buddy2-trading-assistant (contact: alexander410dev@gmail.com)"}

# Cached for the process lifetime — this is a ~10k-entry, infrequently-
# changing reference file, not per-ticker data; refetching it on every call
# would be wasteful and SEC explicitly asks callers to be considerate of
# their infrastructure.
_TICKER_TO_CIK: dict[str, int] | None = None


def _ticker_to_cik(ticker: str) -> int | None:
    global _TICKER_TO_CIK
    if _TICKER_TO_CIK is None:
        try:
            resp = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            _TICKER_TO_CIK = {
                row["ticker"].upper(): int(row["cik_str"]) for row in resp.json().values()
            }
        except (requests.RequestException, ValueError, KeyError):
            _TICKER_TO_CIK = {}
    return _TICKER_TO_CIK.get(ticker.upper())


def recent_filings(
    ticker: str, forms: tuple[str, ...] = ("8-K", "10-Q", "10-K"), limit: int = 5
) -> list[dict]:
    """This ticker's own recent SEC filings, genuinely newest-first, matched
    by CIK (never a keyword/full-text guess). Returns [] on any failure —
    a missing/unmapped ticker (ETFs, indices, and most non-US-listed names
    have no CIK) is a normal case, not an error to surface."""
    cik = _ticker_to_cik(ticker)
    if cik is None:
        return []
    try:
        resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=HEADERS, timeout=15)
        resp.raise_for_status()
        recent = resp.json().get("filings", {}).get("recent", {})
    except (requests.RequestException, ValueError):
        return []

    forms_list = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    out = []
    for form, filed, accession, doc in zip(forms_list, dates, accessions, docs):
        if form not in forms:
            continue
        # Direct link to the actual filing document, not a generic search
        # page — verified against a real filing (200, not a redirect/404)
        # before shipping this URL pattern.
        accession_nodashes = accession.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodashes}/{doc}" if doc else None
        out.append({"form": form, "filed": filed, "url": url})
        if len(out) >= limit:
            break
    return out
