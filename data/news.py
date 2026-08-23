"""News fetching: NewsAPI (or yfinance fallback) for discovery, trafilatura for the
actual article text — the model reasons off full articles, not one-line snippets.
"""

from __future__ import annotations

import difflib
import re
from datetime import datetime, timedelta, timezone

import requests
import trafilatura
import yfinance as yf

import config

NEWSAPI_URL = "https://newsapi.org/v2/everything"
FULL_TEXT_LIMIT = 3  # most-recent articles to fetch full text for; the rest keep their short snippet
FULL_TEXT_TIMEOUT = 6  # seconds; a slow site shouldn't stall the whole chat turn
FULL_TEXT_MAX_CHARS = 4000
TITLE_DEDUP_THRESHOLD = 0.85  # near-duplicate titles (syndicated reposts) collapse to one


def _company_name(ticker: str) -> str | None:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None
    return info.get("shortName") or info.get("longName")


def _from_newsapi(ticker: str, company_name: str | None, lookback_hours: int) -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    # A bare ticker like "MU" is a two-letter word — searching it alone pulls in noise.
    # OR-ing in the company's brand short name (not the legal entity name — "Micron
    # Technology, Inc." is a weak search term; "Micron" is what headlines use)
    # sharply improves relevance without needing a paid entity-search API.
    short_name = _short_company_name(company_name) if company_name else None
    query = f'"{ticker}"' + (f' OR "{short_name}"' if short_name else "")
    resp = requests.get(
        NEWSAPI_URL,
        params={
            "q": query,
            "from": since.strftime("%Y-%m-%dT%H:%M:%S"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 25,
            "apiKey": config.NEWSAPI_KEY,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return [
        {
            "title": a["title"],
            "url": a["url"],
            "source": (a.get("source") or {}).get("name"),
            "published_at": a.get("publishedAt"),
            "body": a.get("description") or "",
        }
        for a in resp.json().get("articles", [])
        if a.get("title") and a.get("url")
    ]


def _from_yfinance(ticker: str) -> list[dict]:
    out = []
    for item in yf.Ticker(ticker).news or []:
        content = item.get("content", item)
        url = (content.get("canonicalUrl") or {}).get("url") or content.get("link")
        if not content.get("title") or not url:
            continue
        out.append(
            {
                "title": content["title"],
                "url": url,
                "source": (content.get("provider") or {}).get("displayName"),
                "published_at": content.get("pubDate"),
                "body": content.get("summary") or "",
            }
        )
    return out


def _dedupe_by_url(articles: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for a in articles:
        if a["url"] in seen:
            continue
        seen.add(a["url"])
        out.append(a)
    return out


def _dedupe_near_duplicate_titles(articles: list[dict]) -> list[dict]:
    """Collapse syndicated reposts of the same story (same headline, different
    outlet/URL) so one story doesn't get triple-counted in the sentiment aggregate."""
    kept: list[dict] = []
    kept_titles: list[str] = []
    for a in articles:
        title = a["title"].lower()
        if any(
            difflib.SequenceMatcher(None, title, kt).ratio() >= TITLE_DEDUP_THRESHOLD
            for kt in kept_titles
        ):
            continue
        kept.append(a)
        kept_titles.append(title)
    return kept


_LEGAL_SUFFIX_RE = re.compile(
    r",?\s+(inc\.?|corp(oration)?\.?|co(mpany)?\.?|ltd\.?|plc|llc|group|holdings?|"
    r"n\.v\.|s\.a\.|ag)\s*$",
    re.IGNORECASE,
)


def _short_company_name(company_name: str) -> str:
    """'Micron Technology, Inc.' -> 'Micron Technology' -> primary word 'Micron'.
    Headlines almost always use the brand short name, never the legal entity name.

    Known limitation: for names where the first word isn't the recognizable brand
    (e.g. "Advanced Micro Devices" -> "Advanced", not "AMD"), this under-matches. Not
    fixed here — it's a secondary OR-condition; the ticker check (primary, exact,
    word-boundary) already covers those cases correctly on its own."""
    cleaned = company_name
    while (stripped := _LEGAL_SUFFIX_RE.sub("", cleaned)) != cleaned:
        cleaned = stripped
    return cleaned.split()[0] if cleaned.split() else cleaned


def _is_relevant(article: dict, ticker: str, company_name: str | None) -> bool:
    """Drop articles where neither the ticker nor the company name actually appears —
    catches false-positive matches from ambiguous short tickers. Word-boundary
    matching throughout: a substring check on "MU" would false-positive inside
    "immune" or "community"."""
    haystack = f"{article['title']} {article.get('body', '')}"
    if re.search(rf"\b{re.escape(ticker)}\b", haystack, re.IGNORECASE):
        return True
    if not company_name:
        return True  # can't check the name; don't over-filter on ticker alone
    short_name = _short_company_name(company_name)
    return bool(re.search(rf"\b{re.escape(short_name)}\b", haystack, re.IGNORECASE))


def fetch_article_text(url: str) -> str | None:
    """Best-effort full-article extraction. Returns None on any failure — callers
    fall back to the short snippet, never crash the news pipeline over one bad site."""
    try:
        resp = requests.get(
            url,
            timeout=FULL_TEXT_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; buddy2-research-bot/1.0)"},
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    text = trafilatura.extract(resp.text, favor_recall=True)
    if not text:
        return None
    return text[:FULL_TEXT_MAX_CHARS]


def _enrich_with_full_text(articles: list[dict], limit: int) -> list[dict]:
    for article in articles[:limit]:
        full = fetch_article_text(article["url"])
        if full:
            article["body"] = full
            article["full_text"] = True
        else:
            article["full_text"] = False
    for article in articles[limit:]:
        article.setdefault("full_text", False)
    return articles


def fetch_news(
    ticker: str,
    lookback_hours: int = 72,
    full_text: bool = True,
    full_text_limit: int = FULL_TEXT_LIMIT,
) -> list[dict]:
    """Return articles for a ticker, newest first, deduped and relevance-filtered.
    When full_text is True (default), the most recent `full_text_limit` articles get
    their actual body text fetched — the rest keep the short discovery snippet."""
    company_name = _company_name(ticker)
    articles = (
        _from_newsapi(ticker, company_name, lookback_hours)
        if config.NEWSAPI_KEY
        else _from_yfinance(ticker)
    )

    articles = _dedupe_by_url(articles)
    articles = _dedupe_near_duplicate_titles(articles)
    articles = [a for a in articles if _is_relevant(a, ticker, company_name)]

    if full_text and articles:
        articles = _enrich_with_full_text(articles, full_text_limit)

    return articles
