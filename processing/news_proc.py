"""News dedupe, summarization, and sentiment tagging.

Sentiment uses a lexicon by default so the pipeline runs with no API key. When an
LLM key is configured, a cheap model produces both the summary and the tag in one
batched call — still pre-processing, so raw articles never reach the reasoning model.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import config

POSITIVE = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "record", "upgrade",
    "upgrades", "growth", "profit", "gains", "gain", "outperform", "strong", "raise",
    "raises", "bullish", "jumps", "soars", "tops", "wins", "approval",
}
NEGATIVE = {
    "miss", "misses", "plunge", "plunges", "fall", "falls", "drop", "drops",
    "downgrade", "downgrades", "loss", "losses", "weak", "cut", "cuts", "bearish",
    "slump", "lawsuit", "probe", "recall", "warns", "warning", "slides", "sinks",
}

_WORD = re.compile(r"[a-z']+")


def _lexicon_sentiment(text: str) -> str:
    words = set(_WORD.findall(text.lower()))
    score = len(words & POSITIVE) - len(words & NEGATIVE)
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def _age_label(published_at) -> str:
    if not published_at:
        return "undated"
    try:
        ts = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
    except ValueError:
        return "undated"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    if hours < 1:
        return "just now"
    if hours < 24:
        return f"{int(hours)}h ago"
    return f"{int(hours // 24)}d ago"


SUMMARIZER_SYSTEM = (
    "Each item's 'body' may be a short snippet or a full article. Extract the single "
    "most concrete, decision-relevant claim — a number, a specific event, a named "
    "cause — not a restatement of the headline. If the body is a full article and it "
    "contains a number or figure central to the story (a price target, a percentage, "
    "a dollar amount), include it. One clause, max 25 words. Then tag sentiment for "
    "the subject company's stock. Be factual, no advice. Reply with JSON: "
    '{"items": [{"i": <index>, "summary": "...", "sentiment": '
    '"positive"|"negative"|"neutral"}]}'
)

# Full articles run ~2000-4000 chars; this is a token-budget cap for the summarizer
# call, not a claim about how much of the article matters — the lead usually carries
# the key facts anyway. Kept modest since the local model (Ollama) is the only
# backend for now and is slow on long inputs: a batch of 8 articles at 1200 chars
# each measured at 354s wall-clock on qwen3:4b. Trimmed to keep the first bundle
# build for a ticker from feeling stalled; full re-tuning can wait until a faster
# backend is back in the provider rotation.
_SUMMARIZER_BODY_CHARS = 800


def _llm_summarize(articles: list[dict]) -> list[dict] | None:
    """Batch summary + sentiment via a cheap model. Returns None if unavailable."""
    payload = [
        {"i": i, "title": a["title"], "body": (a.get("body") or "")[:_SUMMARIZER_BODY_CHARS]}
        for i, a in enumerate(articles)
    ]
    try:
        cfg = config.provider_config()
        if cfg["name"] == "anthropic":
            text = _summarize_anthropic(payload) if cfg["key"] else None
        elif cfg["key"] or cfg["name"] == "ollama":
            text = _summarize_openai_compat(payload, cfg)
        else:
            return None  # no key configured — caller falls back to the lexicon
    except Exception:
        return None

    if not text:
        return None
    try:
        return json.loads(text)["items"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _summarize_openai_compat(payload: list[dict], cfg: dict) -> str | None:
    from openai import OpenAI

    client = OpenAI(api_key=cfg["key"] or "not-needed", base_url=cfg["base_url"])
    response = client.chat.completions.create(
        model=cfg["summarizer"],
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )
    return response.choices[0].message.content


def _summarize_anthropic(payload: list[dict]) -> str | None:
    import anthropic

    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer"},
                        "summary": {"type": "string"},
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral"],
                        },
                    },
                    "required": ["i", "summary", "sentiment"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.ANTHROPIC_SUMMARIZER_MODEL,
        max_tokens=4000,
        system=SUMMARIZER_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    return next((b.text for b in response.content if b.type == "text"), None)


def process(articles: list[dict], limit: int = 5) -> list[dict]:
    """Turn raw articles into compact, sentiment-tagged summaries for the bundle."""
    articles = articles[:limit]
    if not articles:
        return []

    enriched = _llm_summarize(articles) or []
    by_index = {item["i"]: item for item in enriched if isinstance(item, dict) and "i" in item}

    out = []
    for i, a in enumerate(articles):
        llm = by_index.get(i)
        summary = llm["summary"] if llm else a["title"]
        sentiment = llm["sentiment"] if llm else _lexicon_sentiment(
            f"{a['title']} {a.get('body', '')}"
        )
        tag = "full article" if a.get("full_text") else "headline only"
        out.append(
            {
                "title": a["title"],
                "url": a["url"],
                "source": a.get("source"),
                "published_at": a.get("published_at"),
                "summary": (
                    f"{summary} ({a.get('source') or 'unknown'}, "
                    f"{_age_label(a.get('published_at'))}, {tag})"
                ),
                "sentiment": sentiment,
                "full_text": bool(a.get("full_text")),
            }
        )
    return out


def aggregate_sentiment(items: list[dict]) -> str:
    if not items:
        return "unknown"
    score = sum((i["sentiment"] == "positive") - (i["sentiment"] == "negative") for i in items)
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"
