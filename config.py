"""Central config. Secrets come from .env; nothing here is committed with values."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "AAPL")

# Data providers (price data via yfinance needs no key)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# ---------------------------------------------------------------------------
# LLM layer
#
# Every provider below (except Anthropic) speaks the OpenAI chat-completions
# format, so they share one backend and differ only by base URL, key, and model.
# Switch with LLM_PROVIDER in .env.
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

PROVIDERS: dict[str, dict] = {
    # Free tier, no card. Fast. NOTE: Groq (groq.com) is not Grok (x.ai).
    "groq": {
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "summarizer": os.getenv("GROQ_SUMMARIZER_MODEL", "openai/gpt-oss-20b"),
        "signup": "https://console.groq.com/keys",
    },
    # Free tier with daily limits, no card.
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        "summarizer": os.getenv("GEMINI_SUMMARIZER_MODEL", "gemini-flash-lite-latest"),
        "signup": "https://aistudio.google.com/apikey",
    },
    # Aggregator; ":free" models cost nothing but vary in tool-calling quality.
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        "summarizer": os.getenv(
            "OPENROUTER_SUMMARIZER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
        ),
        "signup": "https://openrouter.ai/keys",
    },
    # Local, no key, no limits. Weakest at tool calling.
    "ollama": {
        "key_env": "OLLAMA_API_KEY",  # unused; kept so the shape stays uniform
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "model": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
        "summarizer": os.getenv("OLLAMA_SUMMARIZER_MODEL", "qwen3:4b"),
        "signup": "https://ollama.com/download",
    },
    # Paid — no free tier on the API.
    "grok": {
        "key_env": "XAI_API_KEY",
        "base_url": os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        "model": os.getenv("GROK_MODEL", "grok-4-latest"),
        "summarizer": os.getenv("GROK_SUMMARIZER_MODEL", "grok-3-mini"),
        "signup": "https://console.x.ai",
    },
}


def provider_config(name: str | None = None) -> dict:
    """Resolve a provider preset plus its API key from the environment."""
    name = (name or LLM_PROVIDER).lower()
    if name == "anthropic":
        return {
            "name": "anthropic",
            "key": ANTHROPIC_API_KEY,
            "key_env": "ANTHROPIC_API_KEY",
            "model": ANTHROPIC_MODEL,
            "summarizer": ANTHROPIC_SUMMARIZER_MODEL,
            "signup": "https://console.anthropic.com",
        }
    preset = PROVIDERS.get(name)
    if preset is None:
        known = ", ".join([*PROVIDERS, "anthropic"])
        raise RuntimeError(f"Unknown LLM_PROVIDER {name!r}. Options: {known}")
    return {"name": name, "key": os.getenv(preset["key_env"]), **preset}


# Auto-fallback order when the active provider hits a rate limit or bad auth mid-turn.
# grok is deliberately excluded — that account has no credits, so it fails every
# time and just wastes a hop before reaching a provider that actually works.
FALLBACK_ORDER = ["groq", "gemini", "openrouter", "ollama"]


def fallback_chain(primary: str) -> list[dict]:
    """Other OpenAI-compatible providers with a usable key, in fallback order."""
    return [
        provider_config(name)
        for name in FALLBACK_ORDER
        if name != primary.lower() and (os.getenv(PROVIDERS[name]["key_env"]) or name == "ollama")
    ]


# Anthropic uses its own SDK (native tool use + adaptive thinking)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
ANTHROPIC_EFFORT = os.getenv("ANTHROPIC_EFFORT", "high")  # low|medium|high|xhigh|max
ANTHROPIC_SUMMARIZER_MODEL = os.getenv("ANTHROPIC_SUMMARIZER_MODEL", "claude-haiku-4-5")

# Storage
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "storage" / "buddy.duckdb"))
