"""Central config. Secrets come from .env; nothing here is committed with values."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# Price data (yfinance) needs no key. NEWSAPI_KEY is the only optional key
# this project actually reads — without it, news falls back to yfinance's
# own headlines (fewer articles, still works).
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Storage
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "storage" / "buddy.duckdb"))
