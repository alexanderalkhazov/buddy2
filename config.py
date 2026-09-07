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

# Storage
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "storage" / "buddy.duckdb"))
