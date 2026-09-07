"""Training universe for the cross-sectional ML engine.

Deliberately small (~30 names) so a full dataset build + train completes in
minutes on a laptop, not hours. This is NOT "the S&P 500" — it's liquid,
long-history large-caps spread across sectors so the model sees more than one
industry's price behavior. Expand this list (and re-run
`python -m processing.ml.train`) once you're ready to trade off runtime for
sample size; nothing else in the pipeline assumes this exact set.
"""

from __future__ import annotations

DEFAULT_UNIVERSE = [
    # Mega-cap tech / semis
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "MU", "QCOM", "INTC",
    # Software / internet
    "GOOGL", "META", "AMZN", "CRM", "ADBE", "NFLX",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE",
    # Industrials / energy / materials
    "CAT", "XOM", "CVX", "BA",
    # Consumer
    "WMT", "HD", "COST", "MCD", "SBUX",
]

# Benchmarks used for relative-strength features — fetched alongside the
# universe, never treated as trainable targets themselves.
BENCHMARK = "SPY"
SECTOR_ETFS = ["QQQ", "SOXX", "XLF", "XLV", "XLE", "XLI", "XLY"]

# ---------------------------------------------------------------------------
# Phase 3: a broader, sector-BALANCED universe (~95 names, ~8 per sector),
# built specifically to test whether the ATR%/realized-vol finding from
# Phase 2 survives outside a mega-cap-tech/semis-heavy sample. DEFAULT_UNIVERSE
# above is untouched — Phase 1/2 results remain exactly reproducible.
#
# SURVIVORSHIP-BIAS DISCLOSURE (see also processing/ml/universe.py docstring):
# this is TODAY's list of large/liquid names per sector, not a point-in-time
# historical index constituent list. yfinance exposes no historical
# constituent-membership data, so any company that was delisted, went
# bankrupt, was acquired, or fell out of the liquid/large-cap set during
# 2022-2026 is NOT represented. This biases results toward survivors and
# should be read as "how did today's well-known names in each sector behave,"
# not "how would this strategy have performed on the true historical
# investable universe." No fix is possible without a point-in-time
# constituent-history data source; none is currently wired into this system.
# ---------------------------------------------------------------------------

EXPANDED_UNIVERSE_V2: dict[str, list[str]] = {
    "technology": ["AAPL", "MSFT", "ORCL", "CSCO", "IBM", "NOW", "INTU", "ADSK"],
    "semiconductors": ["NVDA", "AMD", "AVGO", "MU", "QCOM", "TXN", "AMAT", "LRCX"],
    "software_internet": ["GOOGL", "META", "AMZN", "CRM", "ADBE", "NFLX", "SHOP", "UBER"],
    "financials": ["JPM", "BAC", "GS", "MS", "WFC", "SCHW", "AXP", "BLK"],
    "healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT"],
    "industrials": ["CAT", "BA", "HON", "UPS", "GE", "DE", "LMT", "MMM"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "OXY", "MPC"],
    "consumer_discretionary": ["HD", "MCD", "SBUX", "NKE", "LOW", "TJX", "BKNG", "TGT"],
    "consumer_staples": ["WMT", "COST", "PG", "KO", "PEP", "PM", "MDLZ", "CL"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL"],
    "materials": ["LIN", "SHW", "FCX", "NEM", "APD", "ECL", "NUE", "DOW"],
    "communication_services": ["V", "MA", "DIS", "CMCSA", "TMUS", "CHTR", "EA", "TTWO"],
    "real_estate": ["PLD", "AMT", "EQIX", "SPG", "PSA", "O", "WELL", "DLR"],
}

EXPANDED_UNIVERSE = [t for names in EXPANDED_UNIVERSE_V2.values() for t in names]
SECTOR_MAP = {t: sector for sector, names in EXPANDED_UNIVERSE_V2.items() for t in names}
