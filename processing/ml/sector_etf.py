"""Real sector ETF price action as a parallel signal to the synthetic,
bottom-up sector index (sector_index.py).

Research basis: a real, publicly-traded ETF impounds actual institutional
positioning/flow information (creation/redemption activity, real capital
rotating in and out) that a synthetic equal-weight average of 8 member
tickers cannot fully replicate — a genuinely different information channel,
not just a reformulation of the same 8 tickers under another name.

SECTOR_ETF_PROXY maps each of this project's 13 sector buckets to a real,
liquid ETF. Most are the standard SPDR sector ETFs (a clean GICS-sector
match). Two are NOT clean matches, and this says so rather than pretending
otherwise:
  - "software_internet" (GOOGL/META/AMZN/CRM/ADBE/NFLX/SHOP/UBER) has no
    single-sector ETF that fits — QQQ (Nasdaq-100) is used as the closest
    liquid, real-money-tracked proxy, not a faithful sector match.
  - "communication_services" here is this project's OWN bucket (V, MA, DIS,
    CMCSA, TMUS, CHTR, EA, TTWO — payments + media + telecom + gaming, NOT
    GICS Communication Services, which doesn't include V/MA at all). XLC is
    used as the closest available label, not a faithful match either.
Both are flagged in SECTOR_ETF_IS_APPROXIMATE — read features derived from
those two sectors' ETF as weaker evidence than the other 11.
"""

from __future__ import annotations

import pandas as pd

from processing.ml.features import compute_features
from storage.cache import get_ohlcv

SECTOR_ETF_PROXY: dict[str, str] = {
    "technology": "XLK",
    "semiconductors": "SOXX",
    "software_internet": "QQQ",
    "financials": "XLF",
    "healthcare": "XLV",
    "industrials": "XLI",
    "energy": "XLE",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "utilities": "XLU",
    "materials": "XLB",
    "communication_services": "XLC",
    "real_estate": "XLRE",
}

SECTOR_ETF_IS_APPROXIMATE = {"software_internet", "communication_services"}


def etf_history(sector: str, refresh: bool = False) -> pd.DataFrame | None:
    """DATE-indexed frame with etf_ret_20d for this sector's proxy ETF, or
    None if the ticker can't be fetched this run (never fabricated)."""
    etf = SECTOR_ETF_PROXY.get(sector)
    if etf is None:
        return None
    try:
        df = get_ohlcv(etf, force=refresh)
        feats = compute_features(df)
    except Exception:
        return None
    out = pd.DataFrame(index=feats.index)
    out["etf_ret_20d"] = feats["ret_20d"]
    return out
