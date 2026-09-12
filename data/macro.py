"""Official US macroeconomic indicators via FRED's public CSV endpoint.

No API key required for this endpoint — it's the same CSV FRED's own chart-download
button produces. This is a different kind of data than everything else this app
fetches: real economic indicators (CPI, Fed funds rate, unemployment, GDP), not price
series. They update monthly/quarterly, not daily, and are frequently revised after
initial release — both facts are surfaced in the returned dict, not left implicit.
"""

from __future__ import annotations

import csv
import io

import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# series_id -> (label, unit, typical release cadence)
SERIES = {
    "CPIAUCSL": ("CPI (all items, index)", "index", "monthly"),
    "FEDFUNDS": ("Federal funds effective rate", "%", "monthly"),
    "UNRATE": ("Unemployment rate", "%", "monthly"),
    "A191RL1Q225SBEA": ("Real GDP growth (annualized, quarterly)", "%", "quarterly"),
    # Daily-updated official series (not lagged like the four above) — these are
    # FRED's OWN pre-computed spreads, not derived from yfinance ETF prices, so
    # they double as an independent cross-check on processing/macro.py's own
    # yield-curve calculation (the exact kind of value that produced a real bug
    # once already when computed by hand from a mis-scaled input).
    "T10Y2Y": ("10Y minus 2Y Treasury yield spread (official)", "pct points", "daily"),
    "T10Y3M": ("10Y minus 3-month Treasury yield spread (official)", "pct points", "daily"),
    "DGS2": ("2-Year Treasury yield", "%", "daily"),
    "BAMLH0A0HYM2": ("High-yield corporate credit spread (ICE BofA OAS)", "pct points", "daily"),
    "BAMLC0A0CM": ("Investment-grade corporate credit spread (ICE BofA OAS)", "pct points", "daily"),
}


def _fetch_series(series_id: str) -> list[tuple[str, float]]:
    resp = requests.get(FRED_CSV_URL, params={"id": series_id}, timeout=15)
    resp.raise_for_status()
    reader = csv.reader(io.StringIO(resp.text))
    next(reader, None)  # header row: DATE,<series_id>
    out = []
    for row in reader:
        if len(row) != 2 or row[1] in (".", ""):
            continue
        try:
            out.append((row[0], float(row[1])))
        except ValueError:
            continue
    return out


def economic_indicators() -> dict:
    """Latest value + prior value for CPI, Fed funds rate, unemployment, and GDP
    growth — each an official government/Fed-published series via FRED, not a
    market-price proxy. Revised/lagged by nature: a monthly release published now
    describes last month (or earlier), and initial prints are often revised in
    later releases — never call this "current," always cite the as_of date."""
    out = {}
    for series_id, (label, unit, cadence) in SERIES.items():
        try:
            points = _fetch_series(series_id)
        except requests.RequestException as exc:
            out[series_id] = {"label": label, "error": f"{type(exc).__name__}: {exc}"}
            continue
        if not points:
            out[series_id] = {"label": label, "error": "no data returned"}
            continue
        latest_date, latest_value = points[-1]
        prior_value = points[-2][1] if len(points) > 1 else None
        out[series_id] = {
            "label": label,
            "unit": unit,
            "cadence": cadence,
            "as_of": latest_date,
            "value": round(latest_value, 4),
            "prior_value": round(prior_value, 4) if prior_value is not None else None,
            "change": round(latest_value - prior_value, 4) if prior_value is not None else None,
        }
    out["note"] = (
        "Official FRED-published series, not market prices — each updates on its own release "
        "schedule (monthly for CPI/Fed funds/unemployment, quarterly for GDP) and may be revised "
        "in later releases after initial publication. 'as_of' is the period the value describes, "
        "not necessarily today's date — a monthly CPI print released now typically describes last "
        "month, sometimes the month before. Never call these figures 'current.'"
    )
    return out
