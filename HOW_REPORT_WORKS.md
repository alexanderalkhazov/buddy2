# How `report` Works, Step by Step

There is exactly one command in this program, with no arguments:

```bash
python main.py report
```

It prints the hottest US stocks to trade right now — long and short — each
with a plain-English **why, when, and how**. Zero-LLM: every number is
fetched from a real API or computed by a deterministic/ML formula in
`processing/`. No AI-interpretation step is needed to use the output.

*(There used to be a second mode, `report TICKER`, for a single-stock deep
dive. It was removed — this project now does one thing: surface the
hottest tradeable stocks with real reasoning attached, not a general-purpose
research tool.)*

---

## Step 0 — Command dispatch (`main.py`)

`argparse` parses the `report` subcommand (that's the only one), calls
`ui/market_report.py:generate()`, prints the result, and writes it to
`--out` if given. `--refresh` forces a re-fetch of cached price/news data
instead of serving the cache.

---

## Step 1 — Rank every stock in every sector (`_ranked_predictions()`)

This is the core of the report. For all ~104 tickers across all 13 US
equity sectors (technology, semiconductors, software/internet, financials,
healthcare, industrials, energy, consumer discretionary, consumer staples,
utilities, materials, communication services, real estate):

1. Pull each sector's calibrated `P(top-3-of-13 sectors by 20D return)`
   from `sector_prediction_model` (Step 5 below) — the one part of this
   system that's actually walk-forward validated.
2. Take a live technical snapshot of the ticker itself: RSI, Stochastic,
   ADX, ROC, Bollinger %B, MACD, CCI, Ichimoku cloud position — reduced to
   one `technical_composite` score (0-100,
   `processing/indicators.py:technical_composite_score`, 7 indicator
   families).
3. `long_score` = the sector probability, nudged up/down by up to ±10%
   based on how far the ticker's own composite sits from neutral (50).
   `short_score` mirrors this on the bearish side. The nudge changes
   ranking order for real — it's not cosmetic — but only the sector
   probability itself is walk-forward validated; the nudge is a stated,
   unvalidated tiebreaker.
4. Computes a real entry/stop/TP1 per ticker
   (`processing/scoring.py:trade_levels()`, ATR-based).
5. The top 5 by each score become the **TOP LONG CANDIDATES** / **TOP
   SHORT CANDIDATES** — printed right after the model's own measured OOS
   reliability (AUC/Brier vs. a coin-flip baseline).

---

## Step 2 — Why / When / How, per candidate

For every one of the 10 candidates (5 long, 5 short), the report builds a
dedicated section:

**HOW** — the entry/stop/TP1 from Step 1, restated plainly.

**WHY (indicators)** — `_indicator_reasons()` walks every one of the 7
technical families for THIS ticker and states, honestly, whether each one
confirms or conflicts with the candidate's direction (e.g. "MACD histogram
+6.70 — bullish crossover — confirms the long direction," or the reverse
when an indicator disagrees). Nothing is cherry-picked; a candidate can
have a mixed picture and the report says so. It also restates the
sector-level probability that got this ticker onto the list in the first
place.

**WHY (news)** — `_news_for_candidate()` fetches this ticker's own real,
recent news (same pipeline — NewsAPI or yfinance fallback, lexicon
sentiment — as everywhere else in this project) and reports the single
most recent article: title, **source**, publish date, sentiment, a direct
**source link**, and an explicit **SUPPORTS / CONFLICTS WITH / NEUTRAL**
call against the candidate's direction. If no usable news exists, it says
so rather than fabricating a headline.

**WHEN** — `_when_for_candidate()` fetches the ticker's next earnings date
and reports an earnings-proximity severity tier
(`processing/scoring.py:earnings_proximity_risk()`) — an earnings gap can
jump past a stop with no fill available at that level, so this flags when
that risk is live.

Each section closes with the same honest reminder: the sector-level
probability is the validated part; the indicator/news reasoning is real,
live-computed context, not a second backtested signal.

---

## Step 3 — Supporting data

Everything else the report has ever shown is still there, below the
candidate sections, labeled **SUPPORTING DATA**:

- **Market Regime** — SPY/QQQ trend vs. their own SMAs, the full 7-indicator
  technical set per index, a **market_technical_composite** and
  **market_health** label (STRONG/CONSTRUCTIVE/WEAK/POOR — descriptive,
  not a forecast), VIX level, and overall risk-on/risk-off label.
- **Macro Context** — market-based proxies (Treasury yields, dollar, oil,
  gold, a self-cross-checked yield-curve spread), FRED's lagged
  (CPI/GDP/unemployment) and daily (2Y yield, credit spreads) series, US
  market breadth (equal-weight vs. cap-weight, small vs. large-cap) and VIX
  term structure, and Polymarket prediction odds.
- **Market News Digest** — a 5-bellwether, 48h sample of broad market
  headlines (separate from the per-candidate news in Step 2).
- **Sector Opportunity Ranking** — all 13 sectors' current readings, plus
  the machine-computed `research_edge_status` checklist (Phase 5) showing
  exactly which validation tests this system's core finding has and hasn't
  survived.
- **Sector Leaders**, **Indices Snapshot** (S&P 500/Nasdaq/Dow/Russell),
  **Crypto Snapshot** (BTC/ETH/SOL — explicitly outside this project's
  backtested research), and a final **Reliability & Limitations** table.

---

## Step 4 — `sector_prediction_model`

The one validated component. Trains a calibrated gradient-boosting +
logistic-regression ensemble to predict each sector's probability of
landing in the top-3-of-13 by 20-day forward return, using purged,
embargoed, walk-forward validation (never trained-then-graded on the same
data). Measured result: AUC ≈ 0.60, Brier ≈ 0.175 vs. a 0.178 coin-flip
baseline — a real but modest edge, reported as such, not oversold.
Retrain it directly with `python -m processing.ml.sector_prediction_model`.

The deeper research behind it — five phases of deliberately trying to
disprove the finding before trusting it (non-overlapping tests, sector-
neutralization, adversarial shuffling, a frozen holdout) — lives in
`processing/ml/phase1..5_*.py` and `storage/models/*.json`, and is
summarized in the report's Sector Opportunity Ranking section.

---

## The one invariant

**Nothing in `ui/` ever invents a number.** Every value is fetched from a
real API or computed by a named, documented formula in `processing/`.
Where something is uncertain, unvalidated, or conflicting, the report says
so in the same breath as the number — including inside each candidate's
own why/when/how section, not just in a disclaimer at the end.
