# Trading Research Assistant

A local console tool that fetches real market data, runs it through a
calibrated ML model and a set of real technical indicators, and prints the
hottest US stocks to trade right now — long and short — each with a
plain-English **why, when, and how**. Zero-LLM: every number is fetched
from a real API or computed by a deterministic/ML formula. No AI
interpretation step is required to use the output.

**No API key is required to run this repo.** `NEWSAPI_KEY` is optional
(news falls back to yfinance headlines without it).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # optional: add NEWSAPI_KEY for richer news coverage
```

## Usage

```bash
python main.py report                  # the report
python main.py report --refresh        # force-refetch cached price/news data
python main.py report --out today.md   # also write it to a file
```

That's the whole CLI — one command, no arguments needed.

---

## What the report contains

1. **TOP LONG CANDIDATES / TOP SHORT CANDIDATES** — the top 5 stocks by
   each direction, ranked across all ~104 tickers spanning all 13 US
   equity sectors (technology, semiconductors, software/internet,
   financials, healthcare, industrials, energy, consumer
   discretionary/staples, utilities, materials, communication services,
   real estate). Ranking combines a calibrated sector-level probability
   (`sector_prediction_model`, walk-forward validated) with each ticker's
   own live technical reading.
2. **Why / When / How, per candidate** — for every one of the 10
   candidates:
   - **HOW** — a real entry/stop/take-profit (ATR-based).
   - **WHY (indicators)** — all 7 technical indicator families (RSI,
     Stochastic, ADX/ROC, Bollinger %B, MACD, CCI, Ichimoku Cloud) stated
     honestly, including when one *conflicts* with the candidate's
     direction — never cherry-picked to look one-sided.
   - **WHY (news)** — that ticker's own real, recent headline: title,
     source, date, sentiment, a direct source link, and an explicit
     SUPPORTS / CONFLICTS WITH / NEUTRAL call against the direction.
   - **WHEN** — earnings-proximity risk from the ticker's real next
     earnings date (a gap can jump past a stop with no fill available).
   - **DAY TRADE OR SWING TRADE?** — a suitability verdict computed from
     daily-bar tradability stats: 20-day average dollar volume, average
     daily range %, average overnight gap %, the gap's share of total
     movement, ADX14 (Wilder's own 25/20 trending/choppy thresholds), and
     the Kaufman Efficiency Ratio (how much of the last 20 days' total
     travel became net directional movement). The swing verdict is the
     horizon this system actually models — `sector_prediction_model`
     targets 20-trading-day returns. The day-trade line is a *candidacy
     screen only, pending intraday confirmation*, and says so in the same
     breath: this system holds no intraday data and has never tested
     anything at an intraday horizon, so a pass means day traders **could**
     work the stock, never that the system thinks you should.
3. **Supporting data** — Market Regime (index trend, VIX, a 7-indicator
   technical composite, a plain market-health label), Macro Context
   (Treasury yields, FRED CPI/GDP/unemployment + daily credit spreads,
   market breadth, Polymarket odds), a Market News Digest, the full
   13-sector ranking table with the machine-computed research-validation
   checklist, Sector Leaders, Indices, and Crypto snapshots, and a final
   Reliability & Limitations table.

## Does the entry/stop/TP1 mechanics actually work?

A separate walk-forward backtest (`processing/ml/trade_mechanics_backtest.py`)
answers this directly, rather than only reporting sector-level AUC: for each
historical OOS rebalance date, it picks the same ticker the report's own
ranking would have picked, computes the same ATR entry/stop/TP1, and walks
forward through REAL subsequent price bars to see which was hit first.

**LONG**: 47.1% win rate (TP1 vs. stop) over 435 resolved trades (~3 years) →
**+0.18R per trade before costs** — a real, if modest and small-sample,
positive result.
**SHORT**: 31.6% win rate over 418 resolved trades → **-0.21R per trade
before costs** — measurably negative, not merely "unvalidated." The report
now states this plainly next to the short candidates rather than only
flagging them as unvalidated.

Re-run it yourself with `python -m processing.ml.trade_mechanics_backtest`
(regenerates `storage/models/trade_mechanics_backtest_report.json`, which
`ui/market_report.py` reads).

## The one thing that's actually validated: `sector_prediction_model`

A calibrated gradient-boosting + logistic-regression ensemble predicting
each sector's probability of landing in the top-3-of-13 by 20-day forward
return, evaluated with purged, embargoed, walk-forward validation (never
trained-then-graded on the same data). Measured result: **AUC ≈ 0.59,
Brier ≈ 0.176 vs. a 0.178 coin-flip baseline** — a real but modest edge,
reported as such.

That number moved twice in this project's history, both times for reasons
worth stating plainly rather than burying: it was ≈0.60 before a walk-
forward embargo unit bug (calendar days applied against a trading-day label
horizon) let a small amount of leakage into the measurement — fixing that
took it down to ≈0.58. It's back up to ≈0.59 after adding 4 real market-
regime features (trend/vol regime, VIX 5-day change, the 10Y-2Y yield curve
slope) — chosen by walk-forward ablation from 8 candidates, keeping only the
ones that actually moved OOS AUC/Brier in the right direction (see
`processing/ml/sector_dataset.py:REGIME_FEATURE_COLUMNS` for the full
ablation result and what was dropped, including why VIX level, realized
vol, and credit spreads didn't make the cut).

That number is the product of five rounds of rigorous, leakage-tested
research (`processing/ml/phase1..5_*.py`, `storage/models/*.json`,
`storage/models/experiments.jsonl` — permanent, never overwritten) that
deliberately tried to disprove the finding before trusting it:

- **Phase 1/2** — found **no edge** (AUC ≈ 0.50) predicting an individual
  stock's own future direction from price/technical features.
- **Phase 2/3** — found one real univariate signal instead: volatility
  correlates with subsequent returns, and it survived a broader
  103-ticker/13-sector universe and momentum-residualization — but ~75%
  of the raw effect disappeared after sector-neutralizing.
- **Phase 4** — confirmed it directly: idiosyncratic (stock-specific)
  volatility carries almost no signal; the *sector* component dominates.
  Built and backtested the sector-rotation rule that's now
  `sector_prediction_model`.
- **Phase 5** — stress-tested it further (a true non-overlapping test,
  adversarial shuffling, a frozen holdout) — the research-validation
  checklist shown in the report's Sector Opportunity Ranking is the
  direct, machine-computed output of this phase.

**Bottom line**: this system cannot predict whether one specific stock will
go up. It *can* say, with real but modest evidence, which sectors
currently look statistically more interesting — and it says so plainly
rather than implying otherwise. Retrain it yourself with
`python -m processing.ml.sector_prediction_model`; run the leakage tests
with `python tests/test_ml_leakage.py`.

## Architecture

```
main.py       CLI entry point — the single `report` command
ui/           market_report.py — the report itself: ranking, why/when/how
              per candidate, and all supporting-data sections. Pure
              Markdown formatting, zero LLM calls.
processing/   indicators (RSI/MACD/Bollinger/Stochastic/ADX/Ichimoku/CCI/
              Chandelier Exit, plus tradability stats + Kaufman Efficiency
              Ratio), scoring (ATR trade levels, earnings-proximity risk,
              day-vs-swing trade-style fit), regime, macro aggregation, bundle assembly,
              and ml/ (the five research phases: dataset builders, labels,
              leakage tests, training, sector index/dataset construction,
              the sector_prediction_model, the experiment registry)
data/         prices (yfinance), news, polymarket, fundamentals, macro
              (FRED), edgar — each source reduced to a compact dict, no
              raw payloads leak past this layer
storage/      DuckDB cache (prices, news) + models/ (trained ML artifacts,
              experiment log, phase reports)
tests/        leakage tests for the ML pipeline
```

**The core invariant**: nothing in `ui/` or the ML pipeline ever invents a
number. Every value is either fetched from a real API or computed by a
documented, deterministic formula in `processing/`. Where a number is
genuinely uncertain or unvalidated, the report says so in the same breath
as the number, not in a disclaimer at the bottom.

See `HOW_REPORT_WORKS.md` for a full step-by-step walkthrough.

## Expectations

No model in this system reliably predicts short-term price direction for
an individual stock — the ML research phases tested for exactly that and
found none, and say so rather than hiding a negative result. The value
here is fast, honest synthesis of real data with a genuine (if modest)
sector-level statistical edge behind the ranking, plus concrete, real
reasoning (indicators and cited news) for every candidate — not a
guarantee, a starting point for someone who can't do all this research
themselves. Treat every backtest as a sanity check on one historical
window, not a guarantee — they are easy to overfit, and this system
actively tries to disprove its own findings before reporting them.
