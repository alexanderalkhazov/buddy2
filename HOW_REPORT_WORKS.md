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
   ADX, ROC, Bollinger %B, MACD, CCI, Ichimoku cloud position, Money Flow
   Index, Aroon Oscillator, Keltner Channels — reduced to one
   `technical_composite` score (0-100,
   `processing/indicators.py:technical_composite_score`, 10 indicator
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
   reliability (AUC/Brier vs. a coin-flip baseline), which is itself
   immediately followed by the trade-mechanics backtest below.

**Diversification cap** — `_diversified_top_n()` selects the top 5 with a
hard cap of `MAX_CANDIDATES_PER_SECTOR` (2) per sector, applied greedily
in score order (only relaxed, and flagged when it is, if too few distinct
sectors clear the cap to fill 5 slots). Institutional portfolio guidance
caps a single sector around 20-30% of a portfolio; this scales that
convention down to a 5-name list as a risk CONSTRAINT applied at report-
assembly time, not a fitted parameter tuned against backtest P&L — doing
the latter would defeat the point of a risk control. Without it, the
ranking correctly but unhelpfully surfaces 5 names from whichever single
sector scores highest that run.

**Does the mechanics itself work, separate from sector AUC?**
`processing/ml/trade_mechanics_backtest.py` answers this directly rather
than leaving it as an open question: for each historical OOS rebalance
date in the SAME walk-forward folds `sector_prediction_model.py:evaluate()`
uses, it picks the exact ticker this ranking logic would have picked (the
top-3-predicted sector's highest-technical-composite member for LONG,
lowest for SHORT), computes the same ATR entry/stop/TP1
(`processing/scoring.py:trade_levels()`), and walks forward day-by-day
through REAL subsequent price bars to see whether TP1 or the stop was hit
first. Measured result (~3 years, walk-forward OOS): **LONG 47.1% win rate
/ +0.18R expectancy per trade before costs** (435 resolved trades) — a
real, small-sample positive result — versus **SHORT 31.6% win rate /
-0.21R expectancy** (418 resolved trades) — measurably negative, not
merely "unvalidated." The report states both numbers plainly, right next
to the candidate tables they describe, rather than only as a disclaimer.
Re-run with `python -m processing.ml.trade_mechanics_backtest`.

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

The top headline is also checked against
`processing/news_proc.py:is_earnings_surprise_headline()` — a narrow regex
(beat/miss/top/exceed + estimate/expectations/forecast, or "earnings
beat/miss/surprise") that flags EARNINGS-SURPRISE headlines specifically,
not general sentiment. Post-earnings-announcement drift (PEAD) is one of
the most-studied real anomalies in finance, distinct from ordinary
sentiment — but it's genuinely contested for large, liquid names like most
candidates here (Martineau 2022 finds it largely vanished for non-
microcaps after 2006, while other work still finds it 2008-2019), so the
report labels this an event worth noting, not a stronger signal than the
sentiment/alignment call already gives it. Deliberately NOT built out into
a full M&A/guidance/generic taxonomy — there's no comparable evidence base
for those categories, and a keyword classifier there would just be false
precision.

**WHEN** — `_when_for_candidate()` fetches the ticker's next earnings date
and reports an earnings-proximity severity tier
(`processing/scoring.py:earnings_proximity_risk()`) — an earnings gap can
jump past a stop with no fill available at that level, so this flags when
that risk is live.

**DAY TRADE OR SWING TRADE?** — `processing/scoring.py:trade_style_fit()`
takes six daily-bar tradability inputs computed by
`processing/indicators.py:tradability_stats()` (20-day average dollar
volume, average daily range %, average overnight gap %, the gap's share of
total movement, and the Kaufman Efficiency Ratio) plus ADX14, and returns
two *separate* verdicts that are never blended into one score:

- **Swing fit** (`GOOD` / `WORKABLE` / `POOR`) — this is the horizon the
  system actually models, since `sector_prediction_model` targets
  20-trading-day forward returns. It checks liquidity for a multi-day
  hold, trend presence (ADX ≥ 25 trending, < 20 choppy, 20–25 ambiguous —
  Wilder's own bands), path cleanliness (Efficiency Ratio ≥ 0.30), and
  overnight gap exposure, which is the main structural risk of holding
  through sessions.
- **Day-trade candidacy screen** (`TRADABLE_INTRADAY` /
  `NOT_LIQUID_ENOUGH_INTRADAY`) — liquidity ≥ $20M/day, average daily
  range ≥ 1.5%, price ≥ $5. This is a *screen, not a signal*, and the
  report prints that caveat every single time. The system holds daily
  bars only; bid/ask spread, intraday RVOL, premarket activity, and
  opening-range behavior — the things that actually decide whether a day
  trade works — are all unmeasurable here. A pass means intraday traders
  *could* work the name, pending intraday confirmation, never that this
  system has an intraday edge. It has none, and has never tested for one.

Both verdicts above are UNCONDITIONAL 20-day averages — they say whether a
stock is GENERICALLY suited to a style, not whether TODAY specifically
matches or breaks that pattern. `trade_style_fit()` cross-checks the
systemic verdict against three own-history-relative signals from
`tradability_stats()` (chosen from web research specifically into what
sharpens the day-vs-swing decision, not direction prediction):

- **Relative Volume** (`rvol_20d`) — today's volume vs. the mean of the
  PRIOR 20 days (today excluded from its own baseline).
- **ATR% percentile rank** (`atr_pct_percentile_252d`) — where today's
  ATR% (ATR14/close) sits within its own trailing ~year of history.
- **52-week high/low proximity** — descriptive trend-context only (near a
  real structural level tends to produce cleaner follow-through), never a
  pass/fail gate on swing_points.

When a stock fails the generic day-trade floor but RVOL ≥ 1.5x AND ATR%
percentile ≥ 80th simultaneously, a note flags today as a possible
transient exception. When a stock passes generically but RVOL ≤ 0.5x AND
ATR% percentile ≤ 20th, a note flags today as unusually quiet — the
generic PASS may not hold today specifically. Neither case changes
`day_trade_screen`'s systemic value; conflating a one-day event with a
stable trait is exactly the false precision this system avoids.

**VERDICT** — `processing/scoring.py:conviction_verdict()` combines
everything above into one STRONG/MODERATE/WEAK/AVOID call, as an explicit
point system rather than a black box:

- Indicator agreement ratio (`ui/market_report.py:_indicator_agreement_counts()`
  — how many of the 7 technical families actually confirm vs. conflict
  with this direction, not just the composite's own scalar).
- Earnings-proximity severity (IMMEDIATE/ELEVATED subtracts; APPROACHING
  subtracts less).
- Swing-fit (GOOD adds, POOR subtracts).
- **A hard gate**: if the trade-mechanics backtest (Step 1.5 above)
  measured NEGATIVE expectancy for this direction, the verdict is capped
  at AVOID regardless of every other point above. This is deliberate — a
  measured fact about whether the entry/stop/TP1 convention itself has
  worked historically is a stronger, more specific claim than any
  per-candidate heuristic, and should never be outvoted by a few agreeing
  oscillators. It's why every SHORT candidate currently shows AVOID even
  when its own indicators look fine — the cap is about the mechanics
  every SHORT candidate shares, not about that specific ticker.

Each section closes with the same honest reminder: the sector-level
probability is the validated part; the indicator/news reasoning is real,
live-computed context, not a second backtested signal.

---

## Step 3 — Supporting data

Everything else the report has ever shown is still there, below the
candidate sections, labeled **SUPPORTING DATA**:

- **Market Regime** — SPY/QQQ trend vs. their own SMAs, the full 10-indicator
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
data). Measured result: AUC ≈ 0.59, Brier ≈ 0.176 vs. a 0.178 coin-flip
baseline — a real but modest edge, reported as such, not oversold.

Features: the base price/technical set (features.py), sector breadth (%
of member tickers above their own SMA50/200, % with positive 20D return,
median RSI — sector_dataset.py), the vol×momentum interaction terms Phase
5's decomposition motivated, and 4 market-regime features (trend regime,
vol regime, VIX 5-day change, 10Y-2Y yield curve slope —
processing/ml/regime.py + macro_features.py). That last group was chosen
by walk-forward ablation from 8 candidates: each tested alone and in
combination against the base feature set, keeping only what actually
improved OOS AUC/Brier. VIX *level*, 20D realized SPY vol, and both
credit-spread features were tried and DROPPED — each measurably hurt OOS
performance alone, and the credit-spread series turned out to have only
~3 years of history on this project's no-API-key FRED endpoint (vs. 50+
for the yield-curve series), silently cutting ~30% of training rows when
included. See `sector_dataset.py:REGIME_FEATURE_COLUMNS` for the kept/
dropped list and reasoning.

This metric has moved twice in this project's history: ≈0.60 -> ≈0.58
after fixing a walk-forward embargo unit bug (embargo applied in calendar
days against a trading-day label horizon, undercounting the required gap)
and a sector-index construction bug (one member ticker's pre-common-date
drift biasing the whole composite) — both moved it DOWN, consistent with
removing real leakage rather than just adding noise. Then ≈0.58 -> ≈0.59
after adding the regime features above — a genuine, ablation-tested
improvement, not just a change.

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
