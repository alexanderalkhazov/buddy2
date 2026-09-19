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
   own live technical reading, capped at 2 candidates per sector (per web
   research into portfolio-concentration-risk guidance — without the cap,
   the top 5 would often be 5 correlated bets on whichever single sector
   ranks highest, not 5 diversified ideas).
2. **Why / When / How, per candidate** — for every one of the 10
   candidates:
   - **HOW** — a real entry/stop/take-profit (ATR-based).
   - **WHY (indicators)** — all 10 technical indicator families (RSI,
     Stochastic, ADX/ROC, Bollinger %B, MACD, CCI, Ichimoku Cloud, Money
     Flow Index, Aroon Oscillator, Keltner Channels) stated honestly,
     including when one *conflicts* with the candidate's direction — never
     cherry-picked to look one-sided.
   - **WHY (news)** — that ticker's own real, recent headline: title,
     source, date, sentiment, a direct source link, and an explicit
     SUPPORTS / CONFLICTS WITH / NEUTRAL call against the direction —
     flagged separately when it reads as an earnings beat/miss headline
     specifically (post-earnings-announcement drift is a real, if
     contested-for-large-caps, anomaly, distinct from generic sentiment).
   - **WHEN** — earnings-proximity risk from the ticker's real next
     earnings date (a gap can jump past a stop with no fill available).
   - **VERDICT** — STRONG/MODERATE/WEAK/AVOID, an explicit, inspectable
     point system combining indicator agreement ratio, earnings-proximity
     severity, swing-fit, and — as a hard gate that overrides everything
     else — the measured trade-mechanics backtest expectancy for that
     direction (see below). A negative measured expectancy caps every
     candidate in that direction at AVOID regardless of how good any one
     ticker's own technicals look, because that's a fact about the
     mechanics, not a heuristic about the ticker.
   - **DAY TRADE OR SWING TRADE?** — a suitability verdict computed from
     daily-bar tradability stats: 20-day average dollar volume, average
     daily range %, average overnight gap %, the gap's share of total
     movement, ADX14 (Wilder's own 25/20 trending/choppy thresholds), and
     the Kaufman Efficiency Ratio (how much of the last 20 days' total
     travel became net directional movement) — all UNCONDITIONAL 20-day
     averages, cross-checked against three OWN-HISTORY-relative signals
     (per web research into what specifically sharpens this decision):
     Relative Volume vs. the stock's own 20-day norm, ATR% percentile rank
     against its own trailing year, and 52-week high/low proximity. The
     cross-check exists specifically so a stock that fails the generic
     day-trade floor but is having an unusual (high-volume, high-range)
     day gets flagged as a transient exception — and a stock that passes
     generically but is unusually quiet today gets flagged too — without
     either changing the honest systemic verdict, which stays a stable
     trait read, not a one-day event dressed up as one. The swing verdict
     is the horizon this system actually models — `sector_prediction_model`
     targets 20-trading-day returns. The day-trade line is a *candidacy
     screen only, pending intraday confirmation*, and says so in the same
     breath: this system holds no intraday data and has never tested
     anything at an intraday horizon, so a pass means day traders **could**
     work the stock, never that the system thinks you should.
3. **Supporting data** — Market Regime (index trend, VIX, a 10-indicator
   technical composite, a plain market-health label), Macro Context
   (Treasury yields, FRED CPI/GDP/unemployment + daily credit spreads,
   market breadth, Polymarket odds), a Market News Digest, the full
   13-sector ranking table with the machine-computed research-validation
   checklist, Sector Leaders, Indices, and Crypto snapshots, and a final
   Reliability & Limitations table.
4. **A prompt for another AI, at the very end** — the report's own "Layer
   3" hand-off: a copy-paste block for a *separate* AI chat (this system
   stays zero-LLM; nothing here calls an LLM itself) instructing it to
   turn the tables above into a final execution-ready order list —
   entry as a limit order, stop as a stop-market order, take-profit as a
   limit order — while extracting only numbers already in the report
   (never inventing one) and respecting the report's own AVOID gate (a
   candidate the trade-mechanics backtest measured negative expectancy
   for stays excluded, even if it looks fine on paper).

## Does the entry/stop/TP1 mechanics actually work?

A separate walk-forward backtest (`processing/ml/trade_mechanics_backtest.py`)
answers this directly, rather than only reporting sector-level AUC: for each
historical OOS rebalance date, it picks the same ticker the report's own
ranking would have picked, computes the same ATR entry/stop/TP1, and walks
forward through REAL subsequent price bars to see which was hit first.

**LONG**: 47.2% win rate (TP1 vs. stop) over 430 resolved trades (~3 years) →
**+0.18R per trade before costs** — a real, if modest and small-sample,
positive result.
**SHORT**: 34.6% win rate over 413 resolved trades → **-0.13R per trade
before costs** — measurably negative, not merely "unvalidated." The report
states this plainly next to the short candidates rather than only flagging
them as unvalidated.

That SHORT number itself moved once, for a documented reason:
`processing/ml/short_selection_research.py` tested 3 SHORT candidate-
selection rules (the naive "lowest technical_composite" rule production
used before, "highest technical_composite," and "lowest technical_composite
excluding already-oversold names") — all 3 remain net-negative, but
excluding oversold names (the ones most prone to a mean-reversion bounce/
short squeeze) measurably helps: -0.21R → -0.13R. Production now uses that
better rule. It is still AVOID — a real, tested improvement was shipped,
but it was never going to be dressed up as "shorts now work," because they
don't, on any selection rule tested so far.

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

**Methodology review** (web research into the best-evidenced algorithmic
approaches for this exact problem shape): CPCV, LightGBM/XGBoost/CatBoost,
and a learned stacking meta-learner were all researched and NOT
implemented — the evidence says none would help at this sample size (only
2-3 usable OOS folds), and would mainly add overfitting surface. The one
recommendation that looked genuinely promising — per-date cross-sectional
rank normalization of every feature — was implemented and ablation-tested
anyway, and REJECTED: individually-promising features (e.g. one alone
moved AUC to 0.597) collapsed below baseline when combined, the same
instability signature that got the sector-ETF features rejected earlier —
noise from a small OOS sample, not real signal (see
`processing/ml/sector_prediction_model.py:CROSS_SECTIONAL_RANK_SOURCE_COLUMNS`).
A rank-averaged ensemble blend was also tested and rejected (worse on both
AUC and Brier). Honest conclusion from the research: **AUC ≈ 0.59 on this
problem shape (13 cross-sectional units, technical+macro features only, no
fundamentals/alt-data, 20-day horizon) is plausibly near the realistic
ceiling** — comparable to or better than AUC ≈ 0.547 reported in the
literature for a much larger-N, richer-feature single-stock XGBoost study.
More model sophistication is more likely to buy overfitting than edge.

**A second, longer horizon** (`processing/ml/multi_horizon_research.py`)
tested whether predictability differs by horizon — reusing the exact same
feature set and walk-forward machinery, at 1/5/20/60 trading days, with
BOTH the classification target above and an independent Ridge-regression
excess-return target (evaluated by Spearman rank IC). Both target types
agree: 1D has no edge (AUC 0.53, IC -0.01), and predictability rises with
horizon — **60D is genuinely stronger than 20D on both metrics** (AUC
0.607 vs 0.590, IC 0.177 vs 0.104). Two independently-fit target
formulations agreeing on the same horizon ranking is a meaningful cross-
check. Added as an ADDITIVE second output (the report's own "60-Day
Outlook" section) — not a replacement, since 60D hasn't yet been through
the same depth of stress-testing (Phase 4/5-style adversarial/non-
overlapping checks) as 20D, and no candidate ranking or trade level
anywhere in the report uses it yet.

**Abstention gate** (`processing/ml/sector_prediction_model.py:abstention_gate()`):
every sector prediction now carries a `has_edge` flag — data-driven, not a
hardcoded probability cutoff like "P>0.55=BUY". It buckets the model's
pooled OOS predictions into 5 calibration ranges and checks whether a
given probability's bucket actually beat the base rate historically
(`has_edge`), and separately whether it cleared that bucket's own margin
of error (`clears_margin_of_error` — a stricter, confidence-interval-style
version of the same check). Building this surfaced a real methodological
trap worth stating plainly: an earlier version required the margin-
adjusted test to pass for `has_edge` itself, and it failed EVERY live
sector, every run — this model's probabilities are compressed near the
base rate (a real but modest AUC~0.59 ranking edge doesn't imply
confidently separated probabilities), so with only 5 buckets nothing ever
cleared a full one-sided margin. That would have silently made the report
say "no edge" forever, misrepresenting a model that does show real
aggregate discrimination. Fixed to a point-estimate standard for
`has_edge` — the same standard `evaluate()`'s own overall PASS/FAIL check
already uses (mean_auc > 0.55 is also a point estimate) — with the
stricter margin-adjusted result kept alongside, not discarded, so a
reader can tell "real but unremarkable" apart from "confidently
separated."

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
