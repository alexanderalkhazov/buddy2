# Trading Research Assistant

A local console tool that fetches real market data and computes deterministic
statistics about it — zero LLM calls, zero API cost, zero invented numbers. It
prints a large, structured Markdown data package you paste into any chatbot
(or read yourself) for interpretation. The program never tells you to buy or
sell anything; it assembles evidence and, for the market-wide report, an
explicit prompt telling an AI how to reason over that evidence honestly.

**No API key is required to run anything in this repo.** `NEWSAPI_KEY` is
optional (falls back to yfinance headlines without it).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # optional: add NEWSAPI_KEY for richer news coverage
```

## Commands

There is exactly one command, with two modes:

```bash
python main.py report              # market-wide report (no ticker)
python main.py report AAPL         # per-ticker report
python main.py report AAPL --peers MSFT GOOGL --account-size 100000 --risk-pct 1
python main.py report AAPL --refresh --out aapl_report.md
```

| Flag | Applies to | Meaning |
|---|---|---|
| `TICKER` (positional, optional) | both | omit for the market-wide report |
| `--account-size` | per-ticker | enables the Position Sizing section |
| `--risk-pct` | per-ticker | % of account risked per trade (default 1.0) |
| `--peers TICKER...` | per-ticker | adds a peer-relative valuation comparison |
| `--refresh` | both | force-refetch cached price/news/sector data instead of using the cache |
| `--out FILE` | both | also write the report to a file |

---

## `report` (no ticker) — the market-wide report

Answers "what's going on in the market right now, and is there anywhere worth
looking?" It has no opinion about any single stock — it's regime, macro,
news, and a sector ranking, handed to an AI with explicit instructions on how
to reason over it.

**What it fetches/computes** (`ui/market_report.py`):

1. **Market Regime** — SPY & QQQ trend (price vs. their own 20/50/200-day
   moving averages) and the VIX level, reduced to a bull/bear/mixed label and
   a risk-on/risk-off/neutral tag. Index-trend + VIX only — there's no
   breadth/advance-decline data source in this system.
2. **Macro Context**, kept as three explicitly separate groups (never
   blended into one number, since they update on different schedules and
   answer different questions):
   - *Market-based proxies*: 10Y/13W/30Y Treasury yields, US Dollar Index,
     WTI crude, gold — all daily-close market pricing (what the market is
     pricing right now).
   - *Official economic indicators*: CPI, Fed funds rate, unemployment, real
     GDP growth — via FRED's public CSV endpoint, no key needed. Lagged by
     weeks to months, explicitly labeled with their as-of date.
   - *Prediction markets*: Polymarket odds on Fed decisions, recession,
     inflation, unemployment (best-effort; degrades to an empty list if the
     API is unreachable rather than failing the whole report).
3. **Market News Digest** — last 48h of headlines from SPY, QQQ, AAPL, MSFT,
   NVDA (a practical stand-in for "market news," not exhaustive coverage),
   deduped and lexicon-sentiment-tagged.
4. **Sector Opportunity Ranking** — all 13 sectors in the research universe
   (`processing/ml/universe.py`), ranked by 20-day realized volatility, each
   flagged with its momentum bucket. This is the one part of the report
   grounded in this project's own out-of-sample research (see *The ML
   research phases* below) rather than a live data dump — sectors combining
   high volatility with negative recent momentum have historically shown the
   strongest subsequent 20-day returns.
5. **Sector Leaders** — the actual member tickers of the top 3 ranked
   sectors, each with a live price/change%/RSI/trend snapshot, so an AI has
   concrete named candidates instead of just a sector label.
6. **Indices Snapshot** — S&P 500, Nasdaq Composite, Dow, Russell 2000.
7. **Crypto Snapshot** — BTC, ETH, SOL. Explicitly flagged as outside this
   project's backtested research (raw price/trend/RSI only — no validated
   edge behind it, unlike the equity sector ranking).
8. **Reliability & Limitations** — states plainly that no model here predicts
   market *direction* (see Phase 1/2 below), that the sector ranking is real
   but modest, and that the sector universe is survivorship-biased (today's
   liquid large-caps per sector, not point-in-time historical constituents).

The report opens with a reasoning prompt (not a data dump with a disclaimer
tacked on) that instructs an AI to: read the regime/macro/news for agreement
or conflict, weigh the sector ranking honestly, name specific candidates
across stocks/indices/crypto *only* when grounded in a cited data point
above, give a confidence level per asset class, and explicitly answer
**NO TRADE** when the evidence doesn't justify one rather than forcing a
recommendation.

---

## `report TICKER` — the per-ticker report

Answers "what does the evidence say about this one company, right now?" via
the same fetch-then-compute pipeline (`processing/bundle.py` →
`ui/report.py`), covering:

- **Price & Technicals** — last close, RSI14, MACD, SMAs, ATR, volume vs.
  30-day average, trend label, and Relative Strength vs. SPY and the
  ticker's sector/industry ETF (is this move stock-specific or is the whole
  sector moving?).
- **Fundamentals** — margins, growth, cash flow, balance sheet, valuation
  multiples beyond P/E (EV/EBITDA, P/S, PEG), ownership/insider activity.
- **Valuation Sensitivity** — the trailing-to-forward EPS gap stated as a
  plain arithmetic fact (never as "the market is pricing in X% growth," a
  claim the number doesn't support), EPS/revenue estimate dispersion
  (analyst disagreement), a bear/base/bull scenario grid with every
  assumption labeled as an assumption, and **Thesis Dependencies** — the
  concrete, plain-English conditions that have to stay true for the current
  valuation to hold, plus separate fundamental-trigger invalidation
  conditions (distinct from the price-based stop below).
- **News** — recent headlines for that ticker, sentiment-tagged.
- **Macro Context** — the same three groups as the market report.
- **Baseline Score** — see *How the score works* below.
- **Investment Thesis vs. Trade Setup** — two deliberately separate
  numbers: `company_quality` (fundamental + valuation only — "is this a
  good business at a reasonable price") vs. `entry_quality` (price location
  right now — distance from SMA20/52-week-high, RSI extension, ATR% —
  "is now a good time to start a position"). A stock can be a BULLISH
  business with a BEARISH entry; the report says exactly that instead of
  collapsing both into one number.
- **Historical Performance** — trailing 1D through 5Y returns, real price
  data, not a strategy result.
- **Strategy Screen** — 6 published technical systems (golden cross, RSI
  mean-reversion, etc.) backtested and ranked by Sharpe, with an explicit
  **multiple-testing warning** (the top-of-6 result is a maximum over
  trials, not a single a-priori test), out-of-sample and regime-window
  checks on the winner, and a Monte Carlo bootstrap on its actual trade
  sequence.
- **Trade Levels** — hypothetical LONG entry/stop/TP1/TP2 from ATR, with an
  earnings-proximity warning (a stop assumes continuous price movement,
  which an earnings gap can violate) — explicitly labeled "not a
  recommendation," just the arithmetic if a long entry were taken here.
- **ML Prediction Engine** — see *How the score works* below.
- **Data Quality / Model Reliability** — an inventory of what above is
  checkable fact vs. a reproducible-but-unvalidated formula vs. untested,
  so nothing reads as more certain than it is.

### How the score works

**Baseline Score** (`processing/scoring.py`) is fixed, hand-written
arithmetic — the same rule for every ticker, every time, never fitted or
backtested:

| Component | Weight | Formula |
|---|---|---|
| Fundamental | 40% | profit margin (0-40% → 0-50 pts, linear) + revenue growth (-20% to +40% YoY → 0-50 pts, linear) |
| Technical | 30% | # of SMA20/50/200 price sits above (0-50 pts) + RSI centered on 45-65 (0-35 pts) + MACD histogram sign (0-15 pts) |
| Valuation | 20% | cheaper forward P/E scores higher — percentile vs. peers if `--peers` given, else a flat P/E≤10→90 / P/E≥40→10 heuristic |
| Sentiment | 10% | net positive-minus-negative recent headlines, capped ±50 around neutral 50 |

`Overall` = weighted average, re-normalized across whatever components are
actually available (a missing component is excluded, not scored as zero).
**These weights and thresholds are a reasonable, transparent starting point
I chose — not validated as predictive of future returns**, and the report
says so explicitly rather than implying a high score means "safe" or "buy."
`risk_flags` (cyclicality, momentum-extension, volatility) answer a
*different* question — how extended/risky the setup looks — and must be
read alongside the score, never in place of it.

**ML Prediction Engine** (`processing/ml/`) is the actual attempt at
statistically-validated prediction: real historical data, real walk-forward
out-of-sample testing, calibrated probabilities (Platt scaling), a
historical-analog engine. Its honest, tested result on price/technical
features: **no measurable directional edge** (AUC ≈ 0.50, the same as a
coin flip) — which is why it always reports `reliability: LOW` and
`recommended_gate: NO TRADE`. This is not a bug; it's the system correctly
refusing to manufacture confidence it doesn't have. See *The ML research
phases* below for what was actually found to work.

---

## The ML research phases

Four rounds of rigorous, leakage-tested research live in `processing/ml/`,
each documented in a JSON report under `storage/models/` and logged
permanently (never overwritten) to `storage/models/experiments.jsonl`. Short
version, in order:

- **Phase 1** — built a point-in-time-safe, cross-sectional (multi-ticker)
  dataset, walk-forward validation with a purge/embargo, and a calibrated
  direction model. Result: **no edge** (AUC ≈ 0.50) on price/technical
  features for a single stock's future direction.
- **Phase 2** — swept horizons (1D-60D), absolute vs. relative-to-SPY
  targets, and threshold targets, plus adversarial sanity checks (shuffled
  labels correctly collapse to ~0.50, confirming the test harness itself
  isn't leaking). Found one interesting univariate signal: volatility
  (ATR%/realized vol) correlates with subsequent returns — but on only 32
  tickers, mostly mega-cap tech.
- **Phase 3** — re-tested that volatility finding on a broader, sector-
  balanced 103-ticker/13-sector universe. It survived and got *stronger*
  (IC ≈ 0.11), survived momentum-residualization (not a momentum proxy),
  and held across bull/bear regimes — but ~75% of the raw effect disappeared
  after sector-neutralizing, meaning it's mostly a **sector-level**
  phenomenon, not real stock-picking skill.
- **Phase 4** — decomposed volatility into market/sector/idiosyncratic
  components directly: **idiosyncratic (stock-specific) volatility carries
  almost no signal (IC ≈ 0.02)**; the sector component dominates (IC ≈
  0.08). Built and backtested a sector-rotation rule (long the highest-
  volatility sector, especially with negative momentum) that survives
  transaction costs and an out-of-sample holdout. This is the finding
  surfaced in the market report's Sector Opportunity Ranking.

**Bottom line the whole research effort converged on**: this system cannot
predict whether an individual stock will go up. It *can* say, with modest
but real evidence, which *sector* currently looks statistically more
interesting than the others. Every report and the ML engine itself says
this plainly rather than implying otherwise.

Run any phase yourself: `python -m processing.ml.phase4_hierarchical` (etc.
for `train`, `phase2_run`, `phase3_volatility`). Run the leakage tests with
`python tests/test_ml_leakage.py`.

## Architecture

```
main.py       CLI entry point — the single `report` command
ui/           report.py (per-ticker), market_report.py (market-wide) — pure
              Markdown formatters, zero LLM calls, zero interpretation
processing/   indicators, scoring, valuation, backtest, regime, relative
              strength, macro aggregation, and ml/ (the four research phases:
              dataset builders, labels, leakage tests, training, sector
              index/dataset construction, the experiment registry)
data/         prices (yfinance), news, polymarket, fundamentals, macro (FRED),
              edgar — each source reduced to a compact dict, no raw payloads
              leak past this layer
storage/      DuckDB cache (prices, news) + models/ (trained ML artifacts,
              experiment log, phase reports)
tests/        leakage tests for the ML pipeline
```

**The core invariant**: nothing in `ui/` or the ML prediction ever invents a
number. Every value is either fetched from a real API or computed by a
documented, deterministic formula in `processing/`. Where a number is
genuinely uncertain or unvalidated, the report says so in the same breath as
the number, not in a disclaimer at the bottom.

## Expectations

No model in this system reliably predicts short-term price direction for an
individual stock — the ML research phases tested for exactly that and found
none, and say so rather than hiding a negative result. The value here is
fast, honest synthesis of real data, forced explicit reasoning, and a
sector-level finding that did survive rigorous testing. Treat every backtest
as a sanity check on one historical window, not a guarantee — they are easy
to overfit, and this system actively tries to disprove its own findings
before reporting them.
