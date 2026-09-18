# How `report` Works, Step by Step

There is exactly one command in this program: `report`. Whether you get the
**market-wide report** or the **per-ticker report** depends only on whether
you pass a ticker argument.

```bash
python main.py report              # market-wide report
python main.py report AAPL         # per-ticker report
```

Both are **zero-LLM**: every number is either fetched from a real API or
computed by a deterministic formula in `processing/`. No AI writes any of the
data, and no AI-interpretation step is required to use either report — the
market-wide report leads with a direct, ranked LONG/SHORT candidate list
computed by code (Part A1 below); the per-ticker report still opens with a
prompt for whoever wants an LLM's narrative read on the fuller data package
(Part B4), but that step is optional, not required to get a usable answer.

---

## Step 0 — Command dispatch (`main.py`)

1. `argparse` parses the `report` subcommand and its optional positional
   `ticker` argument, plus `--account-size`, `--risk-pct`, `--peers`,
   `--refresh`, `--out`.
2. If a ticker was given, common aliases are resolved first (`SPX` → `^GSPC`,
   `NASDAQ` → `^IXIC`, etc.) so you never need to type a shell-hostile `^`.
3. `main.py` branches:
   - ticker present → `_run_report(args)` → `ui/report.py:generate()`
   - no ticker → `_run_market_report(args)` → `ui/market_report.py:generate()`
4. The returned text is printed to stdout, and written to `--out` if given.
5. Any `ValueError` from a bad/unknown ticker is caught and turned into a
   friendly error (with a "did you mean...?" hint) instead of a stack trace.

---

## PART A — `report` (no ticker): the market-wide report

Source: `ui/market_report.py:generate()`.

### A1. TOP LONG CANDIDATES / TOP SHORT CANDIDATES — the direct output

As of this version, the report leads with a **direct, deterministic
ranking** — no LLM interpretation step, no analyst prompt. `_ranked_predictions()`
scores all ~104 tickers across all 13 US equity sectors:

1. Pull each sector's calibrated `P(top-3-of-13)` from `rocket_science`
   (Part A7 below).
2. For every member ticker of every sector, take a live technical snapshot
   including the full indicator set — RSI, Stochastic, ADX, Bollinger %B,
   ROC, MACD sign, CCI, Ichimoku cloud position — reduced to that ticker's
   own `technical_composite` (0-100,
   `processing/indicators.py:technical_composite_score`, 7 indicator
   families).
3. `long_score` = the ticker's sector probability, nudged up or down by up
   to ±10% based on how far its own `technical_composite` sits from neutral
   (50). `short_score` mirrors this on the bearish side, using
   `1 - sector probability` as the base. **This nudge is not cosmetic — it
   changes ranking order**: two tickers in the same sector (same sector
   probability) are ordered purely by this technical read, so a ticker
   with a stronger composite reliably ranks above a weaker one within its
   sector. Only the sector probability itself is walk-forward validated —
   the technical-composite nudge is an unvalidated tiebreaker, stated as
   such in the report, not hidden in the score.
4. Each ranked row also carries a real entry/stop/TP1
   (`processing/scoring.py:trade_levels()` — ATR-based, same arithmetic as
   Part B's Trade Levels section) — a concrete "buy here, stop there" per
   candidate, not a bare ranking with no action attached.
5. The top 10 by each score are printed as **TOP LONG CANDIDATES** and
   **TOP SHORT CANDIDATES** tables, immediately after the model's own
   measured OOS reliability (AUC/Brier vs. a coin-flip baseline).

**Position sizing is still excluded from this section** — entry/stop/TP1
are shown per candidate, but no share count or account-risk calculation
(that needs `--account-size`, only meaningful per-ticker). This is a ranked
watchlist with real trade levels attached, not a full trade plan;
`report TICKER` is where the fuller picture (TP2, earnings-proximity
warning, alternative stop conventions, position sizing) comes from (Part
B, below).

The rest of the report (Parts A2-A9) follows below as **supporting data** —
the same regime/macro/sector detail as before, kept for anyone who wants to
verify what's behind the rankings, but no longer built around an AI-prompt
framing.

### A2. Market Regime

`processing/regime.py:market_regime()` fetches SPY, QQQ, and `^VIX` and
computes:
- each index's position vs. its own 20/50/200-day SMA (shown as an explicit
  YES/NO table, not just a label, so "mixed" is traceable)
- **the full indicator set per index**: Bollinger %B, Stochastic %K, ADX(14)
  trend strength, ROC(10), CCI(20), Ichimoku cloud position (Tenkan/Kijun/
  Senkou Span A+B), plus a **technical composite (0-100)** blending all of
  those with RSI and MACD sign — 7 indicator families
  (`processing/indicators.py:technical_composite_score`) — never one
  indicator alone
- **market_technical_composite** — the SPY/QQQ average of that composite —
  and a **market_health** label (STRONG/CONSTRUCTIVE/WEAK/POOR), a direct,
  descriptive answer to "is the market technically good right now?" This is
  NOT a forecast — it describes current conditions only; whether strong
  technicals precede good forward returns is a separate, walk-forward-tested
  question answered by Part A7's `rocket_science` model, not by this score.
- VIX level → LOW/NORMAL/ELEVATED/HIGH
- an overall trend (bullish only if **both** SPY and QQQ agree) and a
  risk-on/risk-off/neutral label

### A3. Macro Context

Three explicitly separate groups, via `processing/macro.py`:

1. **Market-Based Proxies** — Treasury yields (10Y/13W/30Y), US Dollar Index,
   WTI crude, gold, and a **derived yield-curve spread (10Y−13W)**. This
   spread is auto-validated: yield values are normalized regardless of which
   unit convention yfinance reports that day (a real historical bug — a
   silent format change once turned a true 1.06pp spread into a wrong
   0.11pp — is now caught automatically).
2. **Official Economic Indicators** (FRED, no key needed), split into:
   - genuinely lagged monthly/quarterly series (CPI, Fed funds, unemployment,
     GDP)
   - daily-updated series (10Y-2Y and 10Y-3M yield-curve spreads, 2Y yield,
     high-yield and investment-grade corporate credit spreads)
   - a **cross-check**: the report's own derived 10Y-13W spread is compared
     against FRED's official 10Y-3M spread every run and flagged if they
     diverge by more than 0.5pp.
3. **US Market Breadth & Volatility Term Structure** — equal-weight (RSP) vs.
   cap-weight (SPY) and small-cap (IWM) vs. large-cap spreads (is the market
   move broad or narrow), plus VIX 9-day/30-day/3-month term structure. If
   the underlying data feed for one leg is stale (this has actually
   happened — yfinance froze two VIX-term tickers for ~2 months once), the
   report reports `UNKNOWN` instead of guessing a contango/backwardation
   state.
4. **Prediction Markets** — Polymarket odds on Fed decisions, recession,
   inflation, unemployment.

### A4. Market News Digest

`data/news.py:fetch_news()` is called for 5 bellwether tickers (SPY, QQQ,
AAPL, MSFT, NVDA) — a practical stand-in for "market news," explicitly
labeled as a sample, not comprehensive coverage. Each ticker's headlines
(NewsAPI if `NEWSAPI_KEY` is set, otherwise yfinance's own feed) are
deduped by URL and near-duplicate title, then cross-ticker deduped, sorted
by recency, and capped at 20. `processing/news_proc.py` tags each with a
lexicon-based sentiment (no LLM key is configured in this project, so the
LLM-summarization path always falls back cleanly to the lexicon).

### A5. Sector Opportunity Ranking — the research-grounded section

This is the one part of the report backed by this project's own multi-phase
research (`processing/ml/`, Phases 1-5), not just a live data pull:

1. All 13 sectors (`processing/ml/universe.py`) get a live snapshot — each
   sector is a **synthetic price index** (`processing/ml/sector_index.py`,
   equal-weighted OHLC average across ~8 member tickers per sector) run
   through the same feature pipeline as an individual stock.
2. Sectors are ranked by 20-day realized volatility, each flagged with a
   **Primary Signal** (HIGH_VOL/LOW_VOL) and **Secondary Modifier**
   (NEGATIVE_MOMENTUM/POSITIVE_MOMENTUM) — named this way, not "bucket,"
   because Phase 5's regression decomposition found volatility is the
   principal driver and momentum only a secondary conditioning variable.
3. Three structured, distinctly-named effect-size fields (never blended):
   `top_sector_vs_equal_weight_benchmark_20d` (excess return), a *different*
   estimand `gross_top_sector_return_20d` (no benchmark subtracted), and an
   itemized `transaction_cost_model` (25bps per side × 2 sides = 50bps
   total) — all pulled directly from `storage/models/phase4_report.json`,
   never hand-typed prose.
4. `research_edge_status` — a machine-computed checklist
   (`storage/models/phase5_report.json`), each check carrying a `PASS` /
   `WEAK` / `NOT_TESTED` status **and a reason**, some marked `[CRITICAL]`.
   The overall status is capped by the *weakest critical check(s)*, not
   averaged — currently **MODERATE** (up from WEAK once
   `non_overlapping_forward_return_test` was actually built and passed —
   see `processing/ml/non_overlapping_test.py`), because `sector_neutral`
   and `point_in_time_universe` still sit below PASS.
5. `current_signal_activation` — computed fresh from the **live** sector
   snapshot (separate from the historical `research_edge_status` above):
   `current_signal_active` (YES/NO) and `current_signal_strength`
   (STRONG/MODERATE/WEAK), each with the exact numeric threshold that was
   or wasn't met, not a subjective label.
6. `research_hypothesis` — the actual regression coefficients from Phase 5's
   interaction decomposition, plus `edge_scope` (equities/sectors/long/20D
   only — explicitly not crypto/indices/individual stocks) and
   `research_horizon`.

### A6. Sector Leaders, Indices, Crypto

- **Sector Leaders**: the actual member tickers of the top 5 ranked sectors
  (kept above 3 so the report visibly spans multiple industries, not just
  whichever single sector is most volatile that week), each with a live
  price/change%/RSI/trend snapshot — concrete named candidates, not just a
  sector label.
- **Indices Snapshot**: S&P 500, Nasdaq, Dow, Russell 2000.
- **Crypto Snapshot**: BTC/ETH/SOL — explicitly flagged as outside this
  project's backtested research (raw price/trend only).

### A7. Next Market Move — the `rocket_science` model

The most sophisticated model in this project (`processing/ml/rocket_science.py`),
built **on top of**, not instead of, everything Phases 1-5 established. It
predicts, for **all 13 sectors** spanning the full US equity map (technology,
semiconductors, software/internet, financials, healthcare, industrials,
energy, consumer discretionary/staples, utilities, materials, communication
services, real estate — never just semiconductors), a **calibrated
probability** of landing in the top-3-of-13 by 20-trading-day forward
return:

- A gradient-boosting + logistic-regression calibrated ensemble, trained on
  the same sector features plus explicit interaction terms (volatility ×
  momentum, breadth × volatility, cross-sectional vol/momentum rank) —
  genuinely more sophisticated math than Phase 4's rule-based ranking.
- Evaluated with the same purged, embargoed, walk-forward discipline as
  every other phase — never trained-then-reported in-sample.
- The report shows the model's own measured OOS reliability (`mean_oos_auc`,
  `mean_oos_brier` vs. a coin-flip baseline, and a `PASS`/`NO_DEMONSTRATED_EDGE`
  status) directly alongside the 13-sector probability table, and is
  explicitly labeled an **experimental extension** — it has not yet been
  through the sector-neutral/adversarial/non-overlapping/point-in-time-
  universe stress tests that the core Phase 4/5 finding has.
- This is *not* a claim that the market will rise or fall. It is the honest
  ceiling of what this system's research supports predicting: which sectors,
  among all 13, look relatively more likely to lead over the next 20 trading
  days.

Directly below the 13-sector table, a **Short / Underperformer Candidates**
section surfaces the 3 *lowest*-probability sectors and lists their member
tickers already showing bearish technicals (below all SMAs, or RSI14 < 40).
This is explicitly labeled **lower-confidence** than the long candidates
above it: the model was trained and validated only to predict *top*-3
sectors, never separately validated as a bottom/short-predicting model, so
a low P(top-3) is the model's least-favored output, not a positively-tested
"will underperform" signal. It compounds two unvalidated-for-this-purpose
signals (sector rank + individual technicals) and says so.

### A9. Reliability & Limitations + closing note

A final table restates the hard limits (no whole-market directional
prediction, sample-only news, survivorship-biased universe, no point-in-time
fundamentals), and the report closes with a one-line note that TOP LONG/
SHORT CANDIDATES at the top are the actual output of the run and everything
below them is supporting detail — no further interpretation step is
expected.

---

## PART B — `report TICKER`: the per-ticker report

Source: `ui/report.py:generate()`, built on `processing/bundle.py:build()`.

### B1. Build the bundle (Layer 1 + Layer 2)

`bundle.build(ticker)` is the single place every data source gets fetched
and reduced to a compact dict — nothing raw (full API payloads, article
bodies) crosses this boundary:

1. **Price/technicals** — `storage/cache.get_ohlcv()` (5y daily bars, DuckDB
   cache, 12h TTL) → `processing/indicators.py:snapshot()` computes RSI14,
   MACD, SMA20/50/200, ATR14, volume-vs-30d-average, and a trend label.
2. **News** — same pipeline as Part A4, but scoped to this one ticker
   (company-name-aware query, up to 3 articles get full-text scraping).
3. **Fundamentals** — `data/fundamentals.py`: margins, growth, valuation
   multiples, dividend yield, short ratio, analyst targets/ratings (labeled
   *external sentiment*, never cited as evidence), EPS estimate dispersion
   and revision history, insider transactions, next-earnings date.
4. **Polymarket** macro context (same as Part A3's group 4).
5. **Baseline Score** — `processing/scoring.py:compute_scorecard()`: a fixed
   0-100 formula (Fundamental 40% / Technical 30% / Valuation 20% /
   Sentiment 10%), explicitly labeled a transparent heuristic, **not**
   validated as predictive. Also computes separate `risk_flags`
   (cyclicality/momentum-extension/volatility risk — a different question
   than the score) and a `company_quality` vs. `entry_quality` split ("is
   this a good business" vs. "is now a good entry," which can and do
   disagree).

### B2. Render the sections

`ui/report.py` walks the bundle and formats each section as Markdown:

- **Price & Technicals** — RSI, MACD, SMAs, Bollinger Bands + %B,
  Stochastic %K/%D, ADX(14), ROC(10), ATR, volume-vs-30d-average, and a
  **technical composite (0-100)** blending five of those into one
  descriptive read of current technical state (not a forecast) — +
  **Relative Strength** vs. SPY and the ticker's sector/industry ETF (is
  this move stock-specific or is the sector moving).
- **Fundamentals**, **Cash Flow & Balance Sheet**, **Valuation Multiples**,
  **Ownership & Positioning** (+ Insider Transactions).
- **Valuation Sensitivity** — the trailing-to-forward EPS gap stated as
  plain arithmetic (never "the market is pricing in X% growth," a claim
  the number doesn't support), estimate dispersion, a bear/base/bull
  scenario grid with every assumption labeled as an assumption, and
  **Thesis Dependencies** — plain-English conditions that must hold for the
  current valuation, plus separate fundamental-trigger invalidation
  conditions (distinct from the price-based stop in Trade Levels below).
- **News**, **Macro Context** (same three groups as Part A3).
- **Baseline Score** + **Risk/Fragility Layer** + **Investment Thesis vs.
  Trade Setup**.
- **Historical Performance** — trailing 1D-5Y returns (real price data, not
  a strategy result).
- **Strategy Screen** — `processing/backtest.py`: 6 published technical
  systems (golden cross, RSI mean-reversion, etc.) backtested and ranked by
  Sharpe, with an explicit **multiple-testing warning** (the top-of-6 result
  is a maximum over trials, not a single a-priori test), plus an
  out-of-sample check, a regime-window test, and a Monte Carlo bootstrap on
  the winning strategy's actual trade sequence.
- **Trade Levels** — hypothetical LONG **and SHORT** entry/stop/TP1/TP2 from
  ATR (`processing/scoring.py:trade_levels(direction=...)`), with an
  earnings-proximity warning (a stop assumes continuous price movement,
  which an earnings gap can violate — a bigger risk for the short side,
  where a gap moves against you). The short side additionally discloses
  costs it does NOT model (borrow fees, unbounded loss risk, squeeze risk).
  Both directions are explicitly labeled "not a recommendation" — showing a
  short setup is not evidence shorting is a good idea, any more than the
  long setup is evidence for going long.
- **ML Prediction Engine** — `processing/ml/predict.py`: a calibrated
  logistic-regression + gradient-boosting ensemble trained across a
  103-ticker universe (Phase 1/2 research). Its own out-of-sample testing
  found **no measurable directional edge** (AUC ≈ 0.50), so it always
  reports `reliability: LOW` and `recommended_gate: NO TRADE` — the system
  correctly refusing to manufacture confidence it doesn't have, not a bug.
- **Data Quality / Model Reliability** — a final inventory of what above is
  checkable fact vs. a reproducible-but-unvalidated formula vs. genuinely
  untested.

### B3. Optional flags

- `--peers TICKER...` adds a peer-relative valuation comparison
  (`processing/compare.py`) instead of the flat sector-agnostic P/E
  heuristic.
- `--account-size` / `--risk-pct` compute a real Position Sizing table
  (`processing/scoring.py:position_size()`) — floored share count so actual
  risk never exceeds the target.
- `--refresh` forces a re-fetch of cached price/news data instead of
  serving the 12h/6h cache.

### B4. Opens with its own analyst prompt

Like Part A, `ui/report.py`'s `SYSTEM_BLOCK` sits at the top and tells the
AI the ground rules (report scores verbatim, never recompute them; risk
flags answer a *different* question than the score; analyst targets are
color, not evidence; the price is a daily close, not real-time; etc.) and
the exact response structure to use (Decision / Confidence breakdown / Why
/ Why not / Scenarios / Key uncertainty / Suggested action / Position
sizing / Invalidation / Backtest summary).

---

## The one invariant across both

**Nothing in `ui/` ever invents a number.** Every value is either fetched
from a real API (yfinance, FRED, Polymarket, NewsAPI) or computed by a
named, documented formula in `processing/`. Where a number is genuinely
uncertain, unvalidated, or stale, the report says so in the same breath as
the number — never as a disclaimer bolted on separately at the end.
