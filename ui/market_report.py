"""Zero-LLM market-wide report generator — the same design as ui/report.py
(Layer 1 fetched facts + Layer 2 deterministic calculations, Layer 3
interpretation left for you to run through any chatbot) but scoped to the
whole market instead of one ticker: regime, macro, broad news, and a sector
opportunity ranking grounded in this project's own validated research
(processing/ml Phase 3/4 — see the Sector Opportunity section's caveats).

This does NOT predict market direction. No model in this codebase has
demonstrated that ability (see Phase 1/2 ML reports). What IS grounded in
actual out-of-sample research is sector-level volatility rotation — that
finding is surfaced here as a ranked, caveated snapshot, not a forecast.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from data import macro as macro_mod_data
from data import news
from processing import macro as macro_mod
from processing import news_proc, regime
from processing.indicators import snapshot as price_snapshot
from processing.ml.features import FEATURE_COLUMNS, compute_features
from processing.ml.sector_index import build_sector_index
from processing.ml.universe import BENCHMARK, EXPANDED_UNIVERSE_V2
from storage.cache import get_ohlcv

# Bellwether tickers used as a stand-in for "market news" — no single feed
# covers "everything happening in the market," so this blends broad-index
# proxies with the handful of mega-caps whose headlines most often move the
# tape. Documented, not claimed to be exhaustive.
NEWS_BELLWETHERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

INDICES = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^DJI": "Dow Jones Industrial Average", "^RUT": "Russell 2000 (small-cap)"}
CRYPTO = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana"}
N_SECTOR_LEADERS = 3  # how many top-ranked sectors get their member tickers listed as named candidates

SYSTEM_BLOCK = """\
# MARKET CONDITION REPORT — HARDENED ANALYST PROMPT

You are acting as a trading research analyst.

Everything inside the DATA section below was obtained from external market-data sources \
or calculated by deterministic code. Treat those values as the authoritative inputs for \
this report.

Your job is NOT to predict market direction.

Your job is to determine whether the currently available evidence supports:

- TRADE
- WAIT
- NO TRADE

and, when appropriate, identify a small number of candidates that deserve ticker-specific \
investigation.

The report must never manufacture certainty, invent missing information, or turn \
qualitative reasoning into a factual claim.

---
## 1. CORE PRINCIPLES

### 1.1 No directional prediction

This system has not demonstrated a reliable out-of-sample model for predicting broad \
market direction. Phase 1/2 directional ML research produced approximately AUC ≈ 0.50 and \
therefore provides no demonstrated OOS directional edge.

Do NOT: predict that the market will rise or fall; assign unsupported directional \
probabilities; describe qualitative reasoning as a forecast; imply that a current regime \
guarantees a future return; convert historical correlations into guaranteed future outcomes.

You may say: "the current evidence is supportive of risk-taking"; "the evidence is \
defensive"; "the setup is mixed"; "risk/reward appears less attractive"; "this creates a \
watchlist opportunity"; "the evidence is insufficient to justify a trade." These are \
interpretations of the evidence, not predictions.

---
## 2. DECISION FRAMEWORK

Use this sequence exactly: DATA QUALITY → MARKET ENVIRONMENT → EVIDENCE BALANCE → \
VALIDATED RESEARCH EDGE → CANDIDATE FILTER → CONFLICT CHECK → TRADE / WAIT / NO TRADE.

Do not skip stages. A favorable market regime does NOT automatically create a trade. A \
favorable sector ranking does NOT automatically create a trade. A strong ticker does NOT \
automatically create a trade. A news headline does NOT automatically create a trade. The \
final decision must emerge from the complete evidence set.

---
## 3. DATA QUALITY GATE — PERFORM THIS FIRST

**Freshness**: determine whether each major input is current/live, ~1 trading day old, \
several days old, weeks/months old, or otherwise stale. Daily market prices and yields are \
daily-close data unless explicitly marked intraday/live. Monthly/quarterly economic \
indicators are inherently lagged and revised.

**Timestamp alignment**: do not compare indicators as though observed at the same time if \
their as_of dates differ materially — especially VIX term structure, Treasury yields, \
credit spreads, economic releases, prediction-market data, news, market prices.

**Missing / UNKNOWN values**: treat UNKNOWN, missing, stale, malformed, or mismatched data \
as unknown. Do NOT guess the missing value, infer the state from another indicator, \
substitute a nearby value, or silently ignore the missingness. If an indicator is unusable, \
explicitly say so.

**Internal consistency**: look for obvious inconsistencies — different maturities \
incorrectly compared as identical, stale VIX legs, impossible/malformed values, conflicting \
timestamps, derived values differing materially from official cross-checks. Do not attempt \
to "repair" the data yourself; state the inconsistency and reduce confidence accordingly.

---
## 4. EVIDENCE HIERARCHY

**Tier 1 — Recent deterministic market data with known timestamps** (highest priority): \
index prices, SMA relationships, RSI, Treasury yields, yield spreads, credit spreads, VIX, \
breadth proxies, commodity prices, dollar, sector momentum/volatility, ticker-level \
technical data.

**Tier 2 — Historical empirical research**: Phase 4 sector opportunity research, \
walk-forward results, out-of-sample testing, cost-adjusted backtests. Influences the \
conclusion only within its stated limitations.

**Tier 3 — Contextual information**: economic releases, news headlines, prediction \
markets, qualitative macro context. Context, not automatically predictive signals.

**Tier 4 — Analyst interpretation**: your own reasoning is the lowest-level evidence. \
Never present your interpretation as if it were independently validated empirical evidence. \
Never allow Tier 3 or Tier 4 reasoning to silently override contradictory Tier 1 data.

Do not count highly correlated indicators as independent evidence — e.g. SPY trend, QQQ \
trend, and RSI moving together during the same rally are one piece of information viewed \
three ways, not three confirmations; several Tier 1 indicators agreeing because they are \
mechanically linked (same underlying index, same time window) does not multiply the weight \
of the evidence. Correlation or simultaneous movement between two indicators does not \
establish that one causes or explains the other — state that two things moved together, \
not that one is driving the other, unless the supplied research has specifically \
demonstrated that link.

---
## 5. CURRENT MARKET ENVIRONMENT

The Market Regime section is primarily an index-trend + volatility context, but additional \
breadth, credit, rates, and volatility-term-structure data may also be available elsewhere \
in the report — do NOT claim these datasets are unavailable when they are explicitly present.

**Index trend**: SPY, QQQ, major supplied indices, 20D/50D/200D relationships, RSI. Do not \
turn "above the 200 SMA" into a prediction.

**Volatility**: VIX, VIX9D, VIX3M, VVIX. When term-structure dates are materially mismatched \
or stale, classify term structure as UNKNOWN rather than inferring contango/backwardation.

**Breadth / participation**: RSP vs SPY, IWM vs SPY, other supplied participation metrics. \
Interpret negative equal-weight/cap-weight or small-cap/large-cap spreads as evidence of \
weaker participation, not proof the market must fall. These are ETF-return proxies, not a \
complete advance/decline or percentage-above-SMA breadth model.

**Credit**: HY OAS, IG OAS, their recent changes. A widening HY spread is evidence of \
increasing credit stress/risk aversion. Do NOT claim a spread level predicts near-term \
equity returns unless this system has separately demonstrated that relationship OOS.

**Rates**: 2Y, 10Y, 13-week, 30Y, supplied yield-curve spreads. Distinguish level, recent \
change, and curve shape. Do not treat curve inversion/un-inversion as a short-term timing \
signal unless explicitly supported by validated research in this system.

**Commodities / dollar**: oil, gold, dollar — contextual evidence only. Do not invent \
causal explanations for a move unless supported by supplied evidence.

---
## 6. LEVEL VS. CHANGE

Distinguish current level, recent direction/change, and historical context (if provided). \
Do not interpret an absolute value as automatically bullish or bearish — "HY OAS = X" and \
"HY OAS increased by Y" are separate pieces of information. When historical \
percentile/range information is not supplied, do NOT invent it.

---
## 7. FRED / ECONOMIC DATA

CPI, unemployment, GDP are lagged economic observations — never call them "current market \
conditions." Use them as structural macro context, not short-term market timing signals. \
Daily FRED series (Treasury spreads, Treasury yields, credit spreads) can be used as more \
current market-based evidence, subject to their stated as_of date.

---
## 8. PREDICTION MARKETS

Treat Polymarket values as ALTERNATIVE MARKET EXPECTATION DATA, not official economic data \
and not proven forecasts. Do not treat 0.00/0.03/0.90 etc. as objective probabilities of \
reality. Use only as supplementary evidence about market participants' current \
expectations. Do not allow prediction-market data to override official market/economic \
data without explicit justification.

---
## 9. NEWS

The Market News Digest is a MARKET NEWS SAMPLE, not comprehensive market news — say so. \
Absence of a topic from this sample is NOT evidence the topic doesn't exist. Headline-only \
information is weaker evidence than verified underlying market data. Do not infer an \
event's fundamental importance beyond what the supplied headline supports. Do not let one \
sensational headline dominate the entire market conclusion.

---
## 10. PHASE 4 SECTOR RESEARCH

The Phase 4 sector research is the strongest empirical pattern identified in this \
project's Phase 1-4 research that remained positive in the reported out-of-sample, \
sector-neutral, adversarial, and transaction-cost tests. Because the historical universe \
has survivorship bias, treat this as preliminary empirical evidence rather than a fully \
validated production trading edge. The reported finding is approximately ~1-2% excess \
return per 20D holding period under the stated research methodology — a historical \
tendency, not a forecast. The high-volatility/negative-momentum bucket is "the \
historically strongest bucket found in this research," NOT "the sector most likely to \
rise," "a guaranteed winner," "a buy signal," or "a proven alpha source."

**Mandatory limitation**: the sector universe has survivorship bias (current liquid \
large-cap names, not point-in-time historical constituents) — acknowledge this explicitly \
when discussing the Phase 4 edge, don't describe it as fully validated or production-ready, \
treat it as preliminary empirical evidence, and reduce confidence when relying heavily on \
it. Do not claim statistical significance, robustness beyond reported tests, or invent \
confidence intervals, p-values, Sharpe ratios, drawdowns, or sample counts not supplied.

---
## 11. SECTOR OPPORTUNITY ≠ AUTOMATIC TRADE

A sector ranked highly does NOT automatically justify buying one of its members. Required: \
validated sector evidence + current sector condition + instrument-level evidence + \
acceptable evidence quality = a candidate worthy of further investigation. A sector ranking \
alone is insufficient.

---
## 12. CANDIDATE FILTER

Name a stock/index/crypto asset only if the DATA contains a concrete supporting fact about \
that instrument (trend alignment, RSI, momentum, relative strength, sector membership, \
price behavior, or other explicitly supplied instrument-level evidence). Never name a \
ticker solely because it's famous, large, in a high-ranked sector, in the news, "high \
quality," or a personal preference. Every candidate needs at least one explicit \
data-grounded reason; prefer candidates where multiple independent pieces of evidence agree.

---
## 13. CANDIDATE DIVERSIFICATION

Consider equities, indices, and crypto — do not force candidates from every category; \
write NO CANDIDATE where evidence is insufficient. Crypto is fundamentally different from \
the equity research system: the Phase 1-4 sector research does NOT apply to crypto. Base \
crypto conclusions only on supplied price/trend/RSI/momentum. Crypto confidence should \
normally be lower than confidence from a validated equity-sector signal given comparable \
evidence. Do not invent crypto fundamentals, on-chain conditions, funding rates, ETF \
flows, or derivatives information not supplied.

---
## 14. MARKET REGIME DOES NOT EQUAL TRADE SIGNAL

A strong regime doesn't guarantee a long; a weak regime doesn't guarantee a short; a mixed \
regime doesn't automatically mean avoid everything. Determine whether the environment \
supports risk-taking, discourages it, or provides insufficient information — context for \
evaluating candidates, not a standalone directional prediction.

---
## 15. CONFLICT ANALYSIS

Evaluate trend, volatility, breadth, credit, rates, sector research, instrument-level \
data, and news/context for conflicts. Classify overall conflict as LOW (most independent \
evidence agrees), MODERATE (several supportive signals but at least one important category \
disagrees), or HIGH (major categories strongly disagree or key information is \
missing/stale). When conflict is HIGH, prefer WAIT or NO TRADE unless unusually strong \
validated evidence justifies proceeding. Never resolve conflicts merely through intuition.

---
## 16. EVIDENCE BALANCE

**Evidence For**: concrete supplied facts supporting taking risk. **Evidence Against**: \
concrete supplied facts arguing against it. **Unknown / Unreliable**: important evidence \
that's stale, missing, mismatched, unreliable, or outside this system's research coverage. \
Do not convert UNKNOWN into either positive or negative evidence.

Absence of a validated positive signal is not automatically a validated negative signal — \
e.g. "the Phase 4 sector edge is not currently present" is a neutral/no-signal statement, \
not evidence FOR taking a bearish or defensive stance. Do not let a missing or non-firing \
signal quietly slide into the Evidence Against column. It should be represented as NO \
SIGNAL, or Unknown/Unreliable only when the underlying data itself is unavailable or \
unreliable — those are two different things: a signal that was checked and didn't fire is \
NO SIGNAL, a signal that couldn't be checked at all is Unknown/Unreliable.

---
## 17. CONFIDENCE

LOW / MEDIUM / HIGH describes the quality, consistency, and completeness of the evidence, \
NOT the probability the market will move as discussed. HIGH only when data is sufficiently \
fresh, major evidence categories agree, missing data is not material, candidate-level \
evidence exists, and the conclusion doesn't depend primarily on weak/unvalidated \
assumptions. MEDIUM when evidence is reasonably coherent but one or more important \
limitations/conflicts remain. LOW when evidence is mixed, important inputs are \
stale/missing, evidence relies heavily on qualitative interpretation, candidate evidence is \
weak, or the conclusion depends significantly on unvalidated research. Never use HIGH \
merely because many indicators exist.

HIGH confidence is PROHIBITED when any major decision-relevant input is materially stale, \
UNKNOWN, or timestamp-mismatched, even if all remaining indicators agree — e.g. VIX term \
structure reading UNKNOWN caps overall confidence below HIGH regardless of how coherent \
everything else looks.

---
## 18. UNCERTAINTY BUDGET

List the most important uncertainties limiting the conclusion, prioritizing: stale/missing \
market data; incomplete breadth/volatility information; survivorship bias; incomplete news \
coverage; lack of ticker-specific fundamentals; lack of point-in-time earnings/options \
data; lack of validated directional prediction; lack of crypto-specific research; \
daily-close data instead of intraday. Do not hide these in a generic disclaimer — explain \
which ones actually matter for THIS conclusion.

---
## 19. DECISION RULE

Exactly one of: **TRADE** (means: proceed to ticker-specific validation — evidence is \
sufficiently coherent to justify running `python main.py report TICKER` on the named \
candidate(s) next; it does NOT mean enter a position now, and never means that), **WAIT** \
(an identifiable opportunity exists but confirmation or better data quality is needed), \
**NO TRADE** (evidence doesn't justify taking risk — a successful result, not a failure to \
answer). Never force a trade simply because an actionable conclusion was requested.

---
## 20. NO FORCED DIRECTION

Valid conclusions include: no equity trade but an index watchlist opportunity; no equity \
trade but crypto is more interesting; no trade in any category. Do not force a BUY or SELL \
direction. The output represents the strength of the evidence, not the desire to produce \
an exciting recommendation.

---
## 21. TICKER-SPECIFIC RISK CONTROLS

This report does NOT contain enough information for entry price, stop loss, target, \
position size, leverage, or risk/reward calculation — do not invent any of these. For \
every surviving candidate, instruct the user to run `python main.py report TICKER` before \
acting; only that report supplies those trade-specific values.

---
## 22. WHAT WOULD CHANGE MY MIND

Specify the exact supplied variable(s) that would alter the conclusion — e.g. breadth \
improving substantially; HY OAS widening materially; SPY losing the 200D SMA; VIX moving \
into a higher regime; sector momentum reversing; candidate losing its trend structure; \
previously-UNKNOWN volatility-term-structure data becoming available; sector-research \
conditions no longer being satisfied. Do not invent numerical thresholds unless the DATA \
already provides validated thresholds. Never write generic statements like "if the market \
changes."

---
## 23. CANDIDATE RANKING

Rank surviving candidates as BEST SUPPORTED / SECONDARY / WATCH ONLY, reflecting evidence \
quality and alignment — never arbitrary numerical probabilities or personal preference.

Recommend no more than 3 candidates total unless the supplied evidence clearly justifies \
more. Prefer a smaller set of stronger candidates over a broad watchlist.

---
## 24. OUTPUT FORMAT

Return exactly this structure:

**Market Read** — current market environment; whether major evidence agrees or conflicts; \
overall conflict level; important data-quality limitations. Do NOT predict future market \
direction.

**Evidence For** — the strongest concrete pieces of supplied evidence supporting \
risk-taking or candidate investigation, each citing the specific DATA field/value used.

**Evidence Against** — the strongest concrete evidence against taking risk or against the \
candidate thesis, citing the supplied DATA.

**Sector Focus** — strongest sector evidence; whether the Phase 4 historical edge is \
currently present; important limitations; or NO SECTOR EDGE if no meaningful opportunity \
is supported. Do not equate sector ranking with a guaranteed trade.

**Recommended Candidates** — for each surviving candidate: `TICKER — [BEST SUPPORTED / \
SECONDARY / WATCH ONLY]` plus one concise evidence-based explanation using actual supplied \
data. If none pass the filter: NO CANDIDATES. Do not force candidates.

**Decision** — exactly TRADE / WAIT / NO TRADE, with why.

**Confidence** — Overall / Equity / Index / Crypto, each LOW/MEDIUM/HIGH (only for \
categories that exist in the DATA), with the main reason for each level.

**What Would Change My Mind** — the specific future data changes that would alter the \
conclusion.

**Uncertainty Budget** — the 3-5 most important limitations affecting this decision.

**Next Step** — for every recommended candidate, the exact command `python main.py report \
TICKER`. No entry/stop/target/position-size instructions here.

---
## 25. ABSOLUTE PROHIBITIONS

Never: invent data; invent historical statistics; invent probabilities; invent backtest \
results; claim an OOS edge that is not explicitly supplied; call lagged economic data \
"current"; treat stale data as current; infer UNKNOWN values; describe prediction-market \
prices as objective probabilities; describe the news sample as comprehensive; ignore \
survivorship bias; recommend a ticker with no explicit data-grounded rationale; convert \
sector membership into a trade by itself; provide ticker-specific entry/stop/target/\
position sizing; claim that a trade will make money; force a recommendation; present \
qualitative reasoning as empirical evidence; use confidence to imply a probability of \
market direction.

Your objective is maximum analytical honesty and decision usefulness, not maximum \
decisiveness.
"""


def _fmt(value, suffix: str = "", none: str = "n/a") -> str:
    return none if value is None else f"{value:,.2f}{suffix}" if isinstance(value, (int, float)) else str(value)


def _kv_table(rows: list[tuple[str, str]]) -> list[str]:
    out = ["| Field | Value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return out


def _sector_snapshot(refresh: bool = False) -> pd.DataFrame:
    """Latest point-in-time feature row per sector — NOT the stride-sampled
    historical training dataset (processing.ml.sector_dataset), which drops
    the most recent row unpredictably depending on stride alignment. This
    walks each sector's full synthetic index and takes the true last bar."""
    bench_df = get_ohlcv(BENCHMARK, force=refresh)
    rows = []
    for sector in EXPANDED_UNIVERSE_V2:
        try:
            idx_df = build_sector_index(sector, refresh=refresh)
        except Exception as exc:
            rows.append({"sector": sector, "error": str(exc)[:120]})
            continue
        feats = compute_features(idx_df, benchmark_df=bench_df)
        # A member ticker occasionally has a trailing NaN-close bar (yfinance
        # partial-day artifact) — since this synthetic index is an elementwise
        # average, one NaN member poisons the whole index's last row(s). Use
        # the most recent row with a valid close, same fix as
        # processing/indicators.py:snapshot() uses for individual tickers.
        valid = feats.dropna(subset=["close"])
        if valid.empty:
            rows.append({"sector": sector, "error": "no valid close price in sector index"})
            continue
        last = valid[FEATURE_COLUMNS].iloc[-1]
        rows.append({"sector": sector, "as_of": str(valid.index[-1].date()), **last.to_dict()})
    df = pd.DataFrame(rows)
    if "realized_vol_20d" in df.columns:
        df["vol_rank_pct_of_sectors"] = df["realized_vol_20d"].rank(pct=True) * 100
        median_vol = df["realized_vol_20d"].median()
        df["vol_bucket"] = df["realized_vol_20d"].apply(lambda v: "high_vol" if v is not None and v >= median_vol else "low_vol")
        df["momentum_bucket"] = df["ret_20d"].apply(lambda v: "up" if (v or 0) >= 0 else "down")
        df = df.sort_values("realized_vol_20d", ascending=False)
    return df


def _market_news_digest(refresh: bool = False) -> list[dict]:
    all_items = []
    for ticker in NEWS_BELLWETHERS:
        try:
            raw = news.fetch_news(ticker, lookback_hours=48, full_text=False)
            processed = news_proc.process(raw, limit=4)
            for item in processed:
                item["_source_query"] = ticker
            all_items.extend(processed)
        except Exception as exc:
            all_items.append({"title": f"[{ticker} news fetch failed: {str(exc)[:80]}]", "sentiment": "unknown", "summary": "", "_source_query": ticker})

    seen_titles, deduped = set(), []
    for item in all_items:
        key = item.get("title", "")[:80].lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        deduped.append(item)

    def _sort_key(it):
        return it.get("published_at") or ""

    return sorted(deduped, key=_sort_key, reverse=True)[:20]


def generate(refresh: bool = False) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append(f"# Market Condition Report — generated {now}")
    lines.append(SYSTEM_BLOCK)
    lines.append("\n---\n# DATA")

    # ---- Market Regime -------------------------------------------------
    lines.append("\n## Market Regime")
    lines.append(
        "This section alone is only an index-trend + VIX filter — the component table below "
        "shows exactly which SMAs each index is above/below so the 'mixed' label isn't a black "
        "box. It is NOT a full regime picture by itself: real market-breadth (equal-weight vs. "
        "cap-weight, small-cap vs. large-cap), credit-spread, and VIX-term-structure data DO "
        "exist in this report — see the 'US Market Breadth & Volatility Term Structure' "
        "subsection under Macro Context below — and must be read together with this section, "
        "not instead of it. This system still has no true advance/decline or "
        "%-of-all-stocks-above-SMA breadth count across the full market — that specific gap is "
        "real and remains, distinct from the breadth proxies that do exist."
    )
    mr = regime.market_regime()
    for key, label in (("spy", "SPY"), ("qqq", "QQQ")):
        idx = mr.get(key, {})
        lines.append(f"\n**{label}**")
        lines += _kv_table(
            [
                ("Above 20d SMA", "YES" if idx.get("above_20sma") else "NO"),
                ("Above 50d SMA", "YES" if idx.get("above_50sma") else "NO"),
                ("Above 200d SMA", "YES" if idx.get("above_200sma") else "NO"),
                ("RSI14", _fmt(idx.get("rsi14"))),
                ("Trend bucket", idx.get("trend_bucket", "n/a")),
            ]
        )
    lines.append("\n**Summary**")
    lines += _kv_table(
        [
            ("**Overall trend (SPY & QQQ agreement required)**", f"**{mr.get('overall_trend', 'n/a')}**"),
            ("VIX level", _fmt(mr.get("vix", {}).get("last"))),
            ("VIX regime", mr.get("vix", {}).get("regime", "n/a")),
            ("**Risk label**", f"**{mr.get('risk_label', 'n/a')}**"),
        ]
    )
    if mr.get("note"):
        lines.append(f"\n*{mr['note']}*")

    # ---- Macro Context ---------------------------------------------------
    lines.append("\n## Macro Context")
    try:
        macro_ctx = macro_mod.full_macro_context()
    except Exception as exc:
        macro_ctx = {"error": str(exc)[:200]}

    lines.append(
        "\n**MACRO FRESHNESS** — this data is NOT all the same age. Read the numbers below "
        "with these gaps in mind, don't treat them as one equally-current snapshot:"
    )
    _today = datetime.now(timezone.utc).date()
    _freshness_rows = []
    _mp = macro_ctx.get("market_proxies", {})
    _proxy_dates = [v.get("as_of") for v in _mp.values() if isinstance(v, dict) and v.get("as_of")]
    if _proxy_dates:
        _age = (_today - pd.Timestamp(max(_proxy_dates)).date()).days
        _freshness_rows.append(("Market prices / yields / dollar / commodities", f"{max(_proxy_dates)} ({_age}d old — daily close, not intraday)"))
    _econ = macro_ctx.get("economic_indicators", {})
    _econ_dates = [v.get("as_of") for v in _econ.values() if isinstance(v, dict) and v.get("as_of")]
    if _econ_dates:
        _oldest = min(_econ_dates)
        _newest = max(_econ_dates)
        _age = (_today - pd.Timestamp(_oldest).date()).days
        _freshness_rows.append(("Economic releases (FRED)", f"{_oldest} to {_newest} — oldest series is {_age}d old (monthly/quarterly cadence, expect this)"))
    _freshness_rows.append(("News digest", "last 48h lookback window"))
    _freshness_rows.append(("Prediction markets", "queried live just now — no historical timestamp exposed by this data source"))
    lines += _kv_table(_freshness_rows)

    proxies = macro_ctx.get("market_proxies", {})
    if proxies:
        lines.append("\n### Market-Based Proxies (daily-close pricing)")
        rows = []
        for k, v in proxies.items():
            if k in ("note", "yield_curve_note"):
                continue
            if isinstance(v, dict):
                rows.append((v.get("label", k), f"{_fmt(v.get('last'))} ({_fmt(v.get('change_pct'))}%)" if "error" not in v else f"error: {v['error']}"))
            elif k == "yield_curve_10y_minus_13w_pct_points":
                rows.append(("Yield curve (10Y − 13W, pct points)", _fmt(v)))
        lines += _kv_table(rows)
        if proxies.get("note"):
            lines.append(f"\n*{proxies['note']}*")
        if proxies.get("yield_curve_note"):
            lines.append(f"\n*{proxies['yield_curve_note']}*")

    econ = macro_ctx.get("economic_indicators", {})
    if econ and "error" not in econ:
        lagged = [(k, v) for k, v in econ.items() if isinstance(v, dict) and "value" in v and v.get("cadence") in ("monthly", "quarterly")]
        daily = [(k, v) for k, v in econ.items() if isinstance(v, dict) and "value" in v and v.get("cadence") == "daily"]

        if lagged:
            lines.append("\n### Official Economic Indicators (FRED — monthly/quarterly, genuinely lagged)")
            lines += _kv_table([(v.get("label", k), f"{_fmt(v.get('value'))} {v.get('unit', '')} (as of {v.get('as_of', 'n/a')})") for k, v in lagged])
        if daily:
            lines.append("\n### Official Daily Rate & Credit Indicators (FRED — updates daily, NOT lagged like the series above)")
            lines.append(
                "These update with roughly a 1-day lag, unlike CPI/GDP/unemployment above — a genuinely "
                "current read on rates and corporate credit stress, from FRED's own official series."
            )
            lines += _kv_table([(v.get("label", k), f"{_fmt(v.get('value'))} {v.get('unit', '')} (change {_fmt(v.get('change'))}, as of {v.get('as_of', 'n/a')})") for k, v in daily])
    elif econ.get("error"):
        lines.append(f"Error: {econ['error']}")

    xcheck = macro_ctx.get("yield_curve_cross_check")
    if xcheck:
        flag_style = "**MISMATCH**" if "MISMATCH" in xcheck["flag"] else xcheck["flag"]
        lines.append(
            f"\n*Cross-check: our own derived 10Y−13W spread ({_fmt(xcheck['our_derived_10y_minus_13w'])}) vs. "
            f"FRED's official 10Y−3M spread ({_fmt(xcheck['fred_official_10y_minus_3m'])}) — different "
            f"maturities so an exact match isn't expected, but a large gap would flag a calculation bug. "
            f"Difference: {_fmt(xcheck['abs_difference_pct_points'])}pp — {flag_style}.*"
        )

    breadth = macro_ctx.get("market_breadth", {})
    if breadth and "error" not in breadth:
        lines.append("\n### US Market Breadth & Volatility Term Structure")
        lines.append(
            "Is the advance broad or narrow, and is the options market pricing near-term calm or stress? "
            "Neither question is answered by the Market Regime section above (index trend + VIX level only)."
        )
        lines += _kv_table(
            [
                ("VIX9D / VIX / VIX3M", f"{_fmt(breadth.get('^VIX9D', {}).get('last'))} / {_fmt(breadth.get('^VIX', {}).get('last'))} / {_fmt(breadth.get('^VIX3M', {}).get('last'))}"),
                ("**VIX term structure**", f"**{breadth.get('vix_term_structure_state', 'n/a')}**"),
                ("VVIX (vol-of-vol)", _fmt(breadth.get("^VVIX", {}).get("last"))),
            ]
        )
        ew = breadth.get("equal_weight_vs_cap_weight", {})
        sc = breadth.get("small_cap_vs_large_cap", {})
        lines.append("\n**Participation breadth:**")
        lines += _kv_table(
            [
                ("Equal-weight (RSP) vs. cap-weight (SPY), 20D", f"{_fmt(ew.get('spread_pct_points'))}pp"),
                ("Small-cap (IWM) vs. large-cap (SPY), 20D", f"{_fmt(sc.get('spread_pct_points'))}pp"),
            ]
        )
        lines.append(f"\n*{ew.get('note', '')}*")
        lines.append(f"\n*{sc.get('note', '')}*")
        if breadth.get("note"):
            lines.append(f"\n*{breadth['note']}*")
    elif breadth.get("error"):
        lines.append(f"\nMarket breadth unavailable this run: {breadth['error']}")

    poly = macro_ctx.get("prediction_markets")
    lines.append("\n### Prediction Markets (Polymarket)")
    if poly:
        lines.append("| Market | Odds | Volume |")
        lines.append("|---|---|---|")
        for m in poly:
            lines.append(f"| {m.get('market', 'n/a')} | {_fmt(m.get('odds'))} | {_fmt(m.get('volume'))} |")
    else:
        lines.append("No related markets found (or the Polymarket API was unreachable this run).")

    # ---- Market News Digest ----------------------------------------------
    lines.append("\n## Market News Digest")
    lines.append(
        f"Broad-market headlines from the last 48h, sourced from {', '.join(NEWS_BELLWETHERS)} "
        "(index ETFs + mega-cap bellwethers) as a practical stand-in for \"market news\" — no single "
        "feed covers every market-moving event, so treat this as a sample, not exhaustive coverage."
    )
    news_items = _market_news_digest(refresh=refresh)
    if news_items:
        overall_sentiment = news_proc.aggregate_sentiment(news_items)
        lines.append(f"\n**Aggregate sentiment across this digest: {overall_sentiment}**\n")
        for item in news_items:
            lines.append(f"- [{item.get('_source_query', '?')}] {item.get('summary') or item.get('title', '')}")
    else:
        lines.append("No news items retrieved this run.")

    # ---- Sector Opportunity Ranking ---------------------------------------
    lines.append("\n## Sector Opportunity Ranking")
    lines.append(
        "Ranked by 20D realized volatility — the strongest empirical pattern identified in this "
        "project's Phase 1-4 research that remained positive across broad-universe, sector-neutral, "
        "adversarial, and out-of-sample testing (see storage/models/phase4_report.json). Because the "
        "historical universe has survivorship bias, treat this as preliminary empirical evidence, not "
        "a fully validated production edge. Historically: the single HIGHEST-volatility sector, "
        "especially when its own 20D momentum is NEGATIVE (a stressed/sold-off sector, not a rallying "
        "one), showed the strongest subsequent 20D returns in walk-forward testing — not a guarantee, "
        "an out-of-sample-tested tendency with a modest, cost-surviving effect size (~1-2% excess vs. "
        "an equal-weight sector benchmark per 20D holding period in Phase 4's dev/holdout testing)."
    )
    snap = _sector_snapshot(refresh=refresh)
    if "error" in snap.columns and snap["realized_vol_20d"].isna().all():
        lines.append("Sector snapshot unavailable this run.")
    else:
        lines.append("\n| Sector | 20D Realized Vol | Vol Rank | 20D Momentum | Vol/Momentum Bucket | RSI14 |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in snap.iterrows():
            if pd.isna(r.get("realized_vol_20d")):
                continue
            bucket = f"{r.get('vol_bucket', 'n/a')} / {r.get('momentum_bucket', 'n/a')}"
            flag = " **← historically strongest bucket**" if r.get("vol_bucket") == "high_vol" and r.get("momentum_bucket") == "down" else ""
            lines.append(
                f"| {r['sector'].replace('_', ' ').title()} | {_fmt(r.get('realized_vol_20d'))}% | "
                f"{_fmt(r.get('vol_rank_pct_of_sectors'))}%ile | {_fmt(r.get('ret_20d'))}% | {bucket}{flag} | {_fmt(r.get('rsi14'))} |"
            )

    # ---- Sector Leaders — named candidate tickers --------------------------
    lines.append(f"\n## Sector Leaders — Candidate Tickers (top {N_SECTOR_LEADERS} sectors by the ranking above)")
    lines.append(
        "Member tickers of the highest-ranked sectors, each with its own current snapshot — concrete, "
        "named candidates for the Recommended Candidates section, not just a sector label."
    )
    top_sectors = [r["sector"] for _, r in snap.iterrows() if not pd.isna(r.get("realized_vol_20d"))][:N_SECTOR_LEADERS]
    for sector in top_sectors:
        lines.append(f"\n### {sector.replace('_', ' ').title()}")
        lines.append("| Ticker | Last | Change % | RSI14 | Trend |")
        lines.append("|---|---|---|---|---|")
        for ticker in EXPANDED_UNIVERSE_V2.get(sector, []):
            try:
                s = price_snapshot(ticker, get_ohlcv(ticker, force=refresh))
                lines.append(f"| {ticker} | {_fmt(s.last)} | {_fmt(s.change_pct)}% | {_fmt(s.rsi14)} | {s.trend} |")
            except Exception as exc:
                lines.append(f"| {ticker} | error: {str(exc)[:60]} | | | |")

    # ---- Indices Snapshot ----------------------------------------------------
    lines.append("\n## Indices Snapshot")
    lines.append("| Index | Last | Change % | RSI14 | Trend |")
    lines.append("|---|---|---|---|---|")
    for ticker, label in INDICES.items():
        try:
            s = price_snapshot(ticker, get_ohlcv(ticker, force=refresh))
            lines.append(f"| {label} ({ticker}) | {_fmt(s.last)} | {_fmt(s.change_pct)}% | {_fmt(s.rsi14)} | {s.trend} |")
        except Exception as exc:
            lines.append(f"| {label} ({ticker}) | error: {str(exc)[:60]} | | | |")

    # ---- Crypto Snapshot -------------------------------------------------------
    lines.append("\n## Crypto Snapshot")
    lines.append(
        "Trades 24/7 — NOT covered by this project's sector/regime research (that's equity-only). "
        "Price/trend/momentum only; no fundamentals, no on-chain data, no derivatives/funding data."
    )
    lines.append("| Asset | Last | Change % | RSI14 | Trend |")
    lines.append("|---|---|---|---|---|")
    for ticker, label in CRYPTO.items():
        try:
            s = price_snapshot(ticker, get_ohlcv(ticker, force=refresh))
            lines.append(f"| {label} ({ticker}) | {_fmt(s.last)} | {_fmt(s.change_pct)}% | {_fmt(s.rsi14)} | {s.trend} |")
        except Exception as exc:
            lines.append(f"| {label} ({ticker}) | error: {str(exc)[:60]} | | | |")

    # ---- Reliability & Limitations -----------------------------------------
    lines.append("\n## Reliability & Limitations")
    lines += _kv_table(
        [
            ("Market-direction prediction", "NOT AVAILABLE — no model in this system has demonstrated OOS directional edge (Phase 1/2 ML: AUC ≈ 0.50)"),
            ("Sector ranking basis", "Phase 4 out-of-sample research, ~4 years of data, 103-ticker/13-sector universe — real but modest effect size, not a high-confidence signal"),
            ("News coverage", f"{len(NEWS_BELLWETHERS)} bellwether tickers, 48h lookback — a sample of market-moving headlines, not comprehensive coverage"),
            ("Survivorship bias", "Sector universe uses TODAY's liquid large-cap names per sector, not point-in-time historical constituents"),
            ("Point-in-time fundamentals/earnings/options", "NOT available from this system's data source (yfinance) — excluded from all predictive claims above"),
            ("Crypto research coverage", "NONE — the Phase 1-4 research (sector rotation, regime testing) is equity-only. Crypto snapshot is raw price/trend/RSI, no backtested edge behind it"),
            ("Indices research coverage", "Indices snapshot is raw price/trend/RSI only — same as crypto, not covered by the sector-rotation research"),
        ]
    )
    lines.append(
        "\n---\nNow respond using the exact structure specified in Section 24 of the prompt "
        "above (Market Read / Evidence For / Evidence Against / Sector Focus / Recommended "
        "Candidates / Decision / Confidence / What Would Change My Mind / Uncertainty Budget "
        "/ Next Step)."
    )

    return "\n".join(lines)
