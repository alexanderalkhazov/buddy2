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
You are acting as a trading research analyst. Everything in the DATA section below \
was fetched by real market-data APIs and computed by deterministic code — no AI wrote \
any of these numbers. Your job: reason from this evidence to a specific next-trade \
decision for me, or explicitly conclude NO TRADE if the evidence doesn't justify one.

Ground rules:
1. No model in this system predicts market direction (Phase 1/2 research: AUC ≈ 0.50, \
no demonstrated OOS edge). Do NOT claim or imply the market will go up or down. Reason \
qualitatively from regime/macro/breadth instead, and label it as reasoning, not a fact.
2. The Sector Opportunity Ranking IS grounded in real out-of-sample research (Phase 4: \
103-ticker/13-sector universe, walk-forward validated, survives sector-neutral and cost \
tests) — treat it as real but modest evidence (~1-2% excess return per 20D holding period \
in backtest), not a strong signal. The "high_vol/down momentum" flag marks the historically \
strongest bucket found — cite it as that, not as a guarantee.
3. The Market News Digest is a 5-ticker, 48-hour SAMPLE, not comprehensive coverage — don't \
treat the absence of a topic in it as evidence that nothing relevant happened.
4. FRED macro indicators are lagged (weeks to months) and revised — never call them "current."
5. This report has NO ticker-specific entry/stop/target/position-sizing for ANY instrument — \
name specific candidates below, but tell me to run `python main.py report TICKER` on each named \
candidate before acting, since only that report computes real trade levels for it.
6. Concluding NO TRADE (overall, or for any one asset class) is a legitimate, often correct \
answer when the evidence doesn't line up — do not force a recommendation to seem useful. You are \
allowed to say "no equity trade, but crypto/indices look more interesting" or the reverse.
7. You have data on stocks (via the Sector Opportunity Ranking and each leading sector's member \
tickers below), broad indices (S&P 500 / Nasdaq / Dow / Russell 2000), and major crypto (BTC/ETH/ \
SOL) — consider all of them, not just equities. Crypto trades 24/7 and is NOT covered by this \
system's sector/regime research at all — treat any crypto view as pure price/trend/momentum \
reasoning from the snapshot below, with LOWER confidence than the equity-sector reasoning, and \
say so explicitly.
8. When you name a specific candidate (a stock ticker, an index, or a crypto asset), ground it in \
a concrete data point from above (e.g. "MU: RSI 49, in the semiconductors sector which ranks #1 \
by volatility with negative momentum — the historically strongest bucket") — never name a ticker \
with no cited evidence from this report.

Reason in this order, then answer in this structure:
**Market Read** — regime + macro + news, do they agree or conflict?
**Evidence For (a trade)** — the strongest concrete points, cited from the data above.
**Evidence Against** — the strongest concrete points against, cited from the data above.
**Sector Focus** — which sector(s), if any, the Sector Opportunity Ranking evidence supports, or "NO SECTOR EDGE" if none stand out.
**Recommended Candidates** — specific named instruments across stocks / indices / crypto (use the Sector Leaders, Indices Snapshot, and Crypto Snapshot sections below), each with one line of cited reasoning and a confidence level — or "NO CANDIDATES" for any category where the evidence doesn't support one.
**Confidence** — how strong is this evidence, honestly (LOW/MEDIUM/HIGH per asset class), given point 1, 2, and 7 above.
**What Would Change My Mind** — the specific data that would flip this conclusion.
**Next Step** — the exact `report TICKER` commands to run for each recommended candidate.
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
        "This is an index-trend + VIX filter, not a full breadth/credit/rates regime model — "
        "the component table below shows exactly which SMAs each index is above/below so the "
        "'mixed' label isn't a black box. No market-breadth (% of stocks above their own SMA, "
        "advance/decline, equal-weight vs. cap-weight), credit-spread, or VIX-term-structure data "
        "is in this system — a real gap flagged rather than glossed over."
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
        lines.append("\n### Official Economic Indicators (FRED — lagged, subject to revision)")
        rows = []
        for series, info in econ.items():
            if isinstance(info, dict) and "value" in info:
                rows.append((info.get("label", series), f"{_fmt(info.get('value'))} {info.get('unit', '')} (as of {info.get('as_of', 'n/a')})"))
        if rows:
            lines += _kv_table(rows)
    elif econ.get("error"):
        lines.append(f"Error: {econ['error']}")

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
        "Ranked by 20D realized volatility — the ONE signal across this project's Phase 1-4 research "
        "that survived broad-universe, sector-neutral, adversarial, and out-of-sample testing (see "
        "storage/models/phase4_report.json). Historically: the single HIGHEST-volatility sector, "
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
        "\n---\nNow respond using the structure requested at the top of this report "
        "(Market Read / Evidence For / Evidence Against / Sector Focus / Recommended Next "
        "Action / Confidence / What Would Change My Mind)."
    )

    return "\n".join(lines)
