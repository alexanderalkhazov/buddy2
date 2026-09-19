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

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data import macro as macro_mod_data
from data import news
from data.fundamentals import fetch_fundamentals
from processing import macro as macro_mod
from processing import news_proc, regime, scoring
from processing.indicators import snapshot as price_snapshot
from processing.ml.features import FEATURE_COLUMNS, compute_features
from processing.ml.registry import ARTIFACT_DIR
from processing.ml.sector_index import build_sector_index
from processing.ml.universe import BENCHMARK, EXPANDED_UNIVERSE_V2
from storage.cache import get_ohlcv

_PHASE5_REPORT_PATH = ARTIFACT_DIR / "phase5_report.json"
_TRADE_MECHANICS_REPORT_PATH = ARTIFACT_DIR / "trade_mechanics_backtest_report.json"

# Bellwether tickers used as a stand-in for "market news" — no single feed
# covers "everything happening in the market," so this blends broad-index
# proxies with the handful of mega-caps whose headlines most often move the
# tape. Documented, not claimed to be exhaustive.
NEWS_BELLWETHERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

INDICES = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^DJI": "Dow Jones Industrial Average", "^RUT": "Russell 2000 (small-cap)"}
CRYPTO = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana"}
N_SECTOR_LEADERS = 5  # how many top-ranked sectors get their member tickers listed as named candidates — kept above 3 so the report visibly spans multiple industries, not just whichever single sector is most volatile this week

# Per web research into portfolio-construction practice (FINRA/J.P. Morgan
# Private Bank concentration-risk guidance): institutional convention caps
# a single sector at roughly 20-30% of a portfolio. Scaled to a 5-name
# candidate list, that's ~1-1.5 names — MAX_CANDIDATES_PER_SECTOR=2 is a
# deliberately round, slightly looser number (not fitted against backtest
# P&L, which would defeat the point of a risk CONSTRAINT rather than an
# alpha input), chosen because without a cap the ranking has repeatedly
# surfaced top-5 lists concentrated almost entirely in one sector — 5
# correlated names dressed up as 5 diversified ideas.
MAX_CANDIDATES_PER_SECTOR = 2


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


def _indicator_reasons(row: dict, direction: str) -> list[str]:
    """Plain-English read of each indicator for THIS ticker, stated honestly —
    including when an indicator conflicts with the candidate's own direction.
    This is not a cherry-picked list of only-supportive signals; every family
    that fed technical_composite is reported here, aligned or not."""
    bullish = direction == "long"
    reasons = []

    rsi = row.get("rsi14")
    if rsi is not None:
        if 50 <= rsi <= 70:
            reasons.append(f"RSI14 {rsi:.1f} — healthy bullish momentum, not yet overbought" + ("" if bullish else " (works against a short thesis)"))
        elif rsi > 70:
            reasons.append(f"RSI14 {rsi:.1f} — overbought" + (", momentum strong but extended" if bullish else " — a genuine mean-reversion risk factor for a short"))
        elif rsi < 30:
            reasons.append(f"RSI14 {rsi:.1f} — oversold" + (" (works against a long thesis)" if bullish else ", momentum weak and extended down"))
        else:
            reasons.append(f"RSI14 {rsi:.1f} — soft/neutral momentum" + (" (a weaker point for this long)" if bullish else " (a weaker point for this short)"))

    stoch = row.get("stoch_k")
    if stoch is not None:
        if stoch >= 80:
            reasons.append(f"Stochastic %K {stoch:.1f} — strongly overbought" + (", confirms upward momentum but extended" if bullish else " — supports a mean-reversion short case"))
        elif stoch <= 20:
            reasons.append(f"Stochastic %K {stoch:.1f} — strongly oversold" + (" — supports a mean-reversion long case" if bullish else ", confirms downward momentum but extended"))

    adx, roc = row.get("adx14"), row.get("roc10")
    if adx is not None and roc is not None:
        trend_up = roc >= 0
        strength = "strong" if adx >= 25 else ("moderate" if adx >= 15 else "weak")
        agrees = trend_up == bullish
        reasons.append(f"ADX14 {adx:.1f} ({strength} trend strength), ROC10 {roc:+.1f}% ({'up' if trend_up else 'down'}) — {'confirms' if agrees else 'CONFLICTS WITH'} the {direction} direction")

    bb = row.get("bb_pctb")
    if bb is not None:
        if bb >= 0.8:
            reasons.append(f"Bollinger %B {bb:.2f} — price near/above the upper band" + (", extended but trend-confirming" if bullish else " — a mean-reversion risk for a short"))
        elif bb <= 0.2:
            reasons.append(f"Bollinger %B {bb:.2f} — price near/below the lower band" + (" — a mean-reversion risk for a long" if bullish else ", extended but trend-confirming"))

    macd = row.get("macd_hist")
    if macd is not None:
        macd_bullish = macd > 0
        reasons.append(f"MACD histogram {macd:+.2f} — {'bullish' if macd_bullish else 'bearish'} crossover state — {'confirms' if macd_bullish == bullish else 'CONFLICTS WITH'} the {direction} direction")

    cci = row.get("cci20")
    if cci is not None:
        if cci > 100:
            reasons.append(f"CCI20 {cci:.0f} — strong uptrend/overbought reading" + ("" if bullish else " — works against a short"))
        elif cci < -100:
            reasons.append(f"CCI20 {cci:.0f} — strong downtrend/oversold reading" + (" — works against a long" if bullish else ""))

    cloud = row.get("ichimoku_cloud_position")
    if cloud and cloud != "UNKNOWN":
        cloud_bullish = cloud == "above_cloud"
        cloud_bearish = cloud == "below_cloud"
        if cloud_bullish or cloud_bearish:
            agrees = cloud_bullish == bullish
            reasons.append(f"Ichimoku: price is {cloud.replace('_', ' ')} — {'confirms' if agrees else 'CONFLICTS WITH'} the {direction} direction")
        else:
            reasons.append("Ichimoku: price is in the cloud — no clear trend signal either way")

    mfi = row.get("mfi14")
    if mfi is not None:
        if mfi > 80:
            reasons.append(f"MFI14 {mfi:.1f} — strong volume-weighted overbought reading" + (", confirms upward pressure but extended" if bullish else " — supports a mean-reversion short case"))
        elif mfi < 20:
            reasons.append(f"MFI14 {mfi:.1f} — strong volume-weighted oversold reading" + (" — supports a mean-reversion long case" if bullish else ", confirms downward pressure but extended"))

    aroon_osc = row.get("aroon_oscillator")
    if aroon_osc is not None and abs(aroon_osc) >= 50:
        aroon_bullish = aroon_osc > 0
        agrees = aroon_bullish == bullish
        reasons.append(f"Aroon Oscillator {aroon_osc:+.0f} — a recent new {'high' if aroon_bullish else 'low'} (recency, not magnitude, of the extreme) — {'confirms' if agrees else 'CONFLICTS WITH'} the {direction} direction")

    keltner = row.get("keltner_pctk")
    if keltner is not None:
        if keltner >= 1.0:
            reasons.append(f"Keltner %K {keltner:.2f} — price above the ATR-based upper band" + (", confirms a real breakout (not just a std-dev outlier)" if bullish else " — an extended move against a short thesis"))
        elif keltner <= 0.0:
            reasons.append(f"Keltner %K {keltner:.2f} — price below the ATR-based lower band" + (" — an extended move against a long thesis" if bullish else ", confirms a real breakdown (not just a std-dev outlier)"))

    return reasons


def _indicator_agreement_counts(row: dict, direction: str) -> tuple[int, int]:
    """(n_confirm, n_conflict) across the same indicator families
    _indicator_reasons() renders as text — kept as a separate, structured
    function rather than parsing that function's prose, so a future wording
    change there can't silently break the conviction-verdict math here.
    Only indicators with an unambiguous directional read count; a soft/
    neutral reading (e.g. RSI between 30-50 with no clear signal) counts
    toward neither, same as _indicator_reasons() treats it as informational
    only."""
    bullish = direction == "long"
    confirm, conflict = 0, 0

    rsi = row.get("rsi14")
    if rsi is not None:
        if rsi > 70 or (50 <= rsi <= 70):
            confirm += 1 if bullish else 0
            conflict += 1 if not bullish else 0
        elif rsi < 30:
            confirm += 1 if not bullish else 0
            conflict += 1 if bullish else 0

    stoch = row.get("stoch_k")
    if stoch is not None:
        if stoch >= 80:
            confirm += 1 if bullish else 0
            conflict += 1 if not bullish else 0
        elif stoch <= 20:
            confirm += 1 if not bullish else 0
            conflict += 1 if bullish else 0

    adx, roc = row.get("adx14"), row.get("roc10")
    if adx is not None and roc is not None:
        agrees = (roc >= 0) == bullish
        confirm += 1 if agrees else 0
        conflict += 0 if agrees else 1

    bb = row.get("bb_pctb")
    if bb is not None:
        if bb >= 0.8:
            confirm += 1 if bullish else 0
            conflict += 1 if not bullish else 0
        elif bb <= 0.2:
            confirm += 1 if not bullish else 0
            conflict += 1 if bullish else 0

    macd = row.get("macd_hist")
    if macd is not None:
        agrees = (macd > 0) == bullish
        confirm += 1 if agrees else 0
        conflict += 0 if agrees else 1

    cci = row.get("cci20")
    if cci is not None:
        if cci > 100:
            confirm += 1 if bullish else 0
            conflict += 1 if not bullish else 0
        elif cci < -100:
            confirm += 1 if not bullish else 0
            conflict += 1 if bullish else 0

    cloud = row.get("ichimoku_cloud_position")
    if cloud in ("above_cloud", "below_cloud"):
        agrees = (cloud == "above_cloud") == bullish
        confirm += 1 if agrees else 0
        conflict += 0 if agrees else 1

    mfi = row.get("mfi14")
    if mfi is not None:
        if mfi > 80:
            confirm += 1 if bullish else 0
            conflict += 1 if not bullish else 0
        elif mfi < 20:
            confirm += 1 if not bullish else 0
            conflict += 1 if bullish else 0

    aroon_osc = row.get("aroon_oscillator")
    if aroon_osc is not None and abs(aroon_osc) >= 50:
        agrees = (aroon_osc > 0) == bullish
        confirm += 1 if agrees else 0
        conflict += 0 if agrees else 1

    keltner = row.get("keltner_pctk")
    if keltner is not None:
        if keltner >= 1.0:
            confirm += 1 if bullish else 0
            conflict += 1 if not bullish else 0
        elif keltner <= 0.0:
            confirm += 1 if not bullish else 0
            conflict += 1 if bullish else 0

    return confirm, conflict


def _news_for_candidate(ticker: str, direction: str, refresh: bool = False) -> dict:
    """Fetches this ticker's own real news (same pipeline as the Market News
    Digest) and reports the single most recent item plus whether its sentiment
    aligns with or conflicts with the candidate's direction — never silently
    picks the most convenient headline out of several."""
    try:
        raw = news.fetch_news(ticker, lookback_hours=72, full_text=False)
        processed = news_proc.process(raw, limit=3)
    except Exception as exc:
        return {"available": False, "error": str(exc)[:150]}
    if not processed:
        return {"available": False, "reason": "no recent news found for this ticker"}
    top = processed[0]
    sentiment = top.get("sentiment", "unknown")
    direction_bullish = direction == "long"
    sentiment_bullish = sentiment == "positive"
    sentiment_bearish = sentiment == "negative"
    if sentiment == "unknown" or sentiment == "neutral":
        alignment = "NEUTRAL — no clear directional signal from this headline"
    else:
        agrees = sentiment_bullish == direction_bullish
        alignment = ("SUPPORTS" if agrees else "CONFLICTS WITH") + f" the {direction} direction"
    return {
        "available": True,
        "title": top.get("title"),
        "source": top.get("source"),
        "published_at": top.get("published_at"),
        "url": top.get("url"),
        "sentiment": sentiment,
        "alignment": alignment,
        "n_articles_checked": len(processed),
        "is_earnings_surprise": news_proc.is_earnings_surprise_headline(top.get("title", ""), top.get("summary", "")),
    }


def _when_for_candidate(ticker: str, as_of: str, refresh: bool = False) -> dict:
    """WHEN to act — currently the one thing this system can say about timing
    with real data: how close the next earnings report is, and the severity
    tier that implies for entry risk (an earnings gap can jump past a stop
    with no fill available at that level)."""
    try:
        fund = fetch_fundamentals(ticker)
    except Exception as exc:
        return {"available": False, "error": str(exc)[:150]}
    next_earnings = fund.get("next_earnings_date")
    proximity = scoring.earnings_proximity_risk(as_of, next_earnings)
    return {"available": True, **proximity}


def _diversified_top_n(rows: list[dict], score_key: str, top_n: int, max_per_sector: int = MAX_CANDIDATES_PER_SECTOR) -> list[dict]:
    """Greedily takes the top_n highest-scoring rows, capping how many can
    come from the same sector — otherwise the ranking (correctly, since
    score is dominated by the sector-level probability) tends to surface a
    top-N list concentrated almost entirely in whichever single sector
    ranks highest that run, which is 5 correlated bets on one sector's
    outcome dressed up as 5 diversified ideas. See MAX_CANDIDATES_PER_SECTOR
    for the sourcing/rationale on the cap value.

    Falls back to relaxing the cap only if there genuinely aren't enough
    distinct sectors to fill top_n slots otherwise (flagged in the result
    so this is never silent) — the cap is a risk constraint, not a promise
    to always show 5 different sectors regardless of how few pass the
    scoring bar at all."""
    # rows is shared between the long_score and short_score calls (a ticker
    # can in principle rank highly for both) — never mutate the shared row
    # dicts in place; copy before annotating.
    ranked = sorted(rows, key=lambda r: r[score_key], reverse=True)
    selected, sector_counts, deferred = [], {}, []
    for r in ranked:
        if len(selected) >= top_n:
            break
        sector = r["sector"]
        if sector_counts.get(sector, 0) < max_per_sector:
            selected.append({**r, "_diversification_cap_relaxed": False})
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        else:
            deferred.append(r)

    if len(selected) < top_n:
        # Not enough distinct sectors cleared the cap to fill top_n — relax
        # it rather than silently returning fewer candidates than asked for.
        for r in deferred:
            if len(selected) >= top_n:
                break
            selected.append({**r, "_diversification_cap_relaxed": True})

    return selected


def _ranked_predictions(refresh: bool = False, top_n: int = 5) -> dict:
    """Direct, deterministic LONG and SHORT ticker rankings across ALL 13
    sectors' member tickers (~104 names) — no LLM, no interpretation step.
    composite score = the ticker's sector-level calibrated P(top-3) from
    sector_prediction_model, nudged by the ticker's OWN technical_composite (RSI +
    Stochastic + ADX/ROC-signed trend strength + Bollinger %B + MACD sign +
    CCI + Ichimoku cloud position — see
    processing/indicators.py:technical_composite_score, seven distinct
    indicator families, not one). Each row also carries a hypothetical
    entry/stop/TP1 (processing/scoring.py:trade_levels(), ATR-based) — NOT
    a position size, and not a claim these levels are calibrated to this
    specific ticker's own history.

    NEITHER the sector probability NOR the technical-composite nudge has
    been shown, on its own, to predict which INDIVIDUAL ticker within a
    sector outperforms its peers — only the sector-level probability is
    walk-forward validated (sector_prediction_model). The technical-composite nudge
    is a small (+/-10% of the sector probability, capped) tiebreaker among
    otherwise-similar tickers in the same sector, not a second validated
    signal — stated plainly, not hidden in the score."""
    from processing.ml.sector_prediction_model import predict_next_move

    rocket = predict_next_move(refresh=refresh)
    sector_prob = {p["sector"]: p.get("p_top3_ensemble", 0.0) for p in rocket.get("predictions_all_13_sectors", []) if "error" not in p}
    sector_edge = {p["sector"]: p for p in rocket.get("predictions_all_13_sectors", []) if "error" not in p}
    reliability = rocket.get("model_oos_reliability", {})

    rows = []
    for sector, tickers in EXPANDED_UNIVERSE_V2.items():
        p_top3 = sector_prob.get(sector)
        if p_top3 is None:
            continue
        for ticker in tickers:
            try:
                s = price_snapshot(ticker, get_ohlcv(ticker, force=refresh))
            except Exception:
                continue
            # technical_composite is 0-100, 50=neutral. Rescale to a nudge
            # capped at +/-10% of the sector probability, signed by distance
            # from neutral — bullish composite nudges long_score up and
            # short_score down, symmetric for a bearish composite.
            tc = s.technical_composite
            nudge = ((tc - 50) / 50) * 0.10 * p_top3 if tc is not None else 0.0

            long_levels = scoring.trade_levels(s.last, s.atr14, direction="long") if s.atr14 else {"error": "no ATR"}
            short_levels = scoring.trade_levels(s.last, s.atr14, direction="short") if s.atr14 else {"error": "no ATR"}

            rows.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "as_of": s.as_of,
                    "p_top3_sector": round(p_top3, 4),
                    "last": s.last,
                    "change_pct": s.change_pct,
                    "rsi14": s.rsi14,
                    "stoch_k": s.stoch_k,
                    "adx14": s.adx14,
                    "roc10": s.roc10,
                    "bb_pctb": s.bb_pctb,
                    "macd_hist": s.macd_hist,
                    "cci20": s.cci20,
                    "ichimoku_cloud_position": s.ichimoku_cloud_position,
                    "mfi14": s.mfi14,
                    "aroon_oscillator": s.aroon_oscillator,
                    "keltner_pctk": s.keltner_pctk,
                    # RISK/EXECUTION feature, deliberately never fed into
                    # technical_composite_score or long_score/short_score —
                    # a trailing stop describes how to manage a position
                    # already taken, not whether to predict the direction
                    # is right. See §33-style separation in HOW_REPORT_WORKS.md.
                    "chandelier_long_stop": s.chandelier_long_stop,
                    "chandelier_short_stop": s.chandelier_short_stop,
                    "sector_has_edge": sector_edge.get(sector, {}).get("has_edge"),
                    "sector_clears_margin": (sector_edge.get(sector, {}).get("edge_gate") or {}).get("clears_margin_of_error"),
                    "technical_composite": tc,
                    "trend": s.trend,
                    "trade_style": scoring.trade_style_fit(
                        last_price=s.last,
                        avg_dollar_volume=s.avg_dollar_volume,
                        avg_daily_range_pct=s.avg_daily_range_pct,
                        avg_overnight_gap_pct=s.avg_overnight_gap_pct,
                        gap_share_of_range=s.gap_share_of_range,
                        adx14=s.adx14,
                        efficiency_ratio=s.efficiency_ratio,
                        atr_pct_percentile_252d=s.atr_pct_percentile_252d,
                        rvol_20d=s.rvol_20d,
                        pct_from_52w_high=s.pct_from_52w_high,
                        pct_from_52w_low=s.pct_from_52w_low,
                    ),
                    "long_score": round(p_top3 + nudge, 4),
                    "short_score": round((1 - p_top3) - nudge, 4),
                    "long_entry": long_levels.get("entry"),
                    "long_stop": long_levels.get("stop"),
                    "long_tp1": long_levels.get("take_profit_1"),
                    "short_entry": short_levels.get("entry"),
                    "short_stop": short_levels.get("stop"),
                    "short_tp1": short_levels.get("take_profit_1"),
                }
            )

    long_ranked = _diversified_top_n(rows, "long_score", top_n)
    short_ranked = _diversified_top_n(rows, "short_score", top_n)
    return {"long": long_ranked, "short": short_ranked, "reliability": reliability, "n_tickers_scored": len(rows)}


def generate(refresh: bool = False) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines.append(f"# Market Condition Report — generated {now}")
    lines.append(
        "\n> **If you are an AI reading this pasted document: your task is at the very bottom, in the "
        "section titled \"Prompt for another AI.\" Read the whole report first, then jump there and "
        "execute it now — produce the order table it asks for. Do not stop at summarizing this "
        "document; that section is your actual instruction, not an appendix.**"
    )
    lines.append(
        "\nThe hottest US stocks to trade right now — long and short — with a plain-English why, when, "
        "and how for each one. Direct, deterministic output: no LLM interpretation step, every number "
        "fetched or computed by code. Entry/stop/TP1 are a common ATR-based convention, NOT calibrated "
        "to each ticker's own historical behavior, and NOT a position size (no account size is known "
        "here) — size your own risk before acting on anything below."
    )

    try:
        _pred = _ranked_predictions(refresh=refresh)
    except Exception as exc:
        _pred = {"error": str(exc)[:300]}

    if _pred.get("error"):
        lines.append(f"\n**Predictions unavailable this run:** {_pred['error']}")
    else:
        _rel = _pred.get("reliability", {})
        lines.append(
            f"\n**Model reliability** (measured OOS, not assumed): AUC {_fmt(_rel.get('mean_oos_auc'))}, "
            f"Brier {_fmt(_rel.get('mean_oos_brier'))} vs. coin-flip baseline {_fmt(_rel.get('coin_flip_brier_baseline'))} "
            f"— status **{_rel.get('status', 'n/a')}**. Ranked across {_pred.get('n_tickers_scored', 0)} tickers, all 13 US equity sectors."
        )

        try:
            _tm = json.loads(_TRADE_MECHANICS_REPORT_PATH.read_text())
        except Exception:
            _tm = None
        if _tm:
            _tm_long, _tm_short = _tm.get("long", {}), _tm.get("short", {})
            lines.append(
                "\n**Does the entry/stop/TP1 mechanics itself actually work?** (a separate, walk-forward "
                "backtest of the exact ranking+ATR-levels chain below — not sector AUC again)"
            )
            def _pct(v):
                return "n/a" if v is None else f"{v * 100:.1f}%"

            lines += _kv_table(
                [
                    (
                        "LONG: win rate (TP1 vs stop, resolved trades)",
                        f"{_pct(_tm_long.get('win_rate_of_resolved'))} "
                        f"({_tm_long.get('n_wins', 'n/a')}W/{_tm_long.get('n_losses', 'n/a')}L of {_tm_long.get('n_resolved', 'n/a')} resolved, "
                        f"{_tm_long.get('n_timeout', 'n/a')} timed out)",
                    ),
                    ("**LONG: expectancy per trade (R-multiples, before costs)**", f"**{_fmt(_tm_long.get('expectancy_r_per_trade'))}R**"),
                    (
                        "LONG: cost-adjusted expectancy (LOW/BASE/HIGH)",
                        " / ".join(f"{_fmt(v)}R" for v in (_tm_long.get('expectancy_r_per_trade_cost_adjusted') or {}).values()) or "n/a",
                    ),
                    (
                        "SHORT: win rate (TP1 vs stop, resolved trades)",
                        f"{_pct(_tm_short.get('win_rate_of_resolved'))} "
                        f"({_tm_short.get('n_wins', 'n/a')}W/{_tm_short.get('n_losses', 'n/a')}L of {_tm_short.get('n_resolved', 'n/a')} resolved, "
                        f"{_tm_short.get('n_timeout', 'n/a')} timed out)",
                    ),
                    ("**SHORT: expectancy per trade (R-multiples, before costs)**", f"**{_fmt(_tm_short.get('expectancy_r_per_trade'))}R**"),
                    (
                        "SHORT: cost-adjusted expectancy (LOW/BASE/HIGH)",
                        " / ".join(f"{_fmt(v)}R" for v in (_tm_short.get('expectancy_r_per_trade_cost_adjusted') or {}).values()) or "n/a",
                    ),
                ]
            )
            _short_r = _tm_short.get("expectancy_r_per_trade")
            _long_r = _tm_long.get("expectancy_r_per_trade")
            lines.append(
                "\n*Methodology: for each walk-forward OOS rebalance date, the top-3-predicted-sector's "
                "highest-technical-composite member ticker (LONG) / lowest (SHORT) is picked — the same "
                "logic that ranks the tables below — and its ATR entry/stop/TP1 (processing/scoring.py) is "
                "walked forward day-by-day against REAL subsequent price bars to see which was hit first. "
                "expectancy_r_per_trade = win_rate×1.5R − (1−win_rate)×1R, the R-multiples TP1/stop use by "
                "construction. Cost-adjusted expectancy subtracts a round-trip LOW/BASE/HIGH bps cost "
                "(5/15/40 bps of entry price, converted into each trade's own R-terms via its own risk-per-"
                "share) — conventional retail-equity ranges, not fitted to this data. Gap-past-stop fills "
                "and market impact are still not modeled at any tier — real expectancy is worse than even "
                "the HIGH-cost number, never better."
                + (
                    f" **The SHORT side's measured expectancy is NEGATIVE ({_fmt(_short_r)}R/trade)** — this "
                    "system's short mechanics would have LOST money on this historical sample even before "
                    "real-world costs, not merely 'unvalidated.' Weight the SHORT candidates above accordingly."
                    if isinstance(_short_r, (int, float)) and _short_r < 0
                    else ""
                )
                + (
                    f" The LONG side's measured expectancy is positive ({_fmt(_long_r)}R/trade) on this sample — "
                    "a real but small-N (few hundred trades, ~3 years) result, not a guarantee."
                    if isinstance(_long_r, (int, float)) and _long_r >= 0
                    else ""
                )
            )
        else:
            lines.append(
                "\n*Trade-mechanics backtest not yet run this session — run "
                "`python -m processing.ml.trade_mechanics_backtest` to generate it. Until then, the entry/"
                "stop/TP1 levels below are unvalidated arithmetic, not a backtested rule.*"
            )

        lines.append("\n## TOP LONG CANDIDATES")
        lines.append("| # | Ticker | Sector | P(top-3) | Entry | Stop | TP1 | RSI14 | Tech. Composite | Long Score |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(_pred["long"], 1):
            lines.append(
                f"| {i} | {r['ticker']} | {r['sector'].replace('_', ' ').title()} | {_fmt(r['p_top3_sector'])} | "
                f"{_fmt(r['long_entry'])} | {_fmt(r['long_stop'])} | {_fmt(r['long_tp1'])} | {_fmt(r['rsi14'])} | "
                f"{_fmt(r['technical_composite'])} | {_fmt(r['long_score'])} |"
            )

        lines.append("\n## TOP SHORT CANDIDATES")
        lines.append("| # | Ticker | Sector | P(top-3) | Entry | Stop | TP1 | RSI14 | Tech. Composite | Short Score |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(_pred["short"], 1):
            lines.append(
                f"| {i} | {r['ticker']} | {r['sector'].replace('_', ' ').title()} | {_fmt(r['p_top3_sector'])} | "
                f"{_fmt(r['short_entry'])} | {_fmt(r['short_stop'])} | {_fmt(r['short_tp1'])} | {_fmt(r['rsi14'])} | "
                f"{_fmt(r['technical_composite'])} | {_fmt(r['short_score'])} |"
            )

        lines.append(
            "\n*Entry/Stop/TP1 are ATR-based arithmetic (2.0x ATR stop, 1.5R first target) — a common "
            "convention, NOT calibrated to each ticker's own historical behavior, and NOT a position "
            "size (no account size is known here). long_score = sector's calibrated P(top-3-of-13 by "
            "20D return), nudged up to +/-10% by the ticker's OWN technical_composite (0-100, ten "
            "indicator families: RSI momentum, Stochastic, ADX trend-strength signed by ROC direction, "
            "Bollinger %B band position, MACD histogram sign, CCI, Ichimoku cloud position, Money Flow "
            "Index, Aroon Oscillator, Keltner Channel position — "
            "processing/indicators.py:technical_composite_score). short_score mirrors this on the "
            "bearish side. Only the sector probability is walk-forward validated; the technical-"
            "composite nudge is an unvalidated tiebreaker among tickers in the same sector, not a "
            "second tested signal. These are RANKINGS, not probabilities of profit — the underlying "
            "model's measured edge is modest (see Model reliability above)."
        )
        lines.append(
            f"\n*Diversification: each table caps at {MAX_CANDIDATES_PER_SECTOR} candidates per sector "
            "(institutional portfolio-construction convention limits a single sector to roughly 20-30% "
            "of a portfolio — this is that guidance scaled down to a 5-name list, a risk CONSTRAINT, not "
            "a fitted parameter) — without it, the ranking would correctly but unhelpfully surface 5 "
            "names from whichever single sector currently scores highest, which is 5 correlated bets on "
            "one sector's outcome, not 5 diversified ideas."
            + (
                " Not enough distinct sectors cleared the cap this run, so it was relaxed for: "
                + ", ".join(sorted({r["ticker"] for r in _pred["long"] + _pred["short"] if r.get("_diversification_cap_relaxed")}))
                + "."
                if any(r.get("_diversification_cap_relaxed") for r in _pred["long"] + _pred["short"])
                else ""
            )
        )

        try:
            _tm_expectancy = json.loads(_TRADE_MECHANICS_REPORT_PATH.read_text())
            _expectancy_by_direction = {
                d: _tm_expectancy.get(d, {}).get("expectancy_r_per_trade") for d in ("long", "short")
            }
            _mean_days_held_by_direction = {
                d: _tm_expectancy.get(d, {}).get("mean_days_held_resolved") for d in ("long", "short")
            }
        except Exception:
            _expectancy_by_direction = {"long": None, "short": None}
            _mean_days_held_by_direction = {"long": None, "short": None}

        # ---- Why / When / How — one detailed section per candidate ----------
        for label, direction, entry_key, stop_key, tp1_key in (
            ("LONG", "long", "long_entry", "long_stop", "long_tp1"),
            ("SHORT", "short", "short_entry", "short_stop", "short_tp1"),
        ):
            lines.append(f"\n## Why / When / How — {label} Candidates, Explained")
            for i, r in enumerate(_pred[direction], 1):
                lines.append(f"\n### {i}. {r['ticker']} ({r['sector'].replace('_', ' ').title()}) — {label}")

                lines.append("\n**HOW — the trade arithmetic**")
                lines += _kv_table(
                    [
                        ("Entry (latest close)", _fmt(r.get(entry_key))),
                        ("Stop-loss (fixed ATR-multiple)", _fmt(r.get(stop_key))),
                        ("Take-profit 1", _fmt(r.get(tp1_key))),
                        ("Sector", r["sector"].replace("_", " ").title()),
                    ]
                )
                chandelier_key = "chandelier_long_stop" if direction == "long" else "chandelier_short_stop"
                chandelier_val = r.get(chandelier_key)
                if chandelier_val is not None:
                    lines.append(
                        f"- *RISK/EXECUTION info, not a prediction: Chandelier Exit trailing stop = "
                        f"{_fmt(chandelier_val)} — anchors off the {'highest high' if direction == 'long' else 'lowest low'} "
                        f"of the trailing 22 days (3x ATR), not off entry price like the fixed stop above. "
                        f"Tightens automatically as the trade moves in your favor; a position-management "
                        f"alternative to the fixed stop, not a second signal about whether to take the trade.*"
                    )

                lines.append("\n**WHY — what drove this candidate (indicators)**")
                for reason in _indicator_reasons(r, direction):
                    lines.append(f"- {reason}")
                lines.append(
                    f"- Sector-level model: {r['sector'].replace('_', ' ').title()} has a calibrated "
                    f"P(top-3-of-13 sectors by 20D return) = {_fmt(r['p_top3_sector'])} from "
                    f"sector_prediction_model (walk-forward measured AUC {_fmt(_rel.get('mean_oos_auc'))}, "
                    f"a real but modest edge — see Model reliability above)."
                )

                lines.append("\n**WHY — supporting news**")
                news_result = _news_for_candidate(r["ticker"], direction, refresh=refresh)
                if news_result.get("available"):
                    lines.append(
                        f"- \"{news_result['title']}\" — {news_result.get('source', 'unknown source')}, "
                        f"{news_result.get('published_at', 'date unknown')}. Sentiment: **{news_result['sentiment']}** "
                        f"— **{news_result['alignment']}**. ({news_result['n_articles_checked']} recent article(s) checked; "
                        f"showing the most recent.)"
                    )
                    if news_result.get("url"):
                        lines.append(f"  Source link: {news_result['url']}")
                    if news_result.get("is_earnings_surprise"):
                        lines.append(
                            "  *This reads as an EARNINGS BEAT/MISS headline specifically, not general "
                            "sentiment — post-earnings-announcement drift is one of the most-studied real "
                            "anomalies in finance, though contested for large, liquid names like most "
                            "candidates in this report (evidence for it weakening post-2006 outside "
                            "microcaps). Treat this headline as a real, if disputed, event, not a stronger "
                            "signal than the sentiment/alignment call above already gives it.*"
                        )
                else:
                    lines.append(f"- No usable recent news found for {r['ticker']} this run ({news_result.get('reason') or news_result.get('error', 'unavailable')}).")

                lines.append("\n**WHEN — timing risk**")
                when_result = _when_for_candidate(r["ticker"], r.get("as_of") or "", refresh=refresh)
                if when_result.get("available"):
                    sev = when_result.get("severity", "UNKNOWN")
                    if sev in ("IMMEDIATE", "ELEVATED"):
                        lines.append(f"- ⚠️ **{sev} earnings risk** — {when_result.get('note', '')}")
                    elif sev == "APPROACHING":
                        lines.append(f"- Earnings approaching — {when_result.get('note', '')}")
                    elif sev == "NONE":
                        lines.append("- No near-term earnings risk flagged — a normal window to consider entry timing on its own technical/news merits.")
                    else:
                        lines.append(f"- Earnings timing unknown: {when_result.get('note', '')}")
                else:
                    lines.append(f"- Earnings timing unavailable this run: {when_result.get('error', 'unknown error')}.")

                lines.append("\n**WHEN — entry and exit timing**")
                lines.append(
                    f"- **Entry**: at the next regular market session (9:30am–4:00pm ET) — this system holds "
                    f"daily bars only, so it cannot recommend a specific intraday minute; the {_fmt(r.get(entry_key))} "
                    "entry above is yesterday's close, not a live quote, so expect the actual fill to differ. "
                    "A limit order at that price, or simply entering at the next open, are the two honest options "
                    "this data supports — anything more precise (e.g. \"buy at 10:32am\") would be invented, not computed."
                )
                mean_days = _mean_days_held_by_direction.get(direction)
                from processing.ml.trade_mechanics_backtest import MAX_HOLD_DAYS as _MAX_HOLD_DAYS

                if mean_days is not None:
                    lines.append(
                        f"- **Exit**: automatically at Stop ({_fmt(r.get(stop_key))}) or TP1 ({_fmt(r.get(tp1_key))}) "
                        "above, whichever a broker's standing stop/limit orders trigger first — those execute "
                        "continuously during market hours, not at a fixed clock time. If NEITHER has been hit after "
                        f"**~{mean_days:.0f} trading days** (the measured average time-to-resolution for this exact "
                        f"direction's mechanics, from the trade-mechanics backtest above), the position has "
                        "outlasted where most resolved trades in that backtest already went one way or the other — "
                        f"worth a discretionary look. By **{_MAX_HOLD_DAYS} trading days** with neither level hit, "
                        f"the backtest itself would have called this a TIMEOUT (not counted as a win or loss), which "
                        "is this system's own definition of \"this setup didn't play out as expected\" — a "
                        "data-derived checkpoint to reassess, not a hard rule to close the position."
                    )
                else:
                    lines.append(
                        f"- **Exit**: automatically at Stop ({_fmt(r.get(stop_key))}) or TP1 ({_fmt(r.get(tp1_key))}) "
                        "above. Typical time-to-resolution is unavailable this run (trade-mechanics backtest not "
                        "found) — run `python -m processing.ml.trade_mechanics_backtest` to generate it."
                    )

                style = r.get("trade_style") or {}
                if style:
                    lines.append("\n**DAY TRADE OR SWING TRADE?**")
                    lines.append(f"- **Verdict: {style['primary_recommendation']}** — {style['rationale']}")
                    lines.append(f"- Swing-horizon fit: **{style['swing_fit']}**")
                    for reason in style.get("swing_reasons", []):
                        lines.append(f"  - {reason}")
                    lines.append(
                        f"- Day-trade *candidacy* screen (liquidity + realized range, pending intraday "
                        f"confirmation): **{style['day_trade_screen']}**"
                    )
                    for reason in style.get("day_trade_reasons", []):
                        lines.append(f"  - {reason}")
                    lines.append(f"  - *{style['day_trade_caveat']}*")

                n_confirm, n_conflict = _indicator_agreement_counts(r, direction)
                verdict = scoring.conviction_verdict(
                    direction=direction,
                    n_indicators_confirm=n_confirm,
                    n_indicators_conflict=n_conflict,
                    earnings_severity=when_result.get("severity") if when_result.get("available") else None,
                    swing_fit=style.get("swing_fit"),
                    historical_expectancy_r=_expectancy_by_direction.get(direction),
                    sector_has_edge=r.get("sector_has_edge"),
                    sector_clears_margin=r.get("sector_clears_margin"),
                )
                lines.append(f"\n**VERDICT: {verdict['verdict']}**")
                for reason in verdict["reasons"]:
                    lines.append(f"- {reason}")
                if verdict["capped_by_measured_expectancy"]:
                    lines.append(
                        "- *This verdict is capped by measured historical mechanics, not by this "
                        "candidate's own indicators/news/timing — even a candidate with strong-looking "
                        "technicals inherits this cap, because the cap is about whether the entry/stop/TP1 "
                        "convention itself has worked historically for this direction, not about this "
                        "specific ticker.*"
                    )

                lines.append(
                    "\n*This candidate's ranking is driven mainly by the sector-level probability above "
                    "(the only walk-forward-validated part of this system). The indicator and news "
                    "reasoning here are real, live-computed context — not a second backtested signal. "
                    "Weigh accordingly.*"
                )

    lines.append("\n---\n# SUPPORTING DATA")

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
                ("Stochastic %K", _fmt(idx.get("stoch_k"))),
                ("ADX14 (trend strength)", _fmt(idx.get("adx14"))),
                ("Bollinger %B", _fmt(idx.get("bb_pctb"))),
                ("ROC10", f"{_fmt(idx.get('roc10'))}%"),
                ("Trend bucket", idx.get("trend_bucket", "n/a")),
                ("**Technical composite (0-100)**", f"**{_fmt(idx.get('technical_composite'))}**"),
            ]
        )
    lines.append("\n**Summary**")
    lines += _kv_table(
        [
            ("**Overall trend (SPY & QQQ agreement required)**", f"**{mr.get('overall_trend', 'n/a')}**"),
            ("VIX level", _fmt(mr.get("vix", {}).get("last"))),
            ("VIX regime", mr.get("vix", {}).get("regime", "n/a")),
            ("**Risk label**", f"**{mr.get('risk_label', 'n/a')}**"),
            ("Market technical composite (avg SPY+QQQ, 0-100)", _fmt(mr.get("market_technical_composite"))),
            ("**Market health (is the market technically good right now?)**", f"**{mr.get('market_health', 'n/a')}**"),
        ]
    )
    lines.append(f"\n*{mr.get('market_health_note', '')}*")
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
        "Ranked by 20D realized volatility — Phase 4 identified this as the strongest surviving "
        "empirical pattern in this project's reported research tests. Phase 5 stress-testing found the "
        "interaction term between volatility and momentum is SMALL relative to the volatility main "
        "effect — this is mostly a volatility effect with a secondary momentum tilt, not a strong true "
        "interaction; weight momentum accordingly, lower than the bucket label alone suggests. See the "
        "structured research metadata below rather than this prose for the actual status of each "
        "validation test (Rule 30: research status must come from DATA fields, not be assumed)."
    )

    try:
        _phase4 = json.loads((ARTIFACT_DIR / "phase4_report.json").read_text())
    except Exception:
        _phase4 = {}
    _dev_spread = (_phase4.get("sector_rotation_backtest", {}).get("volatility_only_top1", {}).get("spread", {}))
    _dev_costs = (_phase4.get("sector_rotation_backtest", {}).get("volatility_only_top1", {}).get("cost_adjusted_top_leg_return", {}))
    _holdout_spread = (_phase4.get("golden_holdout", {}).get("holdout_result", {}))
    if _dev_spread or _holdout_spread:
        # Three DIFFERENT estimands, named precisely so they are never read as
        # three measurements of the same quantity: (1) excess return of the top
        # sector OVER the equal-weight benchmark, (2) the top sector's own GROSS
        # return with no benchmark subtracted, (3) that gross return after an
        # EXPLICIT, itemized transaction-cost model.
        lines.append("\n**top_sector_vs_equal_weight_benchmark_20d** (excess return estimand — structured, not prose)")
        lines += _kv_table(
            [
                ("value_dev_period", f"{_fmt(_dev_spread.get('mean_top_minus_bench_pct'))}pp"),
                ("n_dates_dev_period", _fmt(_dev_spread.get("n_dates"))),
                ("std_dev_period", f"{_fmt(_dev_spread.get('top_return_std'))}pp"),
                ("value_golden_holdout", f"{_fmt(_holdout_spread.get('mean_top_minus_bench_pct'))}pp"),
                ("n_dates_holdout", _fmt(_holdout_spread.get("n_dates"))),
                ("source", "storage/models/phase4_report.json"),
                ("status", "VERIFIED — reproducible from logged experiment data" if _dev_spread else "UNAVAILABLE"),
            ]
        )
        if _dev_spread:
            lines.append("\n**gross_top_sector_return_20d** (a DIFFERENT estimand from the excess-return figure above — no benchmark subtracted)")
            lines += _kv_table([("value_dev_period", f"{_fmt(_dev_spread.get('mean_top_return_pct'))}%"), ("source", "storage/models/phase4_report.json, same backtest run")])
        if _dev_costs:
            lines.append(
                "\n**transaction_cost_model** (explicit, itemized — this is WHY gross->net is a 50bps reduction on a "
                "\"25bps\" cost assumption, not 25bps: the backtest charges cost on BOTH sides of each rebalance)"
            )
            lines += _kv_table(
                [
                    ("cost_per_side_bps", 25),
                    ("n_sides_per_rebalance", 2),
                    ("cost_basis", "closing_previous_position + opening_new_position"),
                    ("total_cost_bps", 50),
                    ("gross_top_sector_return_20d_dev", f"{_fmt(_dev_spread.get('mean_top_return_pct'))}%"),
                    ("net_top_sector_return_20d_dev_after_50bps", f"{_fmt(_dev_costs.get('25bps'))}%"),
                ]
            )
        lines.append(
            "\n*Do not compare top_sector_vs_equal_weight_benchmark_20d, gross_top_sector_return_20d, and any "
            "decile-spread or other effect-size figure mentioned elsewhere as though they measure the same "
            "quantity — each is a different estimand (excess-vs-benchmark vs. gross-vs-nothing vs. top-decile-"
            "minus-bottom-decile, etc.), not repeated measurements of one number. n_dates in both periods "
            "counts overlapping 5-day-stride rebalance dates, not independent trials — read std/sharpe_like "
            "as descriptive, not a rigorous significance test.*"
        )

    try:
        _phase5 = json.loads((_PHASE5_REPORT_PATH).read_text())
        _survival = _phase5.get("edge_survival_score", {})
        _decomp = _phase5.get("interaction_decomposition", {})
    except Exception:
        _survival, _decomp = {}, {}

    if _decomp:
        lines.append("\n**research_hypothesis** (what the research actually indicates is driving the effect — see Rule 43)")
        lines += _kv_table(
            [
                ("primary_driver", "HIGH_REALIZED_VOLATILITY"),
                ("secondary_condition", "NEGATIVE_MOMENTUM"),
                ("coef_volatility", _decomp.get("coef_volatility")),
                ("coef_momentum", _decomp.get("coef_momentum")),
                ("coef_interaction", _decomp.get("coef_interaction")),
                ("interaction_strength", "SMALL" if abs(_decomp.get("coef_interaction", 0)) < abs(_decomp.get("coef_volatility", 1)) * 0.3 else "MEANINGFUL"),
                ("source", "storage/models/phase5_report.json:interaction_decomposition"),
            ]
        )

    if _survival:
        lines.append("\n**research_edge_status (historical validation — machine-computed, see Rule 30)**")
        for check_name, check in _survival.get("checks", {}).items():
            crit = " [CRITICAL]" if check.get("critical") else ""
            lines.append(f"- `{check_name}`{crit}: **{check.get('status')}** — {check.get('reason', '')}")
        lines += _kv_table(
            [
                ("**research_edge_status (overall)**", f"**{_survival.get('overall', 'n/a')}**"),
                ("critical_checks_not_passing", ", ".join(_survival.get("critical_checks_not_passing", [])) or "none"),
                ("n_pass / n_weak / n_fail_or_untested", f"{_survival.get('n_pass', 'n/a')} / {_survival.get('n_weak', 'n/a')} / {_survival.get('n_fail_or_untested', 'n/a')}"),
            ]
        )
        if _survival.get("note"):
            lines.append(f"\n*{_survival['note']}*")

    snap = _sector_snapshot(refresh=refresh)
    if "error" in snap.columns and snap["realized_vol_20d"].isna().all():
        lines.append("Sector snapshot unavailable this run.")
    else:
        lines.append("\n| Sector | 20D Realized Vol | Vol Rank | 20D Momentum | Primary Signal | Secondary Modifier | RSI14 |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in snap.iterrows():
            if pd.isna(r.get("realized_vol_20d")):
                continue
            primary = "HIGH_VOL" if r.get("vol_bucket") == "high_vol" else "LOW_VOL"
            secondary = "NEGATIVE_MOMENTUM" if r.get("momentum_bucket") == "down" else "POSITIVE_MOMENTUM"
            flag = " **← primary+secondary both match the research finding**" if primary == "HIGH_VOL" and secondary == "NEGATIVE_MOMENTUM" else ""
            lines.append(
                f"| {r['sector'].replace('_', ' ').title()} | {_fmt(r.get('realized_vol_20d'))}% | "
                f"{_fmt(r.get('vol_rank_pct_of_sectors'))}%ile | {_fmt(r.get('ret_20d'))}% | {primary} | {secondary}{flag} | {_fmt(r.get('rsi14'))} |"
            )

        # current_signal_activation is a SEPARATE question from research_edge_status
        # above: research_edge_status asks "does this pattern have historical merit at
        # all" (answered once, from stored research); this asks "do TODAY's live
        # numbers actually satisfy that pattern right now" (recomputed every run, from
        # the live snapshot). A MODERATE/WEAK historical edge with an ACTIVE current
        # signal, or a well-supported edge with NO current signal, are both valid,
        # different states — never collapse them into one confidence label.
        matching = snap[(snap.get("vol_bucket") == "high_vol") & (snap.get("momentum_bucket") == "down")].dropna(subset=["realized_vol_20d"])
        if matching.empty:
            activation, strength, strength_reason = "NO", "N/A", "No sector currently combines high-volatility-rank with negative 20D momentum."
        else:
            top_match = matching.sort_values("vol_rank_pct_of_sectors", ascending=False).iloc[0]
            activation = "YES"
            vol_rank = top_match.get("vol_rank_pct_of_sectors") or 0
            mom = top_match.get("ret_20d") or 0
            if vol_rank >= 85 and mom <= -5:
                strength, strength_reason = "STRONG", f"{top_match['sector']} sits at the {_fmt(vol_rank)}th vol percentile with a sizeable negative momentum ({_fmt(mom)}%) — a clean, extreme match to the historical pattern on both axes."
            elif vol_rank >= 60:
                unmet = f"volatility is only {_fmt(vol_rank)}th percentile (below the 85th-pct STRONG bar)" if vol_rank < 85 else f"momentum ({_fmt(mom)}%) is negative but not past the -5% STRONG bar"
                strength, strength_reason = "MODERATE", f"{top_match['sector']} matches the high-vol/down-momentum bucket (vol {_fmt(vol_rank)}th pct, momentum {_fmt(mom)}%), but {unmet} — a real but not extreme match."
            else:
                strength, strength_reason = "WEAK", f"{top_match['sector']} technically matches the bucket (high-vol/down) but only barely — {_fmt(vol_rank)}th vol percentile is a marginal case, not a clear signal."
        lines.append("\n**current_signal_activation (live, recomputed every run — distinct from research_edge_status above)**")
        lines += _kv_table(
            [
                ("current_signal_active", activation),
                ("activation_rule", "vol_bucket == high_vol (>= 50th percentile of realized_vol_20d among the 13 sectors) AND momentum_bucket == down (20D return < 0)"),
                ("current_signal_strength", strength),
                ("strength_rule_STRONG", "vol_percentile >= 85th AND 20D momentum <= -5%"),
                ("strength_rule_MODERATE", "activation rule met, but STRONG thresholds not both met"),
                ("strength_rule_WEAK", "vol_percentile below 60th (a marginal activation-rule pass)"),
                ("strength_rule_N/A", "current_signal_active is NO"),
                ("reason", strength_reason),
                ("edge_scope", "asset_class=equities, unit=sectors, direction=long, benchmark=equal_weight_sector — NOT automatically applicable to crypto, indices, or individual stocks"),
                ("research_horizon", "20 trading days"),
                ("current_signal_horizon", "20 trading days — this signal says nothing about tomorrow or next week specifically"),
            ]
        )
        lines.append(
            "\n*current_signal_strength measures how strongly today's data matches the historical signal "
            "DEFINITION (i.e. how extreme the vol/momentum readings are) — it does NOT measure expected "
            "return, probability of success, or trade quality. A STRONG match to the pattern's definition "
            "is not the same claim as a strong expected trade outcome.*"
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

    # ---- Next Market Move — sector prediction model -----------------------------
    lines.append("\n## Next Market Move — Predictive Model (sector_prediction_model)")
    lines.append(
        "A calibrated model (gradient boosting + logistic regression ensemble, walk-forward validated) "
        "covering ALL 13 US equity sectors — technology, semiconductors, software/internet, financials, "
        "healthcare, industrials, energy, consumer discretionary, consumer staples, utilities, materials, "
        "communication services, real estate — not only semiconductors. It does NOT predict whether the "
        "market will rise or fall. It predicts, per sector, a calibrated probability of landing in the "
        "top-3-of-13 sectors by 20-trading-day forward return."
    )
    try:
        from processing.ml.sector_prediction_model import predict_next_move

        _rocket = predict_next_move(refresh=refresh)
    except Exception as exc:
        _rocket = {"error": str(exc)[:200]}

    if _rocket.get("error"):
        lines.append(f"\nUnavailable this run: {_rocket['error']}")
    else:
        _rel = _rocket.get("model_oos_reliability", {})
        lines.append("\n**Model out-of-sample reliability (measured, not assumed)**")
        lines += _kv_table(
            [
                ("mean_oos_auc", _fmt(_rel.get("mean_oos_auc"))),
                ("mean_oos_brier", _fmt(_rel.get("mean_oos_brier"))),
                ("coin_flip_brier_baseline", _fmt(_rel.get("coin_flip_brier_baseline"))),
                ("**status**", f"**{_rel.get('status', 'n/a')}**"),
            ]
        )
        lines.append(
            "\n*status=PASS means the model beat a coin-flip baseline in walk-forward testing — a real "
            "but MODEST edge (see mean_oos_auc above), not a validated trading signal on its own. Read alongside "
            "research_edge_status above; this model has NOT yet been through the same multi-phase "
            "stress-testing (sector-neutral, adversarial, non-overlapping, point-in-time-universe checks) "
            "as the core Phase 4/5 finding — treat it as an experimental extension, not equally validated.*"
        )
        _spread = _rocket.get("top_bottom_spread") or {}
        if _spread.get("available"):
            lines.append(
                f"\n**Cross-sectional spread check** (top-{_spread['k']} vs. bottom-{_spread['k']} predicted "
                f"sectors, realized forward return, pooled OOS): mean spread "
                f"**{_fmt(_spread['mean_top_minus_bottom_k_return_pct'])}%**, positive on "
                f"{_fmt(_spread['pct_dates_positive_spread'])}% of {_spread['n_dates']} OOS dates. "
                f"*{_spread['note']}*"
            )
        _repro = _rocket.get("repro_metadata") or {}
        if _repro.get("config_hash"):
            lines.append(
                f"\n*Reproducibility: config_hash `{_repro['config_hash']}` — {_repro.get('n_features', 'n/a')} "
                f"features, seed {_repro.get('random_seed', 'n/a')}, trained on "
                f"{_repro.get('n_training_rows', 'n/a')} rows spanning {_repro.get('training_date_range', 'n/a')}. "
                "Any two runs with this same hash used the identical feature set, horizon, and random seed.*"
            )
        lines.append("\n**Calibrated probability, ALL 13 sectors (top-3-of-13 by 20D forward return)**")
        lines.append("| Sector | P(top-3) | Edge? | Model Agreement | 20D Vol | 20D Momentum |")
        lines.append("|---|---|---|---|---|---|")
        for p in _rocket.get("predictions_all_13_sectors", []):
            if "error" in p:
                continue
            gate = p.get("edge_gate") or {}
            if p.get("has_edge") and gate.get("clears_margin_of_error"):
                edge_label = "EDGE (clears margin)"
            elif p.get("has_edge"):
                edge_label = "edge (within margin)"
            elif p.get("has_edge") is False:
                edge_label = "NO EDGE"
            else:
                edge_label = "n/a"
            lines.append(
                f"| {p['sector'].replace('_', ' ').title()} | {_fmt(p.get('p_top3_ensemble'))} | {edge_label} | "
                f"{_fmt(p.get('model_agreement'))} | {_fmt(p.get('realized_vol_20d'))}% | {_fmt(p.get('ret_20d'))}% |"
            )
        lines.append(f"\n*{_rocket.get('note', '')}*")
        lines.append(f"\n*{_rocket.get('abstention_note', '')}*")
        lines.append(
            "\n*The lowest-P(top-3) sectors in this table are the model's LEAST-favored, not a positively "
            "validated \"will underperform\" signal (sector_prediction_model was trained and evaluated to "
            "predict TOP-3, never separately validated as a bottom-predictor). Named, ranked short "
            "candidates are in TOP SHORT CANDIDATES at the top of this report — that table, not this one, "
            "is this report's single source of truth for which tickers to short; it is not repeated here "
            "under different selection logic to avoid the two silently disagreeing.*"
        )

    # ---- 60-Day Outlook — secondary horizon, additive research finding ----
    lines.append("\n## 60-Day Outlook (secondary horizon)")
    lines.append(
        "processing/ml/multi_horizon_research.py tested whether sector-rotation predictability differs by "
        "horizon (1/5/20/60 trading days) and found 60D genuinely stronger than the 20D horizon above on "
        "BOTH a classification re-run (AUC) and an independent Ridge regression re-run (Spearman rank IC) "
        "— two different target formulations agreeing is a real, if still small-sample, signal. Shown here "
        "as an ADDITIVE second opinion, not a replacement: 60D has NOT yet been through the same depth of "
        "stress-testing (Phase 4/5-style adversarial/non-overlapping checks) as the 20D target, and no "
        "candidate ranking, trade level, or backtest anywhere else in this report uses it — those all stay "
        "on the more-validated 20D horizon."
    )
    try:
        from processing.ml.sector_prediction_model import SECONDARY_HORIZON_DAYS

        _rocket60 = predict_next_move(refresh=refresh, horizon_days=SECONDARY_HORIZON_DAYS)
    except Exception as exc:
        _rocket60 = {"error": str(exc)[:200]}

    if _rocket60.get("error"):
        lines.append(f"\nUnavailable this run: {_rocket60['error']}")
    else:
        _rel60 = _rocket60.get("model_oos_reliability", {})
        lines.append("\n**60D model out-of-sample reliability (measured, not assumed)**")
        lines += _kv_table(
            [
                ("mean_oos_auc", _fmt(_rel60.get("mean_oos_auc"))),
                ("mean_oos_brier", _fmt(_rel60.get("mean_oos_brier"))),
                ("coin_flip_brier_baseline", _fmt(_rel60.get("coin_flip_brier_baseline"))),
                ("**status**", f"**{_rel60.get('status', 'n/a')}**"),
            ]
        )
        lines.append("\n**Calibrated probability, ALL 13 sectors (top-3-of-13 by 60D forward return)**")
        lines.append("| Sector | P(top-3, 60D) | Model Agreement | 20D Vol | 20D Momentum |")
        lines.append("|---|---|---|---|---|")
        for p in _rocket60.get("predictions_all_13_sectors", []):
            if "error" in p:
                continue
            lines.append(
                f"| {p['sector'].replace('_', ' ').title()} | {_fmt(p.get('p_top3_ensemble'))} | "
                f"{_fmt(p.get('model_agreement'))} | {_fmt(p.get('realized_vol_20d'))}% | {_fmt(p.get('ret_20d'))}% |"
            )

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
        "\n---\n*End of report. TOP LONG CANDIDATES and TOP SHORT CANDIDATES at the top are the direct "
        "output of this run; everything below them is supporting detail for anyone who wants to check the "
        "underlying data. No interpretation step is required or expected.*"
    )

    lines.append(_ai_handoff_prompt())

    return "\n".join(lines)


def _ai_handoff_prompt() -> str:
    """Layer 3, as this module's own docstring calls it: interpretation left
    for the user to run through any chatbot. This is that prompt — a copy-
    paste block for a SEPARATE AI session (this system stays zero-LLM; the
    prompt below is text output, not a call this code makes itself). It
    instructs the reading AI to EXTRACT, never invent, from the report
    above it — the report is deterministic and already carries every
    number a final order needs; the only new work asked of the AI is
    formatting and respecting the report's own AVOID gate."""
    return (
        "\n---\n"
        "## Prompt for another AI: turn this report into orders\n\n"
        "*Paste everything above this line, plus the instructions below, into a separate AI chat "
        "(ChatGPT, Claude, etc.). This report is zero-LLM — every number above is fetched or "
        "computed by code, never invented — this final step is optional and up to you.*\n\n"
        "```\n"
        "TASK (do this now, in this response — do not just summarize the report above, and do not "
        "reply with only a disclaimer): using ONLY the report pasted above this prompt, fill in the "
        "ORDER TABLE template below for every LONG and SHORT candidate whose VERDICT is not AVOID. "
        "This is a formatting task over data that's already fully computed above, not a request for "
        "your own opinion or new analysis — every value you write must be copied from the report's "
        "TOP LONG CANDIDATES / TOP SHORT CANDIDATES tables or that candidate's HOW section.\n\n"
        "Output exactly this table (one row per qualifying candidate, Markdown table syntax):\n\n"
        "| Ticker | Direction | Entry (limit) | Stop (stop-market) | Take-profit (limit) | Why (from the report's own VERDICT) |\n"
        "|---|---|---|---|---|---|\n"
        "| ... | BUY / SELL SHORT | price | price | price | 1 line |\n\n"
        "Rules:\n"
        "1. EXTRACT ONLY — never invent a ticker, price, or fact not stated in the report. If a "
        "number isn't in the report, write \"not stated,\" never an estimate.\n"
        "2. RESPECT THE VERDICT GATE — skip any candidate whose VERDICT is AVOID, even if its own "
        "indicators look good; AVOID means the report's own walk-forward backtest measured NEGATIVE "
        "expectancy for that exact direction's mechanics, which overrides individual technicals. If "
        "EVERY candidate in a direction is AVOID, write one line saying so (e.g. \"No qualifying "
        "SHORT candidates this run\") instead of an empty table or a refusal.\n"
        "3. Entry = LIMIT order at the report's Entry price (note once, briefly, that this is the "
        "latest daily close, not a live quote, so a market order would fill at a different price).\n"
        "4. After the table, state the report's own measured reliability in 1-2 sentences (the AUC/"
        "Brier from \"Model reliability\" and the LONG/SHORT expectancy in R from the trade-mechanics "
        "backtest section) — a real but modest edge, not a guarantee.\n"
        "5. End with ONE short sentence: this is not financial advice, you're not a licensed advisor, "
        "no position sizes are suggested since no account size is known. That single sentence is the "
        "complete disclaimer — the table above it is still required in full; do not substitute the "
        "disclaimer for the table.\n"
        "```"
    )
