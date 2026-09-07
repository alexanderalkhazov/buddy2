"""Deterministic 0-100 scorecard — computed by code, never by the LLM.

Every formula here is fixed and documented in-line. The model's job is to report
these numbers and explain them ("the system scores this X because Y"), never to
invent its own. A component is None when there isn't enough data to compute it
honestly — None must be reported as "unavailable," never guessed.

This does not claim to be a validated quantitative model. The weights and
thresholds are a reasonable, transparent starting point, not a backtested scoring
system — say so if asked how the weights were chosen.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _rescale(value: float, lo: float, hi: float) -> float:
    """Linearly map [lo, hi] -> [0, 100], clipping outside the range."""
    if hi == lo:
        return 50.0
    return _clip((value - lo) / (hi - lo) * 100, 0, 100)


def technical_score(rsi14: float | None, macd_hist: float | None, trend: str | None) -> float | None:
    """Trend (0-50) + momentum (0-50).

    Trend: count of {sma20, sma50, sma200} the price sits above, from the trend
    label ("above_all_smas" etc.), scaled to 0-50.
    Momentum: RSI centered on 50-65 (healthy uptrend, not yet overbought) scores
    highest; RSI beyond 80 or below 20 (extreme, higher mean-reversion risk) scores
    low. MACD histogram > 0 adds up to 15 points on top of the RSI component.
    """
    if rsi14 is None or trend is None:
        return None

    if trend == "above_all_smas":
        trend_component = 50.0
    elif trend == "below_all_smas":
        trend_component = 0.0
    else:
        n_above = trend.count("sma")
        trend_component = n_above / 3 * 50.0

    if 45 <= rsi14 <= 65:
        rsi_component = 35.0
    elif rsi14 > 80 or rsi14 < 20:
        rsi_component = 5.0
    else:
        # linear falloff from the 45-65 sweet spot toward the extremes
        distance = min(abs(rsi14 - 45), abs(rsi14 - 65))
        rsi_component = _clip(35.0 - distance * 0.8, 5, 35)

    macd_component = 15.0 if (macd_hist or 0) > 0 else 0.0

    return round(_clip(trend_component + rsi_component + macd_component, 0, 100), 1)


def fundamental_score(profit_margin: float | None, revenue_growth: float | None) -> float | None:
    """Margin (0-50) + growth (0-50).

    Margin: 0% margin -> 0, 40%+ margin -> 50 (linear).
    Growth: -20% YoY -> 0, +40% YoY -> 50 (linear; revenue_growth from yfinance is a
    fraction, e.g. 0.05 = 5%). Extreme cyclical-recovery growth rates (>40%) are
    capped, not rewarded further — a formula, not a judgment that faster is always
    better.
    """
    if profit_margin is None and revenue_growth is None:
        return None

    margin_component = _rescale(profit_margin, 0, 0.40) / 2 if profit_margin is not None else 25.0
    growth_component = _rescale(revenue_growth, -0.20, 0.40) / 2 if revenue_growth is not None else 25.0
    return round(margin_component + growth_component, 1)


def valuation_score(forward_pe: float | None, peer_forward_pes: list[float] | None = None) -> float | None:
    """Cheaper (lower forward P/E) scores higher.

    With peers: percentile rank among them, inverted (cheapest of the group -> 100).
    Without peers: a fixed, sector-agnostic P/E heuristic — P/E <= 10 -> 90,
    P/E >= 40 -> 10, linear between. This is a rough anchor, explicitly not a
    substitute for real peer comparison; label it as such when peers are missing.
    """
    if forward_pe is None or forward_pe <= 0:
        return None

    if peer_forward_pes:
        valid_peers = [p for p in peer_forward_pes if p and p > 0]
        if valid_peers:
            cheaper_or_equal = sum(1 for p in valid_peers if p >= forward_pe)
            percentile = cheaper_or_equal / len(valid_peers)
            return round(percentile * 100, 1)

    return round(100 - _rescale(forward_pe, 10, 40), 1)


def sentiment_score(news_items: list[dict] | None) -> float | None:
    """50 = neutral/no signal. Each net positive/negative article shifts the score,
    capped at +/-50 so a handful of headlines can't swing it to the extreme."""
    if not news_items:
        return None
    positive = sum(1 for n in news_items if n.get("sentiment") == "positive")
    negative = sum(1 for n in news_items if n.get("sentiment") == "negative")
    total = len(news_items)
    if total == 0:
        return None
    return round(_clip(50 + (positive - negative) / total * 50, 0, 100), 1)


WEIGHTS = {"fundamental": 0.40, "technical": 0.30, "valuation": 0.20, "sentiment": 0.10}


def overall_score(components: dict[str, float | None]) -> float | None:
    """Weighted average over WEIGHTS, renormalized across whatever components are
    actually available (missing components are excluded, not treated as zero)."""
    available = {k: v for k, v in components.items() if v is not None}
    if not available:
        return None
    weight_sum = sum(WEIGHTS[k] for k in available)
    return round(sum(WEIGHTS[k] * v for k, v in available.items()) / weight_sum, 1)


def decision_label(score: float | None) -> str:
    """Fixed thresholds, stated plainly so they can be argued with:
    >=70 BUY | 55-69 CONDITIONAL BUY | 40-54 HOLD | <40 AVOID."""
    if score is None:
        return "INSUFFICIENT_DATA"
    if score >= 70:
        return "BUY"
    if score >= 55:
        return "CONDITIONAL BUY"
    if score >= 40:
        return "HOLD"
    return "AVOID"


def position_size(
    entry_price: float, stop_price: float, account_size: float, risk_pct: float = 1.0
) -> dict:
    """Deterministic risk-based position size — this arithmetic must never be done
    freehand by the LLM. Shares are FLOORED, never rounded, so actual risk can only
    come in at or under the target — rounding up would silently exceed the stated
    risk limit, which is exactly the bug this function exists to prevent."""
    if entry_price <= 0 or account_size <= 0 or risk_pct <= 0:
        return {"error": "entry_price, account_size, and risk_pct must all be positive"}
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return {
            "error": "stop_price must be below entry_price for a long position "
            f"(got entry={entry_price}, stop={stop_price})"
        }

    target_max_loss = account_size * (risk_pct / 100)
    shares = int(target_max_loss // risk_per_share)  # floor, not round
    capital_invested = round(shares * entry_price, 2)
    actual_max_loss = round(shares * risk_per_share, 2)

    return {
        "entry_price": entry_price,
        "stop_price": stop_price,
        "risk_per_share": round(risk_per_share, 4),
        "account_size": account_size,
        "target_risk_pct": risk_pct,
        "target_max_loss": round(target_max_loss, 2),
        "shares": shares,
        "capital_invested": capital_invested,
        "portfolio_allocation_pct": round(capital_invested / account_size * 100, 2),
        "actual_max_loss": actual_max_loss,
        "actual_risk_pct": round(actual_max_loss / account_size * 100, 4),
        "note": (
            "shares is floored (never rounded up) so actual_risk_pct never exceeds "
            "target_risk_pct. portfolio_allocation_pct (capital at work) and "
            "actual_risk_pct (capital actually at risk to the stop) are different "
            "numbers — don't conflate them."
        ),
        "warning": (
            "risk_pct too tight for this stop distance at this account size — 0 "
            "shares fit within the target risk" if shares == 0 else None
        ),
    }


def earnings_proximity_risk(as_of: str | None, next_earnings_date: str | None) -> dict:
    """Severity-tiered warning for how close the next earnings report is to the
    latest available price date — a stop computed from ATR assumes roughly
    continuous price movement, which an earnings gap can violate (price can open
    beyond the stop with no fill available at the stated level). Does not change
    any computed stop/price — purely a warning about execution risk around the
    event, sitting alongside the arithmetic, not inside it."""
    if not as_of or not next_earnings_date:
        return {"calendar_days_until_earnings": None, "severity": "UNKNOWN", "note": "next_earnings_date not available — cannot assess earnings-proximity risk."}

    from datetime import date

    try:
        as_of_date = date.fromisoformat(str(as_of)[:10])
        earnings_date = date.fromisoformat(str(next_earnings_date)[:10])
    except ValueError:
        return {"calendar_days_until_earnings": None, "severity": "UNKNOWN", "note": "could not parse a date — cannot assess earnings-proximity risk."}

    days = (earnings_date - as_of_date).days

    if days < 0:
        severity, label = "PASSED", "Next reported earnings date is in the past relative to the latest price data — likely a stale or already-reported date; treat as unknown until refreshed."
    elif days <= 1:
        severity, label = "IMMEDIATE", "Earnings report is imminent (today or tomorrow, relative to the latest available price). A stop-loss computed from ATR does not account for an overnight/after-close earnings gap — price can open beyond the stop with no fill available at that level."
    elif days <= 6:
        severity, label = "ELEVATED", "Earnings report is within the next week. Historical implied volatility and realized moves typically rise into an earnings date — see get_earnings_reaction_history for this ticker's actual past reaction sizes."
    elif days <= 14:
        severity, label = "APPROACHING", "Earnings report is within two weeks. Not immediate, but worth checking get_earnings_reaction_history before assuming normal (non-event) price behavior between now and the report."
    else:
        severity, label = "NONE", "No earnings report expected in the near term based on next_earnings_date."

    return {
        # Explicitly "calendar days," not trading sessions — a next_earnings_date 3
        # calendar days out could be 1-2 trading sessions away depending on weekends,
        # and "3d" alone invites an AI (or a reader) to assume trading days.
        "calendar_days_until_earnings": days,
        "next_earnings_date": next_earnings_date,
        "severity": severity,
        "note": label,
    }


def trade_levels(
    last_price: float | None,
    atr14: float | None,
    stop_multiplier: float = 2.0,
    tp1_r_multiple: float = 1.5,
    tp2_r_multiple: float = 3.0,
    as_of: str | None = None,
    next_earnings_date: str | None = None,
) -> dict:
    """Entry/stop/take-profit levels for a hypothetical LONG position, derived
    purely from price and ATR — not a recommendation to take the trade, just the
    deterministic arithmetic once you decide to. entry is the last available close
    (this system has no live quote, see data_freshness); stop is the same
    ATR-multiple stop as suggested_stop(); TP1/TP2 are R-multiples of the resulting
    entry-to-stop distance (a common, but arbitrary, convention — 1.5R and 3R are
    not fitted to this specific ticker's behavior). risk_reward_ratio compares TP1
    to the stop distance. This function does not know direction is a good idea —
    it only computes what the numbers would be IF a long entry were taken here."""
    if last_price is None or atr14 is None or atr14 <= 0:
        return {"error": "last_price and a positive atr14 are required"}

    stop = round(last_price - stop_multiplier * atr14, 4)
    risk_per_share = round(last_price - stop, 4)
    if risk_per_share <= 0:
        return {"error": "computed stop is not below last_price — cannot derive R-multiples"}

    tp1 = round(last_price + tp1_r_multiple * risk_per_share, 4)
    tp2 = round(last_price + tp2_r_multiple * risk_per_share, 4)

    return {
        "entry": last_price,
        "stop": stop,
        "risk_per_share": risk_per_share,
        "take_profit_1": tp1,
        "take_profit_1_r_multiple": tp1_r_multiple,
        "take_profit_2": tp2,
        "take_profit_2_r_multiple": tp2_r_multiple,
        "risk_reward_ratio_tp1": tp1_r_multiple,  # by construction: TP1 is exactly tp1_r_multiple * risk
        "invalidation": f"Daily close below {stop} invalidates this long setup (stop-loss level breached).",
        # Does not alter entry/stop/TP1/TP2 above — a separate, additive warning
        # about execution risk around an earnings gap, not a recomputation.
        "earnings_proximity": earnings_proximity_risk(as_of, next_earnings_date),
        "note": (
            "entry is the latest available daily close, not a live executable price — "
            "there may be gap risk between this level and your actual fill. stop/TP1/TP2 "
            f"are {stop_multiplier}x/{tp1_r_multiple}R/{tp2_r_multiple}R ATR-based levels, a "
            "common but arbitrary convention, not calibrated to this specific ticker's "
            "historical behavior around similar setups. This assumes a LONG position — "
            "the system does not generate short-side levels. Whether to take this trade "
            "at all is a separate question this function does not answer; it only "
            "computes the arithmetic for a hypothetical long entry at the current level."
        ),
    }


def suggested_stop(last_price: float | None, atr14: float | None, multiplier: float = 2.0) -> float | None:
    """entry - multiplier * ATR(14) — a volatility-derived stop, not an arbitrary
    percentage. multiplier=2.0 is a common convention (roughly a 2-sigma move on a
    normal-ish distribution of daily ranges), not a fitted or validated parameter."""
    if last_price is None or atr14 is None:
        return None
    return round(last_price - multiplier * atr14, 4)


# Coarse industry-keyword heuristic, not a real earnings-cycle model. A stock in one
# of these industries can look extremely cheap on trailing/forward P/E at the top of
# its cycle precisely because cyclical earnings are temporarily elevated — the multiple
# looking low doesn't mean the price is low. This does not know where in the cycle the
# company currently sits; it only flags that the sector has this property at all.
_CYCLICAL_INDUSTRY_KEYWORDS = (
    "semiconductor", "steel", "airline", "auto", "oil", "gas", "mining", "shipping",
    "metals", "chemical", "homebuilding", "commodity", "coal", "drilling", "copper",
)


def cyclicality_risk(sector: str | None, industry: str | None) -> str:
    """HIGH/LOW/UNKNOWN — a sector heuristic only. Never treat this as a substitute
    for actually checking where the company sits in its earnings cycle; it can't see
    that. See module docstring caveat."""
    if not industry and not sector:
        return "UNKNOWN"
    haystack = f"{industry or ''} {sector or ''}".lower()
    return "HIGH" if any(kw in haystack for kw in _CYCLICAL_INDUSTRY_KEYWORDS) else "LOW"


def momentum_extension_risk(
    last_price: float | None, fifty_two_week_high: float | None, rsi14: float | None
) -> str:
    """HIGH/MODERATE/LOW/UNKNOWN — a proxy for "how extended is this move," built
    from distance below the 52-week high plus RSI. This is NOT a magnitude-of-rally
    calculation (it doesn't know the stock is +242% YTD, only that it's near its
    high and still hot) — a real version would need a trailing-return lookback."""
    if last_price is None or fifty_two_week_high is None or fifty_two_week_high <= 0:
        return "UNKNOWN"
    pct_from_high = (fifty_two_week_high - last_price) / fifty_two_week_high
    hot = (rsi14 or 50) > 55
    if pct_from_high < 0.10 and hot:
        return "HIGH"
    if pct_from_high < 0.25:
        return "MODERATE"
    return "LOW"


def volatility_risk(beta: float | None) -> str:
    """HIGH/MODERATE/LOW/UNKNOWN from beta. A blunt, single-number proxy — ATR/price
    (in the price bundle) is a finer-grained, asset-specific complement."""
    if beta is None:
        return "UNKNOWN"
    if beta > 1.5:
        return "HIGH"
    if beta >= 0.8:
        return "MODERATE"
    return "LOW"


def risk_adjusted_conviction(overall: float | None, risk_flags: dict[str, str]) -> str:
    """Pairs the overall score with how many risk flags came back HIGH, so a high
    score on an extended, cyclical, volatile stock doesn't read as a flatly 'safe'
    number the way a bare 0-100 score can. This is a label, not a new formula —
    it does not re-weight anything, just contextualizes the existing overall score."""
    if overall is None:
        return "UNKNOWN"
    high_count = sum(1 for v in risk_flags.values() if v == "HIGH")
    if overall >= 70:
        return "HIGH — but meaningfully elevated risk" if high_count >= 2 else "HIGH"
    if overall >= 55:
        return "MODERATE — elevated risk" if high_count >= 2 else "MODERATE"
    return "LOW"


def company_quality_score(fundamental: float | None, valuation: float | None) -> float | None:
    """'Is this a good business at a reasonable price' — fundamental(60%)/valuation(40%),
    renormalized over whichever is available. Deliberately excludes technical and
    sentiment: those describe trade TIMING, not business quality, and folding them
    in here is exactly the double-counting this split exists to avoid."""
    parts = {"fundamental": (fundamental, 0.6), "valuation": (valuation, 0.4)}
    available = {k: v for k, (v, _) in parts.items() if v is not None}
    if not available:
        return None
    weight_sum = sum(parts[k][1] for k in available)
    return round(sum(parts[k][1] * available[k] for k in available) / weight_sum, 1)


def entry_quality_score(
    last_price: float | None,
    sma20: float | None,
    sma50: float | None,
    fifty_two_week_high: float | None,
    rsi14: float | None,
    atr14: float | None,
) -> float | None:
    """'Is NOW a favorable price to start a new position' — independent of whether
    the company itself is good. A great company can have a terrible entry (extended,
    overbought, far above its 52w high on elevated volatility); a mediocre company
    can have a fine entry (pulled back, not overbought). Starts at 100 and subtracts
    penalties; never rewards distance from SMAs the way technical_score does — being
    far ABOVE a rising SMA is exactly what makes an entry worse, not better."""
    if last_price is None or last_price <= 0:
        return None

    score = 100.0
    penalized_any = False

    if sma20 is not None and sma20 > 0:
        pct_above_20 = (last_price - sma20) / sma20 * 100
        if pct_above_20 > 0:
            score -= _clip(pct_above_20 * 2.5, 0, 30)
        penalized_any = True

    if fifty_two_week_high is not None and fifty_two_week_high > 0:
        pct_from_high = (fifty_two_week_high - last_price) / fifty_two_week_high * 100
        if pct_from_high < 5:
            score -= _clip((5 - pct_from_high) * 4, 0, 25)
        penalized_any = True

    if rsi14 is not None:
        if rsi14 > 70:
            score -= _clip((rsi14 - 70) * 1.5, 0, 25)
        elif rsi14 < 30:
            score -= _clip((30 - rsi14) * 0.5, 0, 10)
        penalized_any = True

    if atr14 is not None and last_price > 0:
        atr_pct = atr14 / last_price * 100
        if atr_pct > 4:
            score -= _clip((atr_pct - 4) * 2, 0, 15)
        penalized_any = True

    if not penalized_any:
        return None
    return round(_clip(score, 0, 100), 1)


def _direction_label(score: float | None) -> str:
    if score is None:
        return "UNKNOWN"
    if score >= 60:
        return "BULLISH"
    if score >= 40:
        return "NEUTRAL"
    return "BEARISH"


def entry_quality_label(score: float | None) -> str:
    if score is None:
        return "UNKNOWN"
    if score >= 75:
        return "EXCELLENT"
    if score >= 55:
        return "GOOD"
    if score >= 35:
        return "FAIR"
    return "POOR"


def thesis_vs_trade(
    company_quality: float | None,
    entry_quality: float | None,
    risk_flags: dict[str, str],
) -> dict:
    """The core split this scoring module was missing: a strong company is not
    automatically a good trade right now, and a good entry doesn't make a weak
    company worth owning. overall_score/decision stay as-is (they answer 'cheap
    and technically strong'); this answers the two questions separately so the
    LLM/report can say 'bullish company, poor current entry' instead of forcing
    both into one number."""
    high_risk_count = sum(1 for v in risk_flags.values() if v == "HIGH")
    if high_risk_count >= 2:
        risk = "HIGH"
    elif high_risk_count == 1:
        risk = "MEDIUM"
    elif company_quality is None and entry_quality is None:
        risk = "UNKNOWN"
    else:
        risk = "LOW"

    return {
        "investment_thesis_direction": _direction_label(company_quality),
        "trade_setup_direction": _direction_label(entry_quality),
        "risk": risk,
        "entry_quality_label": entry_quality_label(entry_quality),
        "note": (
            "investment_thesis_direction answers 'is this a good business at a "
            "reasonable price' (fundamental+valuation only). trade_setup_direction "
            "answers 'is now a favorable price to start a position' (price location "
            "relative to SMA20/52w-high/RSI/ATR%, independent of business quality). "
            "They can and often do disagree — e.g. BULLISH thesis with BEARISH/NEUTRAL "
            "setup means 'good company, bad entry, consider waiting for a pullback,' "
            "not a contradiction to resolve into one score."
        ),
    }


@dataclass
class Scorecard:
    fundamental: float | None
    technical: float | None
    valuation: float | None
    sentiment: float | None
    overall: float | None
    decision: str
    suggested_stop_price: float | None
    risk_flags: dict = None
    risk_adjusted_conviction: str = "UNKNOWN"
    company_quality: float | None = None
    entry_quality: float | None = None
    thesis_vs_trade: dict = None
    weights: dict = None
    methodology: str = (
        "Overall = weighted average of fundamental(40%)/technical(30%)/valuation(20%)/"
        "sentiment(10%), renormalized over available components. Each component and "
        "the decision thresholds are fixed formulas, not LLM judgment — see "
        "processing/scoring.py for the exact math. This is a transparent heuristic, "
        "not a validated quantitative model. decision/overall answer 'is this cheap "
        "and technically strong' — they do NOT answer 'is this safe.' risk_flags "
        "answer a different question (how extended/cyclical/volatile is this) and "
        "must be read together, never decision/overall alone: a high overall score "
        "with HIGH risk flags means high-conviction bullish AND high-risk, not a "
        "flatly safe or obvious buy."
    )

    def to_dict(self) -> dict:
        return {
            "fundamental": self.fundamental,
            "technical": self.technical,
            "valuation": self.valuation,
            "sentiment": self.sentiment,
            "overall": self.overall,
            "decision": self.decision,
            "suggested_stop_price": self.suggested_stop_price,
            "risk_flags": self.risk_flags or {},
            "risk_adjusted_conviction": self.risk_adjusted_conviction,
            "company_quality": self.company_quality,
            "entry_quality": self.entry_quality,
            "thesis_vs_trade": self.thesis_vs_trade or {},
            "weights": WEIGHTS,
            "methodology": self.methodology,
        }


def compute_scorecard(
    *,
    rsi14: float | None,
    macd_hist: float | None,
    trend: str | None,
    profit_margin: float | None,
    revenue_growth: float | None,
    forward_pe: float | None,
    peer_forward_pes: list[float] | None,
    news_items: list[dict] | None,
    last_price: float | None,
    atr14: float | None,
    sector: str | None = None,
    industry: str | None = None,
    beta: float | None = None,
    fifty_two_week_high: float | None = None,
    sma20: float | None = None,
    sma50: float | None = None,
) -> Scorecard:
    fundamental = fundamental_score(profit_margin, revenue_growth)
    technical = technical_score(rsi14, macd_hist, trend)
    valuation = valuation_score(forward_pe, peer_forward_pes)
    sentiment = sentiment_score(news_items)
    overall = overall_score(
        {"fundamental": fundamental, "technical": technical, "valuation": valuation, "sentiment": sentiment}
    )

    flags = {
        "cyclicality_risk": cyclicality_risk(sector, industry),
        "momentum_extension_risk": momentum_extension_risk(last_price, fifty_two_week_high, rsi14),
        "volatility_risk": volatility_risk(beta),
    }

    company_quality = company_quality_score(fundamental, valuation)
    entry_quality = entry_quality_score(last_price, sma20, sma50, fifty_two_week_high, rsi14, atr14)

    return Scorecard(
        fundamental=fundamental,
        technical=technical,
        valuation=valuation,
        sentiment=sentiment,
        overall=overall,
        decision=decision_label(overall),
        suggested_stop_price=suggested_stop(last_price, atr14),
        risk_flags=flags,
        risk_adjusted_conviction=risk_adjusted_conviction(overall, flags),
        company_quality=company_quality,
        entry_quality=entry_quality,
        thesis_vs_trade=thesis_vs_trade(company_quality, entry_quality, flags),
    )
