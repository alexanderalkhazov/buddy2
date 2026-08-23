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


def trade_levels(
    last_price: float | None,
    atr14: float | None,
    stop_multiplier: float = 2.0,
    tp1_r_multiple: float = 1.5,
    tp2_r_multiple: float = 3.0,
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
    )
