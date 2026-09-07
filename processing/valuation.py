"""Valuation sensitivity — what's actually priced in, and what happens if it's wrong.

Two distinct pieces:
1. implied_growth: real, computed from fetched consensus EPS — zero assumptions.
2. sensitivity_grid: a bear/base/bull × EPS × multiple CALCULATOR, not a forecast.
   The scenario percentages are stated assumptions you (or the LLM) choose — this
   answers "if EPS comes in at X% of consensus and the multiple is Y%, price would
   be Z," not "price will be Z." Never let the grid's output be reported as if it
   were a prediction; it's conditional arithmetic on assumptions the caller supplies.
"""

from __future__ import annotations

# Illustrative defaults, not calibrated to anything — a purely deterministic
# starting point the caller should override with their own view of the range.
DEFAULT_SCENARIOS = {
    "bear": {"eps_pct_of_forward": 0.60, "pe_pct_of_current": 0.70},
    "base": {"eps_pct_of_forward": 1.00, "pe_pct_of_current": 1.00},
    "bull": {"eps_pct_of_forward": 1.30, "pe_pct_of_current": 1.20},
}


def implied_growth(trailing_eps: float | None, forward_eps: float | None) -> dict:
    """The percentage gap between trailing and forward-consensus EPS — a comparison
    of two different reporting periods, not a market forecast. "Forward EPS is X%
    above trailing EPS" is what this number literally is; it is NOT the same claim
    as "the market is pricing in X% growth," which would require knowing the market
    explicitly targets that growth rate (it doesn't — forward P/E is a ratio of
    price to one estimate, not a stated growth expectation). A forward P/E that
    looks "cheap" next to a much higher trailing P/E is arithmetically explained by
    this gap, not by the stock actually being underpriced."""
    if not trailing_eps or not forward_eps or trailing_eps <= 0:
        return {
            "trailing_eps": trailing_eps,
            "forward_eps": forward_eps,
            "trailing_to_forward_eps_gap_pct": None,
            "note": "Missing or non-positive trailing EPS — cannot compute the trailing-to-forward EPS gap.",
        }
    growth_pct = round((forward_eps / trailing_eps - 1) * 100, 2)
    return {
        "trailing_eps": trailing_eps,
        "forward_eps": forward_eps,
        "trailing_to_forward_eps_gap_pct": growth_pct,
        "note": (
            f"Forward-consensus EPS is {growth_pct:+.1f}% above trailing EPS "
            f"({forward_eps} vs {trailing_eps}). This is a comparison of two "
            "different EPS reporting periods, not a market forecast — it does not "
            "by itself establish that 'the market is pricing in' that growth rate. "
            "A low forward P/E next to a much higher trailing P/E is this gap, "
            "arithmetically — not evidence the stock is underpriced on its own. "
            "Whether this growth is durable, or even the specific number analysts "
            "expect, is not something this system has data to answer."
        ),
    }


def sensitivity_grid(
    current_price: float | None,
    forward_eps: float | None,
    forward_pe: float | None,
    scenarios: dict | None = None,
) -> dict:
    """Bear/base/bull implied price from EPS-multiplier x P/E-multiplier assumptions.
    Every scenario's percentages are inputs, not computed facts — report them as
    "assuming EPS comes in at N% of consensus and the multiple compresses/expands
    to M% of today's," never as a prediction of what will happen."""
    scenarios = scenarios or DEFAULT_SCENARIOS
    if not current_price or not forward_eps or not forward_pe:
        return {
            "error": "Missing current_price, forward_eps, or forward_pe — cannot "
            "build a sensitivity grid (likely a non-equity asset, or consensus "
            "estimates aren't covered for this ticker)."
        }

    rows = {}
    for name, params in scenarios.items():
        eps_pct = params["eps_pct_of_forward"]
        pe_pct = params["pe_pct_of_current"]
        implied_eps = round(forward_eps * eps_pct, 4)
        implied_pe = round(forward_pe * pe_pct, 4)
        implied_price = round(implied_eps * implied_pe, 4)
        implied_return_pct = round((implied_price / current_price - 1) * 100, 2)
        rows[name] = {
            "assumption": f"EPS at {eps_pct * 100:.0f}% of forward consensus, multiple at {pe_pct * 100:.0f}% of current forward P/E",
            "implied_eps": implied_eps,
            "implied_pe_multiple": implied_pe,
            "implied_price": implied_price,
            "implied_return_pct": implied_return_pct,
        }

    return {
        "current_price": current_price,
        "forward_eps_consensus": forward_eps,
        "forward_pe_current": forward_pe,
        "scenarios": rows,
        "caveat": (
            "This is a conditional calculator, not a forecast: 'if EPS and the "
            "multiple land here, price would be there.' The scenario percentages "
            "are inputs someone chose (defaults are illustrative, not calibrated to "
            "this company's actual cycle) — override eps_pct_of_forward/"
            "pe_pct_of_current with your own view of the range, and always state "
            "the assumptions alongside any number from this grid, never the price "
            "alone."
        ),
    }


def sensitivity_heatmap(
    current_price: float | None,
    forward_eps: float | None,
    forward_pe: float | None,
    eps_pcts: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4),
    pe_pcts: tuple[float, ...] = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5),
) -> dict:
    """A full EPS% x multiple% grid of implied prices — "at what combination does
    this stop making sense" instead of three fixed points. Local/report use only;
    kept out of the LLM tool's default response to avoid bloating a chat request
    with a large grid the model didn't ask for (call it explicitly if needed)."""
    if not current_price or not forward_eps or not forward_pe:
        return {"error": "Missing current_price, forward_eps, or forward_pe."}

    rows = []
    for eps_pct in eps_pcts:
        implied_eps = forward_eps * eps_pct
        row = {"eps_pct_of_forward": round(eps_pct * 100), "implied_eps": round(implied_eps, 2), "prices": {}}
        for pe_pct in pe_pcts:
            implied_pe = forward_pe * pe_pct
            implied_price = round(implied_eps * implied_pe, 2)
            row["prices"][round(pe_pct * 100)] = implied_price
        rows.append(row)

    return {
        "current_price": current_price,
        "pe_pcts_of_current": [round(p * 100) for p in pe_pcts],
        "rows": rows,
        "caveat": (
            "Every cell is a conditional calculation on two assumptions (EPS % of "
            "forward consensus, multiple % of today's forward P/E) — not a "
            "probability-weighted forecast. Use it to see where the current price "
            "sits relative to the range of plausible outcomes, not to pick a single "
            "'true' price."
        ),
    }


# --- Valuation breakdown labels — descriptive only, never fed back into the
# numeric valuation_score in scoring.py. Exists because a single "Valuation: 100"
# number can read as "valuation risk is low," when it may really mean "cheap IF an
# extreme growth assumption holds" — these labels expose which is true without
# changing the underlying score.

def multiple_attractiveness(forward_pe: float | None) -> str:
    """Absolute cheapness of the multiple alone, independent of what growth it
    assumes. Same thresholds as scoring.valuation_score's no-peer fallback,
    translated to a label instead of a 0-100 score."""
    if forward_pe is None or forward_pe <= 0:
        return "UNKNOWN"
    if forward_pe <= 12:
        return "HIGH"
    if forward_pe <= 25:
        return "MODERATE"
    return "LOW"


def forward_growth_dependency(implied_eps_growth_pct: float | None) -> str:
    """How much of the "cheap multiple" story depends on a large, unproven
    forward-earnings jump actually happening."""
    if implied_eps_growth_pct is None:
        return "UNKNOWN"
    if implied_eps_growth_pct >= 100:
        return "EXTREME"
    if implied_eps_growth_pct >= 40:
        return "HIGH"
    if implied_eps_growth_pct >= 15:
        return "MODERATE"
    return "LOW"


def thesis_dependencies(
    ticker: str,
    forward_eps: float | None,
    eps_dispersion_spread_pct: float | None,
    eps_dispersion_count: int | None,
    trailing_to_forward_eps_gap_pct: float | None,
    cyclicality_risk_label: str | None,
    beta: float | None,
    forward_pe: float | None,
) -> dict:
    """What must stay true for the current forward-looking valuation to hold,
    stated as concrete numeric conditions — not a restatement of the scorecard.
    Purely derived from fields already in the bundle; invents nothing new."""
    depends_on = []
    fragility = {}

    if forward_eps is not None:
        depends_on.append(f"Forward-consensus EPS (${forward_eps}) is approximately achieved — this is an analyst estimate, not a guarantee.")
    if eps_dispersion_spread_pct is not None:
        depends_on.append(
            f"That consensus figure is a meaningful summary of {eps_dispersion_count or 'the'} analysts' individual "
            f"estimates, despite a {eps_dispersion_spread_pct}% spread between the low and high estimate."
        )
        fragility["eps_estimate_dispersion"] = "HIGH" if eps_dispersion_spread_pct >= 30 else "MODERATE" if eps_dispersion_spread_pct >= 15 else "LOW"
    if trailing_to_forward_eps_gap_pct is not None:
        depends_on.append(
            f"The {trailing_to_forward_eps_gap_pct:+.1f}% gap between trailing and forward-consensus EPS closes as "
            "expected — if forward estimates are revised down toward trailing levels, the current forward multiple "
            "re-rates upward with no price change."
        )
        fragility["forward_eps_dependency"] = forward_growth_dependency(trailing_to_forward_eps_gap_pct)
    if cyclicality_risk_label:
        depends_on.append(
            f"Current margins/earnings are not primarily a cyclical peak — this system flags cyclicality risk as "
            f"{cyclicality_risk_label} for this sector/industry, based on a keyword heuristic, not modeled cycle data."
        )
        fragility["cyclicality"] = cyclicality_risk_label
    if beta is not None:
        fragility["volatility"] = "HIGH" if beta >= 1.5 else "MODERATE" if beta >= 1.0 else "LOW"
    if forward_pe is not None:
        fragility["multiple_dependency"] = multiple_attractiveness(forward_pe)

    invalidation_conditions = []
    if forward_eps is not None:
        invalidation_conditions.append(
            f"Forward-consensus EPS is revised down materially from the current ${forward_eps} (e.g. by more than "
            "~10-15%, or by any amount alongside guidance cuts) — this attacks the earnings estimate the thesis "
            "is priced on, distinct from the price-based stop in trade_levels()."
        )
    if eps_dispersion_spread_pct is not None:
        invalidation_conditions.append(
            f"Analyst EPS dispersion widens materially beyond the current {eps_dispersion_spread_pct}% spread — "
            "growing disagreement among analysts is itself evidence the forward number is less reliable, "
            "independent of which direction estimates move."
        )
    if cyclicality_risk_label == "HIGH":
        invalidation_conditions.append(
            "Revenue growth or margins decelerate for 2+ consecutive reported quarters — for a HIGH-cyclicality "
            "sector this is the signature of a cyclical peak already having passed, which the current forward "
            "P/E does not price in."
        )
    invalidation_conditions.append(
        "Next-quarter guidance (if given at the next earnings report) comes in below the forward-consensus EPS "
        "used above — guidance is more current information than the pre-report consensus."
    )

    return {
        "ticker": ticker.upper(),
        "thesis_depends_on": depends_on,
        "thesis_fragility": fragility,
        "thesis_invalidation_conditions": invalidation_conditions,
        "note": (
            "Every item here is derived from data already in this bundle — this is not a new data source, just "
            "the existing forward-P/E, EPS-dispersion, and cyclicality fields restated as explicit conditions "
            "instead of a single score. It does not know what specifically drives this company's earnings "
            "(industry pricing, product cycle, guidance) — those would need a dedicated data source not "
            "currently in this system. thesis_invalidation_conditions answers a DIFFERENT question than "
            "trade_levels().invalidation: the trade-level one is a price/stop trigger ('daily close below $X'), "
            "this one is a fundamental/estimate trigger ('EPS consensus cut,' 'guidance misses') — a position "
            "can be invalidated on either axis independently of the other."
        ),
    }


def valuation_breakdown(
    forward_pe: float | None,
    implied_eps_growth_pct: float | None,
    cyclicality_risk_label: str,
) -> dict:
    """Combine the three dimensions a bare 'Valuation: 100' collapses into one
    number. Does not compute a replacement score — descriptive only."""
    return {
        "multiple_attractiveness": multiple_attractiveness(forward_pe),
        "forward_growth_dependency": forward_growth_dependency(implied_eps_growth_pct),
        "cyclicality_adjustment": cyclicality_risk_label,
        "note": (
            "The numeric Valuation score in the scorecard is computed from the "
            "multiple alone (see scoring.valuation_score) — it does not know "
            "whether that multiple depends on an extreme growth assumption or a "
            "cyclical peak. These three labels are what a high valuation score can "
            "hide: a HIGH forward_growth_dependency or cyclicality_adjustment means "
            "'cheap, conditional on X happening,' not 'cheap, low risk.'"
        ),
    }
