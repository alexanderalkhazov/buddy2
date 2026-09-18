"""Trade-mechanics math — computed by code, never by the LLM: ATR-based entry/
stop/take-profit levels, earnings-proximity risk, and the day-trade-vs-swing-
trade suitability screen. Every formula here is fixed and documented in-line;
nothing in this module invents a number."""

from __future__ import annotations


# --- Trade-style fit thresholds ------------------------------------------
# Screening thresholds, not fitted parameters. Sources and the reasoning for
# each are documented in trade_style_fit() below.
_DAY_MIN_DOLLAR_VOLUME = 20_000_000      # $20M/day — the commonly cited floor for trading real size intraday
_DAY_MIN_DAILY_RANGE_PCT = 1.5           # below this there isn't enough daily travel to cover intraday costs
_DAY_MIN_PRICE = 5.0                     # sub-$5 spreads are proportionally punishing for intraday round trips
_SWING_MIN_DOLLAR_VOLUME = 2_000_000     # $2M/day — enough to get in/out over days without moving the tape
_SWING_TRENDING_ADX = 25.0               # Wilder's own threshold for "a trend worth following exists"
_SWING_CHOPPY_ADX = 20.0                 # Wilder: below 20 is choppy/range-bound; 20-25 is ambiguous
_SWING_MIN_EFFICIENCY_RATIO = 0.30       # Kaufman ER — how much of the path actually went somewhere
_HIGH_GAP_SHARE = 0.35                   # >35% of total movement happening overnight = gap/news-driven name


def trade_style_fit(
    last_price: float | None,
    avg_dollar_volume: float | None,
    avg_daily_range_pct: float | None,
    avg_overnight_gap_pct: float | None,
    gap_share_of_range: float | None,
    adx14: float | None,
    efficiency_ratio: float | None = None,
) -> dict:
    """Should this stock be DAY TRADED or SWING TRADED? — a suitability screen
    from daily bars, with one hard limitation stated up front rather than
    buried:

    THIS SYSTEM HAS NO INTRADAY DATA. Every number here comes from daily
    OHLCV bars. That means:

      * The swing verdict is about the horizon this system actually models —
        sector_prediction_model targets 20-TRADING-DAY forward returns, so
        every candidate in this report is a swing-horizon idea by
        construction, and the swing fit below judges whether this particular
        stock suits being held that long (liquidity, trend persistence,
        overnight gap exposure).
      * The day-trade verdict is ONLY a tradability screen — "is this stock
        liquid and wide-ranging enough that intraday traders can work it at
        all" — and is explicitly NOT a signal, edge, or entry. Nothing in
        this system has ever been tested at an intraday horizon, and daily
        bars cannot test one. A stock passing this screen means day traders
        COULD trade it, never that this system thinks you SHOULD.

    Returns both verdicts separately plus a plain-English primary
    recommendation, so the two can never be read as one blended score.
    """
    reasons_day, reasons_swing = [], []

    # --- Day-trade tradability screen (liquidity + range + price) ---------
    day_checks = []
    if avg_dollar_volume is not None:
        ok = avg_dollar_volume >= _DAY_MIN_DOLLAR_VOLUME
        day_checks.append(ok)
        reasons_day.append(
            f"Liquidity ${avg_dollar_volume / 1e6:,.0f}M/day "
            f"{'clears' if ok else 'is below'} the ${_DAY_MIN_DOLLAR_VOLUME / 1e6:,.0f}M intraday bar"
        )
    if avg_daily_range_pct is not None:
        ok = avg_daily_range_pct >= _DAY_MIN_DAILY_RANGE_PCT
        day_checks.append(ok)
        reasons_day.append(
            f"Average daily range {avg_daily_range_pct:.2f}% "
            f"{'clears' if ok else 'is below'} the {_DAY_MIN_DAILY_RANGE_PCT:.1f}% minimum "
            "(full-session high-to-low — an upper bound no real intraday entry captures in full)"
        )
    if last_price is not None:
        ok = last_price >= _DAY_MIN_PRICE
        day_checks.append(ok)
        if not ok:
            reasons_day.append(f"Price ${last_price:.2f} is under ${_DAY_MIN_PRICE:.0f} — spreads eat intraday round trips")
    if gap_share_of_range is not None and gap_share_of_range > _HIGH_GAP_SHARE:
        reasons_day.append(
            f"{gap_share_of_range:.0%} of this stock's movement happens OVERNIGHT — a day trader who is "
            "flat at the close structurally cannot capture that part of the move"
        )

    day_verdict = "UNKNOWN" if not day_checks else ("TRADABLE_INTRADAY" if all(day_checks) else "NOT_LIQUID_ENOUGH_INTRADAY")

    # --- Swing-horizon fit (the horizon this system actually models) ------
    swing_points = []
    if avg_dollar_volume is not None:
        ok = avg_dollar_volume >= _SWING_MIN_DOLLAR_VOLUME
        swing_points.append(ok)
        reasons_swing.append(
            f"Liquidity ${avg_dollar_volume / 1e6:,.0f}M/day {'is ample' if ok else 'is thin'} for a multi-day hold"
        )
    if adx14 is not None:
        trending = adx14 >= _SWING_TRENDING_ADX
        swing_points.append(trending)
        if trending:
            adx_read = "a real trend is present (Wilder: ADX above 25)"
        elif adx14 >= _SWING_CHOPPY_ADX:
            adx_read = "borderline — 20-25 is the ambiguous band, neither clearly trending nor clearly choppy"
        else:
            adx_read = "no trend; below 20 is choppy/range-bound"
        reasons_swing.append(
            f"ADX14 {adx14:.1f} — {adx_read}, and a multi-day hold depends on trend persistence"
        )
    if efficiency_ratio is not None:
        ok = efficiency_ratio >= _SWING_MIN_EFFICIENCY_RATIO
        swing_points.append(ok)
        reasons_swing.append(
            f"Kaufman Efficiency Ratio {efficiency_ratio:.2f} — {efficiency_ratio:.0%} of the last 20 days' "
            f"total up-and-down travel actually translated into net directional movement, "
            f"{'a reasonably clean path to ride' if ok else 'meaning most of the motion cancelled itself out (whipsaw)'}"
        )
    if avg_overnight_gap_pct is not None:
        gap_ok = gap_share_of_range is None or gap_share_of_range <= _HIGH_GAP_SHARE
        swing_points.append(gap_ok)
        reasons_swing.append(
            f"Overnight gaps average {avg_overnight_gap_pct:.2f}%"
            + ("" if gap_ok else f" and are {gap_share_of_range:.0%} of all movement — holding through them is the main risk of this trade")
        )

    if not swing_points:
        swing_verdict = "UNKNOWN"
    elif all(swing_points):
        swing_verdict = "GOOD_SWING_FIT"
    elif sum(swing_points) >= len(swing_points) - 1:
        swing_verdict = "WORKABLE_SWING_FIT"
    else:
        swing_verdict = "POOR_SWING_FIT"

    # --- Primary recommendation -------------------------------------------
    if swing_verdict in ("GOOD_SWING_FIT", "WORKABLE_SWING_FIT"):
        primary = "SWING"
        rationale = (
            "and note this is the only horizon this system has evidence for: the ranking that "
            "surfaced this candidate is a 20-trading-day sector signal, so the intended hold is weeks, "
            "not hours or months."
        )
    elif swing_verdict == "POOR_SWING_FIT" and day_verdict == "TRADABLE_INTRADAY":
        primary = "NEITHER_WELL_SUITED"
        rationale = (
            "This stock fits the swing horizon poorly (thin trend and/or heavy overnight gap exposure), and "
            "while it is liquid enough that day traders could work it intraday, this system has NO tested "
            "intraday edge to offer — so it has nothing useful to say about trading it that way."
        )
    else:
        primary = "NEITHER_WELL_SUITED"
        rationale = (
            "Fits neither horizon well on these screens — thin trend/liquidity for a multi-day hold, and "
            "not liquid or wide-ranging enough for intraday work either."
        )

    return {
        "primary_recommendation": primary,
        "rationale": rationale,
        "swing_fit": swing_verdict,
        "swing_reasons": reasons_swing,
        "day_trade_screen": day_verdict,
        "day_trade_reasons": reasons_day,
        "day_trade_caveat": (
            "DAY-TRADE SCREEN ONLY — NOT A SIGNAL. This system holds daily bars exclusively and has never "
            "tested anything at an intraday horizon; a PASS here means the stock is liquid and wide-ranging "
            "enough for intraday traders to operate in, nothing more. The system's one validated edge "
            "(sector_prediction_model) is a 20-trading-day signal and says nothing whatsoever about "
            "what happens between the open and the close."
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
    direction: str = "long",
) -> dict:
    """Entry/stop/take-profit levels for a hypothetical LONG or SHORT position,
    derived purely from price and ATR — not a recommendation to take the trade,
    just the deterministic arithmetic once you decide to. entry is the last
    available close (this system has no live quote, see data_freshness); stop
    is the same ATR-multiple stop as suggested_stop() (below entry for long,
    above entry for short); TP1/TP2 are R-multiples of the resulting
    entry-to-stop distance (a common, but arbitrary, convention — 1.5R and 3R
    are not fitted to this specific ticker's behavior). risk_reward_ratio
    compares TP1 to the stop distance. This function does not know direction
    is a good idea — it only computes what the numbers would be IF that
    position were opened here. SHORT-specific costs (borrow fees, hard-to-
    borrow/unavailability risk, dividend obligations while short, theoretically
    unlimited loss vs. a long's floor at zero) are NOT modeled — flagged below,
    not silently ignored."""
    if last_price is None or atr14 is None or atr14 <= 0:
        return {"error": "last_price and a positive atr14 are required"}
    if direction not in ("long", "short"):
        return {"error": f"direction must be 'long' or 'short', got {direction!r}"}

    sign = 1 if direction == "long" else -1
    stop = round(last_price - sign * stop_multiplier * atr14, 4)
    risk_per_share = round(sign * (last_price - stop), 4)
    if risk_per_share <= 0:
        return {"error": "computed stop is on the wrong side of last_price — cannot derive R-multiples"}

    tp1 = round(last_price + sign * tp1_r_multiple * risk_per_share, 4)
    tp2 = round(last_price + sign * tp2_r_multiple * risk_per_share, 4)
    # A long's stop sits BELOW entry (invalidated by a close below it); a
    # short's stop sits ABOVE entry (invalidated by a close above it).
    breach = "below" if direction == "long" else "above"

    out = {
        "direction": direction.upper(),
        "entry": last_price,
        "stop": stop,
        "risk_per_share": risk_per_share,
        "take_profit_1": tp1,
        "take_profit_1_r_multiple": tp1_r_multiple,
        "take_profit_2": tp2,
        "take_profit_2_r_multiple": tp2_r_multiple,
        "risk_reward_ratio_tp1": tp1_r_multiple,  # by construction: TP1 is exactly tp1_r_multiple * risk
        "invalidation": f"Daily close {breach} {stop} invalidates this {direction} setup (stop-loss level breached).",
        # Does not alter entry/stop/TP1/TP2 above — a separate, additive warning
        # about execution risk around an earnings gap, not a recomputation.
        "earnings_proximity": earnings_proximity_risk(as_of, next_earnings_date),
        "note": (
            "entry is the latest available daily close, not a live executable price — "
            "there may be gap risk between this level and your actual fill. stop/TP1/TP2 "
            f"are {stop_multiplier}x/{tp1_r_multiple}R/{tp2_r_multiple}R ATR-based levels, a "
            "common but arbitrary convention, not calibrated to this specific ticker's "
            f"historical behavior around similar setups. This assumes a {direction.upper()} "
            "position. Whether to take this trade at all is a separate question this "
            "function does not answer; it only computes the arithmetic for a hypothetical "
            f"{direction} entry at the current level."
        ),
    }
    if direction == "short":
        out["short_specific_costs_not_modeled"] = (
            "Borrow fees, hard-to-borrow/unavailability risk, and dividend obligations while short are "
            "NOT included in risk_per_share or the R-multiples above — actual short P&L is worse than "
            "these numbers by whatever the broker charges to borrow the shares. Loss on a short is also "
            "theoretically unbounded (price can rise without limit), unlike a long's floor at zero — the "
            "stop above is a discipline level, not a guaranteed cap on loss if price gaps past it."
        )
    return out


