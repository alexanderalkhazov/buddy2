"""Technical indicators computed from an OHLCV frame (pandas only, no TA-Lib)."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + num_std * std, "bb_lower": mid - num_std * std})


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift()
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    low_min = df["low"].rolling(k_window).min()
    high_max = df["high"].rolling(k_window).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    return pd.DataFrame({"stoch_k": k, "stoch_d": k.rolling(d_window).mean()})


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average Directional Index (Wilder) — trend strength, 0-100, direction-agnostic."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    smoothed_tr = _true_range(df).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / smoothed_tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / smoothed_tr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def donchian(df: pd.DataFrame, entry_window: int = 20, exit_window: int = 10) -> pd.DataFrame:
    """Turtle-style channel. Shifted one bar so a breakout compares against prior bars only."""
    return pd.DataFrame(
        {
            "donchian_upper20": df["high"].rolling(entry_window).max().shift(1),
            "donchian_lower10": df["low"].rolling(exit_window).min().shift(1),
        }
    )


def rate_of_change(close: pd.Series, window: int = 10) -> pd.Series:
    return (close / close.shift(window) - 1) * 100


def bollinger_pct_b(close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
    return (close - lower) / (upper - lower)


def ichimoku(df: pd.DataFrame, tenkan_window: int = 9, kijun_window: int = 26, senkou_b_window: int = 52) -> pd.DataFrame:
    """Ichimoku Kinko Hyo (Ichimoku Cloud) — standard periods (9/26/52).

    Tenkan-sen (conversion) and Kijun-sen (base) are simple midpoints of
    their own trailing high/low window — no lookahead. Senkou Span A/B (the
    cloud edges) are conventionally PLOTTED 26 periods ahead of the data
    that produced them; implemented here as `.shift(kijun_window)` so that
    row t's cloud value is exactly what was computed from data as of
    t-26 — the real-world plotting convention, and still strictly
    point-in-time (it uses only older data, never future data). Chikou
    Span (the lagging span, close shifted BACKWARD) is a plotting/visual
    convenience only — it is NOT included in any computed signal here,
    since "shift close by -26" only makes sense for a chart, and including
    it in a live snapshot would just be tomorrow's close under another
    name if naively read as "current."
    """
    high, low, close = df["high"], df["low"], df["close"]
    tenkan = (high.rolling(tenkan_window).max() + low.rolling(tenkan_window).min()) / 2
    kijun = (high.rolling(kijun_window).max() + low.rolling(kijun_window).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_window)
    senkou_b = ((high.rolling(senkou_b_window).max() + low.rolling(senkou_b_window).min()) / 2).shift(kijun_window)
    return pd.DataFrame(
        {
            "ichimoku_tenkan": tenkan,
            "ichimoku_kijun": kijun,
            "ichimoku_senkou_a": senkou_a,
            "ichimoku_senkou_b": senkou_b,
        }
    )


def chandelier_exit(df: pd.DataFrame, window: int = 22, mult: float = 3.0) -> pd.DataFrame:
    """Chandelier Exit (Charles Le Beau) — a volatility-adaptive TRAILING STOP,
    not a directional signal: tightens in calm markets, widens in volatile
    ones. chandelier_long is the stop level for an existing/hypothetical
    LONG position; chandelier_short mirrors it for a SHORT. Distinct from
    the flat stop_multiplier*ATR stop in processing/scoring.py:trade_levels()
    (which anchors off the ENTRY price) — this anchors off the highest high
    (long) / lowest low (short) of the trailing window, the standard
    professional convention for a trend-following trailing stop."""
    high, low = df["high"], df["low"]
    atr_n = atr(df, window=window)
    return pd.DataFrame(
        {
            "chandelier_long_stop": high.rolling(window).max() - mult * atr_n,
            "chandelier_short_stop": low.rolling(window).min() + mult * atr_n,
        }
    )


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Commodity Channel Index — deviation of the typical price from its own
    moving average, scaled by mean absolute deviation (not RSI/Stochastic's
    high-low-range normalization, so it's a genuinely distinct oscillator
    family). >+100 = strong uptrend/overbought; <-100 = strong downtrend/
    oversold."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical_price.rolling(window).mean()
    mean_dev = typical_price.rolling(window).apply(lambda x: (x - x.mean()).abs().mean(), raw=False)
    return (typical_price - sma_tp) / (0.015 * mean_dev)


def money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Money Flow Index — RSI's volume-weighted cousin, and the only oscillator
    in this system that incorporates volume rather than price alone. Typical
    price * volume is "money flow" for the bar; MFI is the RSI formula applied
    to money flow instead of raw price change. Same 0-100 scale and >70/<30
    overbought/oversold convention as RSI, but a strong RSI reading on THIN
    volume reads very differently from the same RSI reading on HEAVY volume —
    MFI is what actually tells the two apart. Backward-only (rolling sums of
    already-elapsed bars), no lookahead."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]
    direction = typical_price.diff()
    positive_flow = money_flow.where(direction > 0, 0.0).rolling(window).sum()
    negative_flow = money_flow.where(direction < 0, 0.0).rolling(window).sum()
    money_ratio = positive_flow / negative_flow
    return 100 - 100 / (1 + money_ratio)


def aroon(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """Aroon Up/Down/Oscillator — how many periods (as a % of the window)
    since the most recent N-period high/low, NOT how far price has moved.
    This is a genuinely distinct trend-strength family from ADX: ADX measures
    the MAGNITUDE of directional movement, Aroon measures its RECENCY — a
    stock making a new high every few days (Aroon Up near 100) can have a
    completely different Aroon read than one that made one big move a month
    ago and has drifted sideways since, even if both show similar ADX.
    aroon_oscillator = aroon_up - aroon_down, ranging -100..+100."""
    high, low = df["high"], df["low"]
    periods_since_high = high.rolling(window + 1).apply(lambda x: window - x.argmax(), raw=True)
    periods_since_low = low.rolling(window + 1).apply(lambda x: window - x.argmin(), raw=True)
    aroon_up = 100 * (window - periods_since_high) / window
    aroon_down = 100 * (window - periods_since_low) / window
    return pd.DataFrame({"aroon_up": aroon_up, "aroon_down": aroon_down, "aroon_oscillator": aroon_up - aroon_down})


def keltner_channels(df: pd.DataFrame, window: int = 20, atr_mult: float = 2.0) -> pd.DataFrame:
    """Keltner Channels — a volatility band built on ATR around an EMA
    midline, not Bollinger's standard deviation around an SMA. This is a
    real, not cosmetic, difference: ATR is a range-based volatility measure
    (insensitive to a single huge close-to-close jump the way std-dev isn't),
    so Keltner and Bollinger can disagree meaningfully on how "extended" the
    same move looks — the two together are more informative than either
    alone. keltner_pctk mirrors bollinger_pct_b's 0=lower/1=upper convention
    for direct comparison."""
    mid = df["close"].ewm(span=window, adjust=False).mean()
    atr_n = atr(df, window=window)
    upper = mid + atr_mult * atr_n
    lower = mid - atr_mult * atr_n
    pctk = (df["close"] - lower) / (upper - lower)
    return pd.DataFrame({"keltner_upper": upper, "keltner_mid": mid, "keltner_lower": lower, "keltner_pctk": pctk})


def ichimoku_cloud_position(close: float, senkou_a: float | None, senkou_b: float | None) -> str:
    if senkou_a is None or senkou_b is None:
        return "UNKNOWN"
    cloud_top, cloud_bottom = max(senkou_a, senkou_b), min(senkou_a, senkou_b)
    if close > cloud_top:
        return "above_cloud"
    if close < cloud_bottom:
        return "below_cloud"
    return "in_cloud"


def tradability_stats(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Liquidity / range / gap measurements used to judge WHICH TRADE STYLE a
    stock suits — all computed from daily bars only, which is exactly the
    limitation that matters here (see processing/scoring.py:trade_style_fit).

    - avg_dollar_volume: close * volume, averaged — the standard liquidity
      screen. Day trading needs far more of it than swing trading, because
      intraday entries/exits pay the spread repeatedly.
    - avg_daily_range_pct: mean (high - low) / close. From daily bars this is
      the FULL-SESSION range, i.e. the theoretical maximum a perfectly-timed
      intraday trade could capture — NOT the range actually available to a
      real day trader, who cannot buy the low and sell the high. Treat it as
      an upper bound on intraday opportunity, never as expected profit.
    - overnight_gap_pct: mean |open - prev_close| / prev_close. This is the
      part of a stock's movement that happens while the market is CLOSED —
      it is unreachable for a day trader (who is flat overnight) and is pure
      unhedgeable risk for a swing trader (who holds through it).
    - gap_share_of_range: what fraction of total movement happens overnight
      rather than intraday. High values mean the stock mostly moves on
      news/gaps, which argues against day trading it and raises the risk of
      holding it overnight.
    """
    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]
    prev_close = close.shift()

    out = pd.DataFrame(index=df.index)
    out["dollar_volume"] = close * df["volume"]
    out["avg_dollar_volume"] = out["dollar_volume"].rolling(window).mean()
    out["daily_range_pct"] = (high - low) / close * 100
    out["avg_daily_range_pct"] = out["daily_range_pct"].rolling(window).mean()
    out["gap_pct"] = (open_ - prev_close).abs() / prev_close * 100
    out["avg_overnight_gap_pct"] = out["gap_pct"].rolling(window).mean()

    total_move = out["avg_daily_range_pct"] + out["avg_overnight_gap_pct"]
    out["gap_share_of_range"] = (out["avg_overnight_gap_pct"] / total_move).where(total_move > 0)

    # Kaufman Efficiency Ratio: net directional travel / total path length over
    # the window. 1.0 = a perfectly straight move, near 0 = pure whipsaw. It is
    # the standard daily-bar proxy for whether a trend is clean enough to hold.
    net_move = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    out["efficiency_ratio"] = (net_move / path_length).where(path_length > 0)

    # --- Own-history-normalized "is TODAY typical or a spike" signals ------
    # Everything above is an UNCONDITIONAL 20-day average — it says whether a
    # stock is generically wide-ranging/liquid, not whether TODAY specifically
    # is a normal day or an outlier for THIS stock. These three answer that,
    # each against the stock's OWN trailing history (never a fixed universal
    # threshold), so trade_style_fit() can tell "systemically day-tradable"
    # apart from "today happens to be an unusual day for a normally-quiet
    # name" — two different claims that get conflated if only cross-sectional
    # thresholds are used.
    if "atr14" in df.columns:
        atr_pct = df["atr14"] / close * 100
        out["atr_pct"] = atr_pct
        # Percentile rank of TODAY's ATR% within its own trailing 252-day
        # (≈1 trading year) history — min_periods=100 so a young ticker
        # reports None rather than a percentile computed against too little
        # history to mean anything.
        out["atr_pct_percentile_252d"] = atr_pct.rolling(252, min_periods=100).apply(
            lambda x: (x <= x[-1]).mean(), raw=True
        )

    # Relative Volume (RVOL): today's volume vs. the mean of the PRIOR 20
    # days (today itself excluded from its own baseline via shift(1), so
    # this can't be inflated by including the very day it's measuring).
    vol_baseline = df["volume"].shift(1).rolling(window).mean()
    out["rvol_20d"] = (df["volume"] / vol_baseline).where(vol_baseline > 0)

    # 52-week high/low proximity — a structural "room to run" / trend-
    # context read, not a volatility or liquidity measure like the rest of
    # this function. Uses the actual traded high/low (not just closes), the
    # standard "52-week high" convention. <=0 exactly at a new high/low.
    high_252 = df["high"].rolling(252, min_periods=50).max()
    low_252 = df["low"].rolling(252, min_periods=50).min()
    out["pct_from_52w_high"] = (close - high_252) / high_252 * 100
    out["pct_from_52w_low"] = (close - low_252) / low_252 * 100

    return out


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all indicator columns to a copy of the OHLCV frame."""
    out = df.copy()
    close = out["close"]
    out["rsi14"] = rsi(close)
    out["rsi2"] = rsi(close, window=2)
    out = out.join(macd(close))
    for w in (20, 50, 200):
        out[f"sma{w}"] = sma(close, w)
    out = out.join(bollinger(close))
    out["atr14"] = atr(out)
    out["vol_avg30"] = out["volume"].rolling(30).mean()
    out = out.join(stochastic(out))
    out["adx14"] = adx(out)
    out = out.join(donchian(out))
    out["roc10"] = rate_of_change(close)
    out["bb_pctb"] = bollinger_pct_b(close, out["bb_upper"], out["bb_lower"])
    out = out.join(ichimoku(out))
    out = out.join(chandelier_exit(out))
    out["cci20"] = cci(out)
    out["mfi14"] = money_flow_index(out)
    out = out.join(aroon(out))
    out = out.join(keltner_channels(out))
    out = out.join(tradability_stats(out))
    return out


def _f(value) -> float | None:
    """Cast to a plain float, mapping NaN to None so the bundle stays JSON-clean."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


@dataclass
class PriceSnapshot:
    ticker: str
    as_of: str
    last: float | None
    change_pct: float | None
    rsi14: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    sma20: float | None
    sma50: float | None
    sma200: float | None
    bb_upper: float | None
    bb_lower: float | None
    atr14: float | None
    volume: float | None
    vol_vs_30d_avg: float | None
    trend: str
    # Computed in enrich() all along but not previously surfaced past it —
    # Bollinger %B, Stochastic, ADX, and ROC10 were being thrown away.
    bb_pctb: float | None = None
    stoch_k: float | None = None
    stoch_d: float | None = None
    adx14: float | None = None
    roc10: float | None = None
    technical_composite: float | None = None
    ichimoku_tenkan: float | None = None
    ichimoku_kijun: float | None = None
    ichimoku_senkou_a: float | None = None
    ichimoku_senkou_b: float | None = None
    ichimoku_cloud_position: str = "UNKNOWN"
    chandelier_long_stop: float | None = None
    chandelier_short_stop: float | None = None
    cci20: float | None = None
    # Trade-style suitability inputs (daily-bar tradability stats).
    avg_dollar_volume: float | None = None
    avg_daily_range_pct: float | None = None
    avg_overnight_gap_pct: float | None = None
    gap_share_of_range: float | None = None
    efficiency_ratio: float | None = None
    # Own-history-normalized "is today typical or a spike" signals — see
    # tradability_stats() docstring.
    atr_pct_percentile_252d: float | None = None
    rvol_20d: float | None = None
    pct_from_52w_high: float | None = None
    pct_from_52w_low: float | None = None
    # Volume-weighted momentum (MFI), time-based trend (Aroon), and an
    # ATR-based volatility band (Keltner) — three more genuinely distinct
    # indicator families, not variations on ones already above.
    mfi14: float | None = None
    aroon_up: float | None = None
    aroon_down: float | None = None
    aroon_oscillator: float | None = None
    keltner_pctk: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def technical_composite_score(
    rsi14: float | None,
    stoch_k: float | None,
    adx14: float | None,
    bb_pctb: float | None,
    roc10: float | None,
    macd_hist: float | None,
    cci20: float | None = None,
    ichimoku_cloud: str | None = None,
    mfi14: float | None = None,
    aroon_oscillator: float | None = None,
    keltner_pctk: float | None = None,
) -> float | None:
    """A single 0-100 technical-strength read, built from up to TEN distinct
    indicator families (momentum oscillator, stochastic, trend strength,
    mean-reversion band position, rate-of-change, CCI, Ichimoku cloud
    position, volume-weighted momentum, time-based trend, and an ATR-based
    volatility band) rather than one signal — this is deliberately not "RSI
    alone" or "trend alone." Each component is scored independently, then
    combined with fixed weights; missing components are excluded and the
    rest renormalized (never treated as 0 or a neutral 50, which would
    silently understate real signal when data is genuinely available for
    the other components).

    This does NOT predict future returns — it is a same-family cousin of the
    Baseline Score (processing/scoring.py), i.e. a transparent, deterministic
    read of the CURRENT technical state, not a validated forecast. Whether
    a high reading here corresponds to good forward performance is an
    empirical question this function does not answer; see the ML/sector_prediction_model
    modules for the parts of this system that are actually walk-forward tested.

    Weights: momentum(RSI) 15%, stochastic 10%, trend strength(ADX, direction
    -agnostic — weighted by whether price is currently rising, via ROC) 12%,
    Bollinger %B (mean-reversion position within the band) 10%, MACD
    histogram sign 8%, CCI 10%, Ichimoku cloud position 8%, Money Flow
    Index (volume-weighted momentum) 10%, Aroon Oscillator (time-since-
    extreme trend measure) 10%, Keltner %K (ATR-based band position) 7%.
    """
    components: dict[str, float] = {}

    if rsi14 is not None:
        # 50 neutral, extremes (very overbought/oversold) pulled toward 50
        # not because "extreme is bad" universally, but because RSI alone
        # is a momentum oscillator, not a directional signal — its raw
        # distance from 50 is what this component measures.
        components["momentum"] = _clip(50 + (rsi14 - 50) * 1.0, 0, 100)

    if stoch_k is not None:
        components["stochastic"] = _clip(stoch_k, 0, 100)

    if adx14 is not None and roc10 is not None:
        # ADX alone is direction-agnostic (a strong downtrend scores just as
        # high as a strong uptrend) — multiply by the sign of recent ROC so
        # a high score here specifically means "strongly trending UP," not
        # just "strongly trending."
        trend_strength = _clip(adx14 * 2, 0, 100)  # ADX rarely exceeds 50 in practice
        # Center 50, scaled by ADX magnitude, signed by ROC direction.
        components["trend_strength_directional"] = _clip(50 + (trend_strength / 2) * (1 if roc10 >= 0 else -1), 0, 100)

    if bb_pctb is not None:
        # %B: 0 = at lower band, 1 = at upper band, 0.5 = at the mid/mean.
        # Scored as "distance above the mean," not "closer to upper band is
        # better" — this is a position read, not a claim that touching the
        # upper band predicts further upside (mean-reversion research would
        # argue the opposite).
        components["band_position"] = _clip(bb_pctb * 100, 0, 100)

    if macd_hist is not None:
        components["macd_sign"] = 65.0 if macd_hist > 0 else 35.0

    if cci20 is not None:
        # CCI is unbounded (typically -200..+200) — map through a soft clip
        # centered on 0=neutral(50), +/-150 saturating the 0-100 range. A
        # distinct family from RSI/Stochastic: deviation from a moving
        # average scaled by mean absolute deviation, not a high-low range.
        components["cci"] = _clip(50 + cci20 / 3, 0, 100)

    if ichimoku_cloud is not None and ichimoku_cloud != "UNKNOWN":
        components["ichimoku_cloud"] = {"above_cloud": 70.0, "in_cloud": 50.0, "below_cloud": 30.0}.get(ichimoku_cloud, 50.0)

    if mfi14 is not None:
        # Same 0-100 scale/overbought-oversold convention as RSI, but
        # volume-weighted — a genuinely distinct read, not RSI twice.
        components["mfi"] = _clip(50 + (mfi14 - 50) * 1.0, 0, 100)

    if aroon_oscillator is not None:
        # -100..+100, already centered — a strong positive reading means a
        # new high happened very recently (Aroon Up near 100, Aroon Down
        # near 0), which is a RECENCY read, not a magnitude read like ADX.
        components["aroon"] = _clip(50 + aroon_oscillator / 2, 0, 100)

    if keltner_pctk is not None:
        # Same 0=lower/1=upper band-position convention as Bollinger %B, but
        # against an ATR-based band instead of a std-dev based one — the two
        # can and do disagree on how extended the same move looks.
        components["keltner"] = _clip(keltner_pctk * 100, 0, 100)

    if not components:
        return None

    weights = {
        "momentum": 0.15, "stochastic": 0.10, "trend_strength_directional": 0.12,
        "band_position": 0.10, "macd_sign": 0.08, "cci": 0.10, "ichimoku_cloud": 0.08,
        "mfi": 0.10, "aroon": 0.10, "keltner": 0.07,
    }
    weight_sum = sum(weights[k] for k in components)
    return round(sum(weights[k] * v for k, v in components.items()) / weight_sum, 2)


def _trend_label(last: float, s20, s50, s200) -> str:
    above = [name for name, val in (("20", s20), ("50", s50), ("200", s200))
             if val is not None and last > val]
    if len(above) == 3:
        return "above_all_smas"
    if not above:
        return "below_all_smas"
    return "above_" + "_".join(f"{n}sma" for n in above)


def snapshot(ticker: str, df: pd.DataFrame) -> PriceSnapshot:
    """Reduce an enriched frame to the compact last-bar summary the LLM sees.

    yfinance occasionally returns a trailing row with open/high/low/volume populated
    but close as NaN (a partial-day or data-provider artifact) — using it directly
    would silently propagate None/NaN through last price, change_pct, and the trend
    label. Drop trailing NaN-close rows and use the most recent complete bar instead;
    its own indicators are unaffected since rolling/ewm windows don't look ahead.
    """
    enriched = enrich(df)
    valid = enriched.dropna(subset=["close"])
    if valid.empty:
        raise ValueError(f"no bar with a valid close price for {ticker}")

    row = valid.iloc[-1]
    prev_close = valid["close"].iloc[-2] if len(valid) > 1 else None
    last = float(row["close"])

    return PriceSnapshot(
        ticker=ticker.upper(),
        as_of=str(valid.index[-1].date()),
        last=_f(last),
        change_pct=_f((last / prev_close - 1) * 100) if prev_close else None,
        rsi14=_f(row["rsi14"]),
        macd=_f(row["macd"]),
        macd_signal=_f(row["signal"]),
        macd_hist=_f(row["hist"]),
        sma20=_f(row["sma20"]),
        sma50=_f(row["sma50"]),
        sma200=_f(row["sma200"]),
        bb_upper=_f(row["bb_upper"]),
        bb_lower=_f(row["bb_lower"]),
        atr14=_f(row["atr14"]),
        volume=_f(row["volume"]),
        # Index/rate tickers (^VIX, ^TNX, etc.) commonly report volume=0, making
        # this a 0/0 division — not a NaN input, so the pd.isna guard alone lets it
        # through and numpy warns on the resulting NaN. Guard the zero case too.
        vol_vs_30d_avg=(
            _f(row["volume"] / row["vol_avg30"])
            if not pd.isna(row["vol_avg30"]) and row["vol_avg30"] != 0
            else None
        ),
        trend=_trend_label(last, _f(row["sma20"]), _f(row["sma50"]), _f(row["sma200"])),
        bb_pctb=_f(row["bb_pctb"]),
        stoch_k=_f(row["stoch_k"]),
        stoch_d=_f(row["stoch_d"]),
        adx14=_f(row["adx14"]),
        roc10=_f(row["roc10"]),
        ichimoku_tenkan=_f(row["ichimoku_tenkan"]),
        ichimoku_kijun=_f(row["ichimoku_kijun"]),
        ichimoku_senkou_a=_f(row["ichimoku_senkou_a"]),
        ichimoku_senkou_b=_f(row["ichimoku_senkou_b"]),
        ichimoku_cloud_position=ichimoku_cloud_position(last, _f(row["ichimoku_senkou_a"]), _f(row["ichimoku_senkou_b"])),
        chandelier_long_stop=_f(row["chandelier_long_stop"]),
        chandelier_short_stop=_f(row["chandelier_short_stop"]),
        cci20=_f(row["cci20"]),
        avg_dollar_volume=_f(row["avg_dollar_volume"]),
        avg_daily_range_pct=_f(row["avg_daily_range_pct"]),
        avg_overnight_gap_pct=_f(row["avg_overnight_gap_pct"]),
        gap_share_of_range=_f(row["gap_share_of_range"]),
        efficiency_ratio=_f(row["efficiency_ratio"]),
        atr_pct_percentile_252d=_f(row.get("atr_pct_percentile_252d")),
        rvol_20d=_f(row.get("rvol_20d")),
        pct_from_52w_high=_f(row.get("pct_from_52w_high")),
        pct_from_52w_low=_f(row.get("pct_from_52w_low")),
        mfi14=_f(row["mfi14"]),
        aroon_up=_f(row["aroon_up"]),
        aroon_down=_f(row["aroon_down"]),
        aroon_oscillator=_f(row["aroon_oscillator"]),
        keltner_pctk=_f(row["keltner_pctk"]),
        technical_composite=technical_composite_score(
            rsi14=_f(row["rsi14"]),
            stoch_k=_f(row["stoch_k"]),
            adx14=_f(row["adx14"]),
            bb_pctb=_f(row["bb_pctb"]),
            roc10=_f(row["roc10"]),
            macd_hist=_f(row["hist"]),
            cci20=_f(row["cci20"]),
            ichimoku_cloud=ichimoku_cloud_position(last, _f(row["ichimoku_senkou_a"]), _f(row["ichimoku_senkou_b"])),
            mfi14=_f(row["mfi14"]),
            aroon_oscillator=_f(row["aroon_oscillator"]),
            keltner_pctk=_f(row["keltner_pctk"]),
        ),
    )
