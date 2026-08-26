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


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * df["volume"]).cumsum()


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
    out["obv"] = on_balance_volume(out)
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

    def to_dict(self) -> dict:
        return asdict(self)


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
    )
