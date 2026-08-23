"""A small library of well-known, published algorithmic trading strategies.

These exist so the model screens real, well-understood systems instead of inventing
an arbitrary threshold (e.g. "RSI < 30") from scratch every time. Each entry is a
plain StrategySpec dict plus a one-line note on the market regime it fits and its
known failure mode — the model is expected to surface that note, not hide it.
"""

from __future__ import annotations

TEMPLATES: dict[str, dict] = {
    "donchian_breakout": {
        "label": "Donchian Breakout (Turtle-style trend following)",
        "regime": "Strongly trending markets. Whipsaws and bleeds small losses in "
        "sideways/choppy conditions — the classic trend-follower tradeoff.",
        "spec": {
            "name": "Donchian Breakout",
            "entry_rules": [{"left": "close", "op": ">", "right": "donchian_upper20"}],
            "exit_rules": [{"left": "close", "op": "<", "right": "donchian_lower10"}],
            "stop_loss_pct": 0.10,
        },
    },
    "bollinger_mean_reversion": {
        "label": "Bollinger Band Mean Reversion",
        "regime": "Range-bound markets. Gets run over in a sustained trend — the band "
        "keeps expanding and the position keeps averaging down.",
        "spec": {
            "name": "Bollinger Mean Reversion",
            "entry_rules": [
                {"left": "close", "op": "<", "right": "bb_lower"},
                {"left": "rsi14", "op": "<", "right": 35},
            ],
            "entry_logic": "all",
            "exit_rules": [{"left": "close", "op": ">", "right": "bb_mid"}],
            "exit_logic": "any",
            "stop_loss_pct": 0.07,
        },
    },
    "macd_trend": {
        "label": "MACD Trend Following (200-SMA filtered)",
        "regime": "Established uptrends. The 200-SMA filter cuts counter-trend "
        "whipsaws but adds lag — entries come well after the bottom.",
        "spec": {
            "name": "MACD Trend",
            "entry_rules": [
                {"left": "macd", "op": "crosses_above", "right": "signal"},
                {"left": "close", "op": ">", "right": "sma200"},
            ],
            "entry_logic": "all",
            "exit_rules": [{"left": "macd", "op": "crosses_below", "right": "signal"}],
            "stop_loss_pct": 0.08,
        },
    },
    "dual_ma_adx": {
        "label": "Golden Cross filtered by ADX (trend-strength gate)",
        "regime": "Works better than a naive MA cross in chop because ADX > 20 filters "
        "out weak/no-trend periods, but still lags at trend turns.",
        "spec": {
            "name": "Dual MA + ADX filter",
            "entry_rules": [
                {"left": "sma50", "op": "crosses_above", "right": "sma200"},
                {"left": "adx14", "op": ">", "right": 20},
            ],
            "entry_logic": "all",
            "exit_rules": [{"left": "sma50", "op": "crosses_below", "right": "sma200"}],
            "stop_loss_pct": 0.12,
            "trailing_stop": True,
        },
    },
    "rsi2_connors": {
        "label": "RSI(2) Mean Reversion (Larry Connors)",
        "regime": "Short-term pullbacks within a longer uptrend (200-SMA filter). High "
        "win rate, small edge per trade — sensitive to slippage and fees; do not run "
        "this at retail commission rates without checking net-of-cost numbers.",
        "spec": {
            "name": "RSI(2) Connors",
            "entry_rules": [
                {"left": "rsi2", "op": "<", "right": 10},
                {"left": "close", "op": ">", "right": "sma200"},
            ],
            "entry_logic": "all",
            "exit_rules": [{"left": "rsi2", "op": ">", "right": 70}],
            "stop_loss_pct": 0.05,
        },
    },
    "stochastic_momentum": {
        "label": "Stochastic Momentum (oversold crossover)",
        "regime": "Choppy-to-trending markets with clear swing structure. Prone to "
        "false signals in strong one-directional trends where oscillators stay pinned.",
        "spec": {
            "name": "Stochastic Momentum",
            "entry_rules": [
                {"left": "stoch_k", "op": "crosses_above", "right": "stoch_d"},
                {"left": "stoch_k", "op": "<", "right": 30},
            ],
            "entry_logic": "all",
            "exit_rules": [{"left": "stoch_k", "op": ">", "right": 80}],
            "stop_loss_pct": 0.06,
        },
    },
}


def list_templates() -> list[dict]:
    return [
        {"template": name, "label": t["label"], "regime": t["regime"]}
        for name, t in TEMPLATES.items()
    ]


def get_spec(name: str) -> dict:
    template = TEMPLATES.get(name)
    if template is None:
        known = ", ".join(TEMPLATES)
        raise ValueError(f"unknown strategy template {name!r}. Known templates: {known}")
    return template["spec"]
