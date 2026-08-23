"""Background refresh and alerting via APScheduler.

Alerts are threshold rules evaluated against the same context bundle the LLM sees, so
an alert always corresponds to something the assistant can explain.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from rich.console import Console

from processing.bundle import build

console = Console()


@dataclass
class Alert:
    name: str
    predicate: Callable[[dict], bool]
    message: str


DEFAULT_ALERTS = [
    Alert("oversold", lambda b: (b["price"]["rsi14"] or 50) < 30, "RSI below 30 (oversold)"),
    Alert("overbought", lambda b: (b["price"]["rsi14"] or 50) > 70, "RSI above 70 (overbought)"),
    Alert(
        "volume_spike",
        lambda b: (b["price"]["vol_vs_30d_avg"] or 1) > 2,
        "Volume more than 2x the 30-day average",
    ),
    Alert(
        "lost_200sma",
        lambda b: b["price"]["sma200"] is not None and b["price"]["last"] < b["price"]["sma200"],
        "Price below the 200-day SMA",
    ),
]


def check(tickers: list[str], alerts: list[Alert] | None = None) -> None:
    for ticker in tickers:
        try:
            data = build(ticker, refresh=True)
        except Exception as exc:
            console.print(f"[red]{ticker}: refresh failed — {exc}[/]")
            continue

        console.print(
            f"[dim]{data['as_of']}[/] [bold]{ticker}[/] "
            f"{data['price']['last']} ({data['price']['change_pct']}%)"
        )
        for alert in alerts or DEFAULT_ALERTS:
            try:
                if alert.predicate(data):
                    console.print(f"  [yellow]⚠ {alert.name}[/]: {alert.message}")
            except (KeyError, TypeError):
                continue


def run(tickers: list[str], minutes: int = 15) -> None:
    """Refresh and evaluate alerts on an interval until interrupted."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(check, "interval", minutes=minutes, args=[tickers], id="refresh")
    scheduler.start()

    console.print(
        f"[cyan]Watching {', '.join(t.upper() for t in tickers)} every {minutes} min. "
        "Ctrl-C to stop.[/]"
    )
    check(tickers)

    stop = signal.sigwait([signal.SIGINT, signal.SIGTERM])
    scheduler.shutdown()
    console.print(f"\n[dim]stopped (signal {stop})[/]")
