"""AI trading research assistant — console entry point.

Subcommands:
    report     zero-LLM data package for one ticker — bundle, scorecard, strategy
               scan, backtest, position sizing — printed for you to paste into any
               chatbot yourself. No API calls, no cost.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI trading research assistant")
    sub = parser.add_subparsers(dest="command")

    p_rp = sub.add_parser("report", help="zero-LLM data package for one ticker")
    p_rp.add_argument("ticker")
    p_rp.add_argument("--account-size", type=float, default=None, help="e.g. 100000")
    p_rp.add_argument("--risk-pct", type=float, default=1.0, help="percent of account to risk, default 1.0")
    p_rp.add_argument("--peers", nargs="*", default=None, help="tickers to compare against")
    p_rp.add_argument("--refresh", action="store_true")
    p_rp.add_argument("--out", type=Path, default=None, help="also write to this file")

    args = parser.parse_args()
    if args.command != "report":
        parser.print_help()
        raise SystemExit(1)

    # Caret-prefixed symbols (^GSPC etc.) are painful to type past shell history
    # expansion (zsh/bash both treat a leading ^ specially) — resolve common
    # names to their real Yahoo symbol here so nobody has to fight their shell
    # to type an index ticker.
    args.ticker = _TICKER_HINTS.get(args.ticker.upper().replace(" ", ""), args.ticker)

    try:
        _run_report(args)
    except ValueError as exc:
        hint = _TICKER_HINTS.get(args.ticker.upper().replace(" ", ""))
        console.print(f"[red]Error:[/] {exc}")
        if hint:
            console.print(f"[dim]Did you mean {hint!r}? Yahoo Finance uses that symbol instead.[/]")
        raise SystemExit(1)


# Common names people type that don't match Yahoo Finance's actual ticker symbols.
# Resolved automatically (see above) so the caret in e.g. ^GSPC never has to be
# typed on the command line.
_TICKER_HINTS = {
    "S&P500": "^GSPC",
    "SP500": "^GSPC",
    "SPX": "^GSPC",
    "GSPC": "^GSPC",
    "NASDAQ": "^IXIC",
    "IXIC": "^IXIC",
    "NASDAQ100": "^NDX",
    "NDX": "^NDX",
    "DOWJONES": "^DJI",
    "DOW": "^DJI",
    "DJI": "^DJI",
    "RUSSELL2000": "^RUT",
    "RUT": "^RUT",
    "VIX": "^VIX",
}


def _run_report(args) -> None:
    from ui.report import generate

    with console.status(f"Building report for {args.ticker.upper()}… (no API calls)"):
        text = generate(
            args.ticker,
            account_size=args.account_size,
            risk_pct=args.risk_pct,
            peers=args.peers,
            refresh=args.refresh,
        )
    print(text)
    if args.out:
        args.out.write_text(text)
        console.print(f"\n[dim]also written to {args.out}[/]")


if __name__ == "__main__":
    main()
