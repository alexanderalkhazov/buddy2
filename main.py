"""AI trading research assistant — console entry point.

One command: `report`. Zero-LLM, zero API cost — fetches real market data,
computes deterministic/ML predictions, and prints the hottest US stocks to
trade (long and short) with a plain-English why/when/how for each one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI trading research assistant")
    sub = parser.add_subparsers(dest="command")

    p_rp = sub.add_parser("report", help="the hottest US stocks to trade right now, long and short")
    p_rp.add_argument("--refresh", action="store_true", help="force-refetch cached price/news data")
    p_rp.add_argument("--full", action="store_true", help="print the full report (model reliability, per-ticker why/when/how, macro/sector/regime detail) instead of just the trade-decisions table")
    p_rp.add_argument("--out", type=Path, default=None, help="also write the report to this file")

    args = parser.parse_args()
    if args.command != "report":
        parser.print_help()
        raise SystemExit(1)

    from ui.market_report import generate

    with console.status("Building report… (no LLM calls — real data fetches for prices/macro/news/13 sector indices)"):
        text = generate(refresh=args.refresh, full=args.full)
    print(text)
    if args.out:
        args.out.write_text(text)
        console.print(f"\n[dim]also written to {args.out}[/]")


if __name__ == "__main__":
    main()
