"""AI trading research assistant — console entry point.

Subcommands:
    quote      price + indicators for one ticker
    bundle     the full context bundle the LLM sees (JSON)
    chat       conversational assistant with tool use
    dash       textual dashboard over the watchlist
    watch      background refresh + threshold alerts
    backtest   run a strategy spec from a JSON file
    watchlist  manage the persisted watchlist (add/remove/list)
    portfolio  show tracked positions, live P&L, and sector concentration
    report     zero-LLM data package for one ticker — bundle, scorecard, strategy
               scan, backtest, position sizing — printed for you to paste into any
               chatbot yourself. No API calls, no cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

import config
from processing.indicators import PriceSnapshot, snapshot
from storage import cache

console = Console()


def _fmt(value, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:,.2f}{suffix}"


def render(snap: PriceSnapshot) -> None:
    table = Table(title=f"{snap.ticker} — as of {snap.as_of}", title_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    change = snap.change_pct
    style = "green" if change and change > 0 else "red" if change else ""

    table.add_row("Last", _fmt(snap.last))
    table.add_row("Change", f"[{style}]{_fmt(change, '%')}[/]" if style else _fmt(change, "%"))
    table.add_row("RSI(14)", _fmt(snap.rsi14))
    table.add_row("MACD / signal", f"{_fmt(snap.macd)} / {_fmt(snap.macd_signal)}")
    table.add_row("SMA 20 / 50 / 200", f"{_fmt(snap.sma20)} / {_fmt(snap.sma50)} / {_fmt(snap.sma200)}")
    table.add_row("Bollinger (20,2)", f"{_fmt(snap.bb_lower)} – {_fmt(snap.bb_upper)}")
    table.add_row("ATR(14)", _fmt(snap.atr14))
    table.add_row("Volume vs 30d avg", _fmt(snap.vol_vs_30d_avg, "x"))
    table.add_row("Trend", snap.trend)

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI trading research assistant")
    sub = parser.add_subparsers(dest="command")

    p_quote = sub.add_parser("quote", help="price + indicators for one ticker")
    p_quote.add_argument("ticker", nargs="?", default=config.DEFAULT_TICKER)
    p_quote.add_argument("--refresh", action="store_true")

    p_bundle = sub.add_parser("bundle", help="the context bundle the LLM sees")
    p_bundle.add_argument("ticker", nargs="?", default=config.DEFAULT_TICKER)
    p_bundle.add_argument("--refresh", action="store_true")
    p_bundle.add_argument("--filings", action="store_true")

    sub.add_parser("chat", help="conversational assistant")

    p_dash = sub.add_parser("dash", help="textual dashboard (defaults to the watchlist)")
    p_dash.add_argument("tickers", nargs="*")

    p_watch = sub.add_parser("watch", help="background refresh + alerts (defaults to the watchlist)")
    p_watch.add_argument("tickers", nargs="*")
    p_watch.add_argument("--minutes", type=int, default=15)

    p_bt = sub.add_parser("backtest", help="run a strategy spec")
    p_bt.add_argument("ticker")
    p_bt.add_argument("spec", type=Path, help="path to a JSON strategy spec")

    p_wl = sub.add_parser("watchlist", help="manage the persisted watchlist")
    wl_sub = p_wl.add_subparsers(dest="wl_command")
    wl_sub.add_parser("list", help="show the watchlist (default)")
    p_wl_add = wl_sub.add_parser("add", help="add a ticker")
    p_wl_add.add_argument("ticker")
    p_wl_remove = wl_sub.add_parser("remove", help="remove a ticker")
    p_wl_remove.add_argument("ticker")

    p_pf = sub.add_parser("portfolio", help="show tracked positions, P&L, and concentration")

    p_rp = sub.add_parser("report", help="zero-LLM data package for one ticker")
    p_rp.add_argument("ticker")
    p_rp.add_argument("--account-size", type=float, default=None, help="e.g. 100000")
    p_rp.add_argument("--risk-pct", type=float, default=1.0, help="percent of account to risk, default 1.0")
    p_rp.add_argument("--peers", nargs="*", default=None, help="tickers to compare against")
    p_rp.add_argument("--refresh", action="store_true")
    p_rp.add_argument("--out", type=Path, default=None, help="also write to this file")

    p_ds = sub.add_parser("datasheet", help="strict neutral data sheet for one ticker (zero interpretation)")
    p_ds.add_argument("ticker")
    p_ds.add_argument("--refresh", action="store_true")
    p_ds.add_argument("--out", type=Path, default=None, help="also write to this file")

    p_sc = sub.add_parser("screen", help="rank multiple tickers by the deterministic scorecard")
    p_sc.add_argument("tickers", nargs="+")
    p_sc.add_argument("--news", action="store_true", help="include news/sentiment (slower)")
    p_sc.add_argument("--refresh", action="store_true")

    p_sec = sub.add_parser("sector", help="peer-group average fundamentals across multiple tickers")
    p_sec.add_argument("tickers", nargs="+")
    p_sec.add_argument("--refresh", action="store_true")

    args = parser.parse_args()
    command = args.command or "quote"

    # Caret-prefixed symbols (^GSPC etc.) are painful to type past shell history
    # expansion (zsh/bash both treat a leading ^ specially) — resolve common
    # names to their real Yahoo symbol here so nobody has to fight their shell
    # to type an index ticker.
    if hasattr(args, "ticker") and args.ticker:
        args.ticker = _TICKER_HINTS.get(args.ticker.upper().replace(" ", ""), args.ticker)
    if hasattr(args, "tickers") and args.tickers:
        args.tickers = [_TICKER_HINTS.get(t.upper().replace(" ", ""), t) for t in args.tickers]

    try:
        _dispatch(args, command)
    except ValueError as exc:
        bad_ticker = getattr(args, "ticker", None) or (getattr(args, "tickers", None) or [None])[0]
        hint = _TICKER_HINTS.get((bad_ticker or "").upper().replace(" ", ""))
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


def _dispatch(args, command: str) -> None:
    if command == "quote":
        ticker = getattr(args, "ticker", config.DEFAULT_TICKER)
        with console.status(f"Fetching {ticker.upper()}…"):
            df = cache.get_ohlcv(ticker, force=getattr(args, "refresh", False))
        render(snapshot(ticker, df))

    elif command == "bundle":
        from processing.bundle import build

        with console.status(f"Building bundle for {args.ticker.upper()}…"):
            data = build(args.ticker, include_filings=args.filings, refresh=args.refresh)
        console.print_json(json.dumps(data, default=str))

    elif command == "chat":
        from ui.chat import run

        run()

    elif command in ("dash", "watch"):
        from storage import watchlist as wl

        tickers = args.tickers or wl.list_all() or [config.DEFAULT_TICKER]
        if command == "dash":
            from ui.app import run

            run(tickers)
        else:
            import scheduler

            scheduler.run(tickers, minutes=args.minutes)

    elif command == "backtest":
        from processing.backtest import run

        console.print_json(json.dumps(run(args.ticker, json.loads(args.spec.read_text()))))

    elif command == "watchlist":
        from storage import watchlist as wl

        sub_cmd = getattr(args, "wl_command", None) or "list"
        if sub_cmd == "add":
            wl.add(args.ticker)
            console.print(f"added {args.ticker.upper()}")
        elif sub_cmd == "remove":
            console.print("removed" if wl.remove(args.ticker) else "not on the watchlist")
        else:
            tickers = wl.list_all()
            console.print(", ".join(tickers) if tickers else "watchlist is empty")

    elif command == "portfolio":
        from processing.portfolio import portfolio_summary

        console.print_json(json.dumps(portfolio_summary(), default=str))

    elif command == "report":
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

    elif command == "datasheet":
        from ui.datasheet import generate as generate_datasheet

        with console.status(f"Building data sheet for {args.ticker.upper()}… (no API calls)"):
            text = generate_datasheet(args.ticker, refresh=args.refresh)
        print(text)
        if args.out:
            args.out.write_text(text)
            console.print(f"\n[dim]also written to {args.out}[/]")

    elif command == "screen":
        from processing.screener import screen

        with console.status(f"Screening {len(args.tickers)} tickers…"):
            rows = screen(args.tickers, use_news=args.news, refresh=args.refresh)

        table = Table(title="Screener — ranked by overall score", title_style="bold cyan")
        for col in ("Ticker", "Score", "Decision", "Fund", "Tech", "Val", "Sent", "Risk", "Fwd P/E", "Last"):
            table.add_column(col, justify="right" if col not in ("Ticker", "Decision", "Risk") else "left")
        for r in rows:
            if "error" in r:
                table.add_row(r["ticker"], "[red]error[/]", r["error"][:40], "", "", "", "", "", "", "")
                continue
            table.add_row(
                r["ticker"],
                _fmt(r["overall_score"]),
                r["decision"],
                _fmt(r["fundamental_score"]),
                _fmt(r["technical_score"]),
                _fmt(r["valuation_score"]),
                _fmt(r["sentiment_score"]),
                r["risk_adjusted_conviction"],
                _fmt(r["forward_pe"]),
                _fmt(r["last_price"]),
            )
        console.print(table)

    elif command == "sector":
        from processing.screener import sector_summary

        with console.status(f"Building peer-group summary for {len(args.tickers)} tickers…"):
            data = sector_summary(args.tickers, refresh=args.refresh)
        console.print_json(json.dumps(data, default=str))


if __name__ == "__main__":
    main()
