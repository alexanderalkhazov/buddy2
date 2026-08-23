"""Rich-based chat REPL over the agent."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

import config
from llm.agent import Agent
from processing import portfolio
from storage import watchlist

console = Console()

BANNER = """\
[bold cyan]AI Trading Research Assistant[/]
Ask about a ticker, request a strategy, or type a command.

  [dim]/note <key> <value>[/]         store a standing note (e.g. /note risk moderate)
  [dim]/notes[/]                      show stored notes
  [dim]/portfolio[/]                  show positions, live P&L, sector concentration
  [dim]/portfolio add T S C [note][/] log a position: ticker, shares, cost/share
  [dim]/portfolio remove <id>[/]      remove a position by id
  [dim]/watchlist[/]                  show the watchlist
  [dim]/watchlist add|remove <T>[/]   manage the watchlist (drives dash/watch defaults)
  [dim]/reset[/]                      clear conversation history (notes/positions kept)
  [dim]/quit[/]                       exit
"""


def _show_tool(name: str, tool_input: dict) -> None:
    tool_input = tool_input or {}
    if name == "__provider_fallback__":
        console.print(
            f"[yellow]  ⚠ {tool_input['from']} unavailable ({tool_input['reason']}) "
            f"— switching to {tool_input['to']}[/]"
        )
        return
    if name == "__recovered_text_tool_call__":
        console.print(
            f"[yellow]  ⚠ model wrote a tool call as plain text instead of calling "
            f"it — recovered and ran {tool_input['name']}({tool_input['arguments']})[/]"
        )
        return
    args = ", ".join(f"{k}={v!r}" for k, v in tool_input.items() if k != "strategy")
    console.print(f"[dim]  → {name}({args})[/]")


def _print_portfolio() -> None:
    summary = portfolio.portfolio_summary()
    if not summary["positions"]:
        console.print("[dim]no positions logged — /portfolio add TICKER SHARES COST[/]")
        return

    table = Table(title="Portfolio", title_style="bold cyan")
    for col in ("id", "ticker", "shares", "cost", "last", "value", "P&L", "P&L %", "weight %"):
        table.add_column(col, justify="right" if col != "ticker" else "left")

    for p in summary["positions"]:
        pl_style = "green" if (p["unrealized_pl"] or 0) >= 0 else "red"
        table.add_row(
            str(p["id"]),
            p["ticker"],
            f"{p['shares']:g}",
            f"{p['cost_basis']:,.2f}",
            f"{p['last_price']:,.2f}" if p["last_price"] is not None else "n/a",
            f"{p['market_value']:,.2f}" if p["market_value"] is not None else "n/a",
            f"[{pl_style}]{p['unrealized_pl']:,.2f}[/]" if p["unrealized_pl"] is not None else "n/a",
            f"[{pl_style}]{p['unrealized_pl_pct']:+.1f}%[/]"
            if p["unrealized_pl_pct"] is not None
            else "n/a",
            f"{p['portfolio_weight_pct']:.1f}%" if p["portfolio_weight_pct"] is not None else "n/a",
        )
    console.print(table)

    total_style = "green" if (summary["total_pl"] or 0) >= 0 else "red"
    console.print(
        f"Total: [{total_style}]{summary['total_pl']:+,.2f} "
        f"({summary['total_pl_pct']:+.1f}%)[/] on {summary['total_value']:,.2f} value"
    )
    if summary["sector_concentration_pct"]:
        conc = ", ".join(f"{k} {v:.0f}%" for k, v in summary["sector_concentration_pct"].items())
        console.print(f"[dim]Sector concentration: {conc}[/]")


def run() -> None:
    console.print(Panel(BANNER, border_style="cyan"))

    try:
        with console.status("[dim]starting…[/]", spinner="dots"):
            agent = Agent()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        return

    console.print(f"[dim]{getattr(agent, 'label', config.LLM_PROVIDER)} · {agent.model}[/]")

    while True:
        try:
            question = console.input("\n[bold green]you[/] › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            return

        if not question:
            continue
        if question in ("/quit", "/exit"):
            return
        if question == "/notes":
            notes = agent.notes()
            console.print(notes or "[dim]no notes stored[/]")
            continue
        if question == "/reset":
            agent.reset()
            console.print("[dim]conversation cleared — starting fresh[/]")
            continue
        if question.startswith("/note "):
            parts = question[len("/note ") :].split(maxsplit=1)
            if len(parts) == 2:
                agent.remember(*parts)
                console.print(f"[dim]noted: {parts[0]}[/]")
            else:
                console.print("[yellow]usage: /note <key> <value>[/]")
            continue

        if question == "/portfolio":
            _print_portfolio()
            continue
        if question.startswith("/portfolio add "):
            parts = question[len("/portfolio add ") :].split(maxsplit=3)
            if len(parts) < 3:
                console.print("[yellow]usage: /portfolio add TICKER SHARES COST [note][/]")
                continue
            ticker, shares, cost, *note = parts
            try:
                pid = portfolio.add_position(
                    ticker, float(shares), float(cost), note[0] if note else None
                )
                console.print(f"[dim]logged position #{pid}: {ticker.upper()} {shares}sh @ {cost}[/]")
            except ValueError:
                console.print("[yellow]shares and cost must be numbers[/]")
            continue
        if question.startswith("/portfolio remove "):
            arg = question[len("/portfolio remove ") :].strip()
            try:
                removed = portfolio.remove_position(int(arg))
                console.print("[dim]removed[/]" if removed else "[yellow]no position with that id[/]")
            except ValueError:
                console.print("[yellow]usage: /portfolio remove <id>[/]")
            continue

        if question == "/watchlist":
            tickers = watchlist.list_all()
            console.print(", ".join(tickers) if tickers else "[dim]watchlist is empty[/]")
            continue
        if question.startswith("/watchlist add "):
            ticker = question[len("/watchlist add ") :].strip()
            watchlist.add(ticker)
            console.print(f"[dim]added {ticker.upper()} to watchlist[/]")
            continue
        if question.startswith("/watchlist remove "):
            ticker = question[len("/watchlist remove ") :].strip()
            removed = watchlist.remove(ticker)
            console.print("[dim]removed[/]" if removed else "[yellow]not on the watchlist[/]")
            continue

        try:
            # No spinner here — tool-call lines stream out during the turn.
            answer = agent.ask(question, on_tool=_show_tool)
        except Exception as exc:
            console.print(f"[red]{type(exc).__name__}: {exc}[/]")
            continue

        console.print()
        console.print(Markdown(answer))
