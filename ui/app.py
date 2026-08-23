"""Textual dashboard: watchlist, live indicator panel, news, and alerts."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, Log, Static

import config
from processing.bundle import build

REFRESH_SECONDS = 300


class Dashboard(App):
    """Watchlist-driven view over the same context bundles the LLM sees."""

    CSS = """
    Screen { layout: vertical; }
    #top { height: 60%; }
    #watchlist { width: 60%; border: round $accent; }
    #detail { width: 40%; border: round $accent; padding: 0 1; }
    #news { height: 40%; border: round $accent; }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, tickers: list[str]):
        super().__init__()
        self.tickers = [t.upper() for t in tickers]
        self.bundles: dict[str, dict] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top"):
            yield DataTable(id="watchlist")
            yield Static("Select a ticker", id="detail")
        yield Log(id="news", highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#watchlist", DataTable)
        table.add_columns("Ticker", "Last", "Chg %", "RSI", "Trend", "Sentiment")
        table.cursor_type = "row"
        self.set_interval(REFRESH_SECONDS, self.action_refresh)
        self.action_refresh()

    def action_refresh(self) -> None:
        for ticker in self.tickers:
            self.refresh_ticker(ticker)

    @staticmethod
    def _fmt(value, suffix: str = "") -> str:
        return "n/a" if value is None else f"{value:,.2f}{suffix}"

    def refresh_ticker(self, ticker: str) -> None:
        log = self.query_one("#news", Log)
        try:
            data = build(ticker)
        except Exception as exc:
            log.write_line(f"[{ticker}] error: {exc}")
            return

        self.bundles[ticker] = data
        table = self.query_one("#watchlist", DataTable)
        price = data["price"]
        row = (
            ticker,
            self._fmt(price["last"]),
            self._fmt(price["change_pct"], "%"),
            self._fmt(price["rsi14"]),
            price["trend"],
            data.get("sentiment", "—"),
        )

        try:
            table.remove_row(ticker)
        except Exception:
            pass  # first render — no existing row
        table.add_row(*row, key=ticker)

        for line in data.get("news_summary", [])[:3]:
            log.write_line(f"[{ticker}] {line}")

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        ticker = str(event.row_key.value)
        data = self.bundles.get(ticker)
        if not data:
            return

        price, fund = data["price"], data.get("fundamentals", {})
        lines = [
            f"[b]{ticker}[/b]  as of {data['as_of']}",
            "",
            f"Last          {self._fmt(price['last'])}",
            f"Change        {self._fmt(price['change_pct'], '%')}",
            f"RSI(14)       {self._fmt(price['rsi14'])}",
            f"MACD hist     {self._fmt(price['macd_hist'])}",
            f"SMA 20/50/200 {self._fmt(price['sma20'])} / {self._fmt(price['sma50'])} / {self._fmt(price['sma200'])}",
            f"ATR(14)       {self._fmt(price['atr14'])}",
            f"Vol vs 30d    {self._fmt(price['vol_vs_30d_avg'], 'x')}",
            f"Trend         {price['trend']}",
            "",
            f"Sector        {fund.get('sector') or 'n/a'}",
            f"Forward P/E   {self._fmt(fund.get('forward_pe'))}",
            f"Sentiment     {data.get('sentiment', 'n/a')}",
        ]
        self.query_one("#detail", Static).update("\n".join(lines))


def run(tickers: list[str] | None = None) -> None:
    Dashboard(tickers or [config.DEFAULT_TICKER]).run()
