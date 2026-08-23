# AI Trading Research Assistant

A local console app that synthesizes market data into explicit, checkable reasoning.
It is a research aid, not a forecaster — see *Expectations* at the bottom.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # add GROQ_API_KEY — free, no card: console.groq.com/keys
```

Price data (yfinance), Polymarket, and EDGAR need no keys. Without `NEWSAPI_KEY` the
news layer falls back to yfinance headlines; without an LLM key everything except
`chat` still works (news sentiment falls back to a lexicon).

### LLM provider

**Groq is the default** — free, no card required. The reasoning layer is
provider-agnostic — `llm/base.py` owns the system prompt, memory, and tool dispatch;
`llm/openai_compat.py` is one backend shared by every OpenAI-compatible provider.
Switch with one line in `.env`:

```bash
LLM_PROVIDER=groq        # free      — GROQ_API_KEY (console.groq.com/keys)
LLM_PROVIDER=gemini      # free tier — GEMINI_API_KEY (aistudio.google.com/apikey)
LLM_PROVIDER=openrouter  # free tier — OPENROUTER_API_KEY (:free models)
LLM_PROVIDER=ollama      # local     — no key, needs Ollama running
LLM_PROVIDER=grok        # paid      — XAI_API_KEY, requires xAI credits
LLM_PROVIDER=anthropic   # paid      — ANTHROPIC_API_KEY, uses the native SDK
```

> **Groq ≠ Grok.** Groq (groq.com, this default) is a hardware company running open
> models for free. Grok (x.ai) is a separate, paid product. Easy to mix up.

Tool schemas live once in `llm/tools.py` and are translated per provider — never
duplicated. `config.provider_config()` resolves the active provider's key, base URL,
and model; `llm.agent.Agent()` picks the right backend from it. See `guide.md` for a
full walkthrough of getting each key and verifying it works.

**A note on smaller free models:** they're looser about tool-call argument types
(sending `"false"` instead of `false`) and can be inconsistent about following a long
system prompt to the letter. `llm/tools.py` tolerates the type looseness (schemas
accept both string and native types, then coerce); prompt adherence is a model
capability, not something the code can force. If output quality matters more than
cost, `anthropic` or `gemini` follow the structure more reliably.

## Commands

```bash
python main.py quote AAPL              # price + indicators
python main.py bundle AAPL --filings   # the exact JSON the LLM sees
python main.py chat                    # conversational assistant with tool use
python main.py dash AAPL MSFT NVDA     # textual dashboard
python main.py watch AAPL --minutes 15 # background refresh + threshold alerts
python main.py backtest AAPL strategies_example.json
```

## Architecture

```
ui/          chat REPL (rich) + dashboard (textual)
llm/         base.py (system prompt, memory, tool dispatch — shared by every
             backend), openai_compat.py (groq/gemini/openrouter/ollama/grok),
             agent.py (anthropic backend + provider factory), tools.py (schemas)
processing/  indicators, news summarization/sentiment, bundle, backtest
data/        prices, news, polymarket, fundamentals, edgar
storage/     DuckDB schema + cache (prices, news, conversation, notes)
```

`processing/backtest.py` imports `vectorbt`, which is slow to import (~2s cold). That
import is deferred to the first `run_backtest` tool call, so `chat` starts instantly
and only pays the cost if a backtest actually runs.

**The core invariant:** raw API data never reaches the model. Every source is reduced
in `processing/bundle.py` to a compact JSON bundle — scalars and one-line summaries —
and that bundle is the only thing tools return.

```json
{
  "ticker": "AAPL",
  "price": {"last": 305.93, "rsi14": 43.7, "trend": "above_200sma", "atr14": 7.86},
  "news_summary": ["Apple and Cisco downgraded (Investing.com, 1h ago)"],
  "sentiment": "positive",
  "polymarket_related": [{"market": "US recession by end of 2026?", "odds": 0.075}],
  "fundamentals": {"forward_pe": 28.4, "sector": "Technology"}
}
```

## Strategy output contract

The system prompt (`llm/base.py`) requires every strategy or investment response to
carry: Thesis, Supporting signals (tagged with source field), Contrary signals/risks,
Scenarios (bull/base/bear), Suggested action, Position sizing (a risk-per-trade figure
and a stop, not just "size by ATR"), Invalidation, and Backtest metrics. A bare
buy/sell with no reasoning trace is not a valid response.

Two rules the prompt enforces to keep reasoning from collapsing onto one weak signal:

- **"Is this a good investment?" and "should I buy today?" are treated as separate
  questions.** The first is about the business — what drives its earnings and whether
  that driver is strengthening. The second is about entry timing, where a single
  indicator threshold (RSI, an MA cross) is weak evidence on its own. A fundamentally
  bullish case with no technical entry should read as "attractive business, wait for a
  better entry," never a bare "no action" that silently drops the fundamental case.
- **Fundamentals and industry-cycle data outrank sentiment and technicals as
  evidence.** News sentiment and analyst targets are noted as color, not cited as
  primary support — analysts are frequently wrong and positive sentiment is often
  already priced in. Claims like "large market cap → slower growth" require a named
  mechanism or get dropped.

Backtest results are reported with their trade count and an explicit note on whether
that sample size is large enough to trust the Sharpe ratio and win rate, and whether
the drawdown shown is survivable at the proposed position size — a backtest is one
weak technical signal, never the sole basis for an investment view on an operating
business.

## Backtesting

The model never emits Python. It emits a declarative spec that compiles to vectorbt
signals, so proposed strategies are executable, inspectable, and safe to run:

```json
{
  "name": "Golden cross",
  "entry_rules": [{"left": "sma50", "op": "crosses_above", "right": "sma200"}],
  "exit_rules":  [{"left": "sma50", "op": "crosses_below", "right": "sma200"}]
}
```

Fields are restricted to an allowlist of indicator columns; operators are `<`, `<=`,
`>`, `>=`, `crosses_above`, `crosses_below`. Reports Sharpe, max drawdown, win rate,
trade count, and a buy-and-hold comparison.

## Caching and memory

DuckDB at `storage/buddy.duckdb`. Prices refresh after 12h, news after 6h; `--refresh`
forces a pull.

Conversation history and standing notes (`/note risk moderate` in chat) persist
**across process restarts**, not just within one `chat` session — quitting and
relaunching `chat` continues the same conversation. This is intentional (the model
remembers your risk tolerance and portfolio without being told again), but it means
the model can anchor on an earlier answer instead of re-deriving one from fresh data.
Use `/reset` in chat to clear conversation history and start a clean thread; `/notes`
and stored notes are untouched by it.

## Expectations

No LLM reliably predicts short-term price direction. The value here is fast synthesis,
surfaced signals, and forced explicit logic behind any idea. Treat backtests as a
sanity check on a single historical window, not a guarantee — they are easy to overfit.


.venv/bin/python main.py report <Ticker>