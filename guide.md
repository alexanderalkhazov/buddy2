# Setup & Verification Guide

Everything below assumes you're in `/Users/alexanderalhazov/Repositories/buddy2`.
The venv already exists with all dependencies installed.

---

## 1. What each key unlocks

Only one key is actually required, and it's free. The app degrades gracefully
without the others.

| Key | Required? | Without it |
|---|---|---|
| `GROQ_API_KEY` | **Yes, for `chat`** | Everything except `chat` works. `chat` exits with a clear error. |
| `NEWSAPI_KEY` | No | News falls back to yfinance headlines (works fine, fewer articles) |
| Other providers | No | Only if you switch `LLM_PROVIDER` |

Price data, Polymarket odds, fundamentals, and SEC filings need **no keys at all**.

---

## 2. Get a Groq key (free, ~1 minute)

> **Groq is not Grok.** Groq (groq.com) is a chip company running open models —
> free API tier, no card. Grok (x.ai) is Elon Musk's model — paid API only. The
> names are unhelpfully similar. This app now defaults to **Groq**.

1. Go to <https://console.groq.com/keys> and sign in (Google/GitHub works).
2. **Create API Key** → copy it. It starts with `gsk_`.
3. No credit card, no billing setup. There are rate limits (requests per minute
   and per day), which are generous for this app's usage.

**Check the model list** — Groq rotates models periodically. The default is
`llama-3.3-70b-versatile`:

```bash
.venv/bin/python -c "
import config
from openai import OpenAI
c = config.provider_config()
client = OpenAI(api_key=c['key'], base_url=c['base_url'])
ids = sorted(m.id for m in client.models.list().data)
print(*ids, sep='\n')
print()
print('configured:', c['model'], '->', 'VALID' if c['model'] in ids else 'NOT IN LIST')
"
```

If it says `NOT IN LIST`, pick any model from the output that supports tool calling
(the Llama 3.3 / 4 and Qwen instruct models do) and set `GROQ_MODEL` in `.env`.

### Switching providers later

The reasoning layer is provider-agnostic. Change one line in `.env`:

```bash
LLM_PROVIDER=groq         # free      — GROQ_API_KEY
LLM_PROVIDER=gemini       # free tier — GEMINI_API_KEY  (aistudio.google.com/apikey)
LLM_PROVIDER=openrouter   # free tier — OPENROUTER_API_KEY
LLM_PROVIDER=ollama       # local     — no key, needs Ollama running
LLM_PROVIDER=grok         # PAID      — XAI_API_KEY, needs credits
LLM_PROVIDER=anthropic    # paid      — ANTHROPIC_API_KEY
```

Tools, system prompt, memory, and the strategy output contract are identical across
all of them.

---

## 3. Get a NewsAPI key (optional)

1. <https://newsapi.org/register> — free tier, instant key.
2. The free tier is developer-only (localhost) and rate-limited to 100 requests/day.
   That's plenty here: news is cached for 6 hours per ticker.

---

## 4. Fill in `.env`

Open `.env` in the project root and set the values. Minimum viable config:

```bash
LLM_PROVIDER=grok
XAI_API_KEY=xai-your-actual-key-here
GROK_MODEL=grok-4-latest          # ← replace if step 2 showed a different ID
```

Optional additions:

```bash
NEWSAPI_KEY=your-newsapi-key
DEFAULT_TICKER=NVDA               # used when you omit the ticker argument
```

To use Claude instead of Grok, change two lines:

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

`.env` is gitignored. Never commit it.

---

## 5. Verify, in order

Run these top to bottom. Each step isolates one layer, so a failure tells you exactly
what broke.

### 5.1 Config loads and keys are visible

```bash
.venv/bin/python -c "
import config
print('provider :', config.LLM_PROVIDER)
print('model    :', config.GROK_MODEL)
print('xai key  :', 'set' if config.XAI_API_KEY else 'MISSING')
print('news key :', 'set' if config.NEWSAPI_KEY else 'not set (yfinance fallback)')
"
```

Expect `xai key : set`. If it says MISSING, `.env` isn't being read — confirm the file
is at the repo root and the line has no quotes or trailing spaces.

### 5.2 Data + indicators (no keys needed)

```bash
.venv/bin/python main.py quote AAPL
```

Expect a table with Last, RSI(14), MACD, SMAs, ATR, and a trend label. If this fails,
it's a network or yfinance issue, not a key issue.

### 5.3 The context bundle — what the model actually sees

```bash
.venv/bin/python main.py bundle AAPL --filings
```

Expect JSON with `price`, `news_summary`, `sentiment`, `polymarket_related`,
`fundamentals`, and `recent_filings`. If a source fails it appears as an `*_error`
key rather than crashing — that's by design.

**With a working LLM key, the summaries change shape**: instead of raw headlines
you'll see one-clause summaries. That's the confirmation your key works for the cheap
summarizer path.

### 5.4 Caching

Run `bundle` twice and time it:

```bash
time .venv/bin/python main.py bundle AAPL > /dev/null
time .venv/bin/python main.py bundle AAPL > /dev/null
```

The second run should be noticeably faster (DuckDB cache hit). Force a refresh with
`--refresh`.

### 5.5 Backtesting

```bash
.venv/bin/python main.py backtest AAPL strategies_example.json
```

Expect Sharpe ratio, max drawdown, win rate, trade count, and a buy-and-hold
comparison over ~5 years.

Try your own spec — save as `my_strategy.json`:

```json
{
  "name": "RSI dip above trend",
  "entry_rules": [
    {"left": "rsi14", "op": "<", "right": 35},
    {"left": "close", "op": ">", "right": "sma200"}
  ],
  "exit_rules": [{"left": "rsi14", "op": ">", "right": 65}],
  "entry_logic": "all",
  "exit_logic": "any"
}
```

Fields allowed: `open, high, low, close, volume, rsi14, macd, signal, hist, sma20,
sma50, sma200, bb_upper, bb_lower, bb_mid, atr14, vol_avg30`.
Operators: `<  <=  >  >=  crosses_above  crosses_below`.

### 5.6 The chat assistant — the real smoke test

```bash
.venv/bin/python main.py chat
```

Then paste this exact prompt:

```
Give me a strategy for NVDA.
```

**What a healthy run looks like:**

1. Dimmed tool-call lines appear as the model fetches data:
   ```
     → get_context_bundle(ticker='NVDA')
     → run_backtest(ticker='NVDA')
   ```
2. The answer comes back with all seven required sections: Thesis, Supporting signals,
   Contrary signals/risks, Suggested action, Position sizing, Invalidation, Backtest.
3. Each supporting signal names its source field (e.g. "RSI 63.0", "Polymarket: US
   recession by end of 2026 at 7.5%").

**If you get an answer with no tool-call lines**, the model answered from memory —
that's a failure, not a success. Re-run; if it persists, the model ID may not support
tool calling.

Other things to try in chat:

```
/note risk moderate, max 2% per position
/note portfolio long AAPL, MSFT
/notes
What changed for AAPL in the last 90 days?
Compare AAPL and MSFT on valuation.
```

Notes persist across sessions in DuckDB and are injected into the system prompt.

**Conversation history also persists across sessions** — quitting and relaunching
`chat` continues the same thread, not a fresh one. That's usually what you want (the
model remembers context), but it means a later answer can anchor on an earlier one
instead of re-deriving it from fresh data. Run `/reset` to clear conversation history
and start clean; it leaves notes untouched. Do this before re-testing a prompt change,
or a stale answer from before the change can resurface.

`/quit` to exit.

### 5.7 Dashboard

```bash
.venv/bin/python main.py dash AAPL MSFT NVDA
```

Arrow keys move between tickers, `r` refreshes, `q` quits. The detail pane on the
right mirrors the bundle for the highlighted row.

### 5.8 Background alerts

```bash
.venv/bin/python main.py watch AAPL MSFT --minutes 15
```

Prints a line per ticker plus any triggered alerts (RSI < 30, RSI > 70, volume > 2x
30-day average, price below 200-day SMA). Ctrl-C to stop.

> Note: this is the one command I never ran through a full interval. The first check
> fires immediately, so you'll know within seconds whether it works; if the scheduled
> repeat misbehaves, that's the first place to look.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `XAI_API_KEY is not set` | `.env` missing or key line blank | Check `.env` at repo root, no quotes around the value |
| `404` / `model not found` on chat | `GROK_MODEL` doesn't exist | Run the `curl` in step 2, set the exact ID |
| `401 Unauthorized` | Bad or revoked key | Regenerate at console.x.ai |
| `402` / insufficient credits | No xAI billing set up | Add credits to the xAI account |
| Chat answers without calling tools | Model ignored the tool contract | Check the model supports function calling; try a different `GROK_MODEL` |
| News summaries are raw headlines | LLM summarizer unavailable | Expected without a key — lexicon fallback is working as designed |
| `no cached price data` | First run for that ticker failed | Re-run with `--refresh`; check network |
| Stale prices | Cache TTL is 12h | Add `--refresh` to any command |

**Reset the cache** (safe — it re-downloads, but wipes conversation history and notes):

```bash
rm storage/buddy.duckdb
```

---

## 7. A note on what this is

The backtests are a sanity check on one historical window, not a forecast — they are
easy to overfit. No model reliably predicts short-term price direction. The value here
is fast synthesis and forced-explicit logic you can argue with. Every number the
assistant cites should be traceable to a tool call you can re-run yourself.
