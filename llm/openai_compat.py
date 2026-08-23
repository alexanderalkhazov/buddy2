"""Backend for any OpenAI-compatible chat-completions API.

Groq, Gemini, OpenRouter, xAI, and Ollama all expose the same wire format, so they
differ only by base URL, key, and model name — see config.PROVIDERS.
"""

from __future__ import annotations

import json

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

# Errors that mean "this provider can't serve the request right now" — worth trying
# the next configured provider rather than failing the whole turn. The SDK already
# retries 5xx/429/connection errors internally (default 2 attempts with backoff)
# before raising, so anything that reaches here has already outlasted that — a
# sustained outage or overload, not a single blip. Excludes BadRequestError, which is
# a request-shape problem that will be identical on every provider.
_UNAVAILABLE = (
    RateLimitError,
    AuthenticationError,
    PermissionDeniedError,
    InternalServerError,
    APIConnectionError,
)


def _extract_fake_tool_call(text: str) -> dict | None:
    """Detect a tool call the model wrote as plain-text JSON instead of using the
    real tool-calling mechanism — a known failure mode on smaller/local models
    (e.g. qwen3:4b) that never gets a proper `tool_calls` entry and would otherwise
    print raw '{"name": "...", "arguments": {...}}' straight into the chat."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    candidate = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                if (
                    isinstance(candidate, dict)
                    and candidate.get("name") in tools.HANDLERS
                    and isinstance(candidate.get("arguments"), dict)
                ):
                    return candidate
                return None
    return None

import config
from llm import tools
from llm.base import MAX_TURNS, BaseAgent


def _tool_use_error(exc: BadRequestError) -> str | None:
    """Return the provider's message if this was a tool-argument rejection."""
    body = getattr(exc, "body", None) or {}
    error = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        return None
    if error.get("code") != "tool_use_failed":
        return None
    return str(error.get("message", exc))[:400]


def _openai_tools() -> list[dict]:
    """Translate the canonical tool schemas into OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools.TOOLS
    ]


class OpenAICompatAgent(BaseAgent):
    """Tool-use loop against an OpenAI-compatible endpoint.

    Falls back to the next configured provider on a rate limit or auth failure, so
    one free provider hitting its daily cap mid-conversation doesn't stop the app.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        label: str = "provider",
        history_turns: int = 20,
        allow_fallback: bool = True,
    ):
        super().__init__(history_turns)
        primary = {"name": label, "key": api_key, "base_url": base_url, "model": model}
        self._candidates = [primary] + (config.fallback_chain(label) if allow_fallback else [])
        self._active = 0
        self._switch_to(0)
        self.tools = _openai_tools()

    def _switch_to(self, index: int) -> None:
        cfg = self._candidates[index]
        self._active = index
        # Ollama runs without auth; the SDK still requires a non-empty string.
        self.client = OpenAI(api_key=cfg.get("key") or "not-needed", base_url=cfg["base_url"])
        self.model = cfg["model"]
        self.label = cfg["name"]

    def _next_fallback(self) -> bool:
        """Advance to the next candidate provider. Returns False if none remain."""
        if self._active + 1 >= len(self._candidates):
            return False
        self._switch_to(self._active + 1)
        return True

    def _fallback_or_raise(self, exc: Exception, on_tool) -> None:
        from_label = self.label
        if not self._next_fallback():
            raise RuntimeError(
                f"{from_label} is unavailable ({type(exc).__name__}) and no fallback "
                "provider has a key configured. Add another key to .env (see "
                "config.FALLBACK_ORDER) or try again later."
            ) from exc
        if on_tool:
            on_tool(
                "__provider_fallback__",
                {"from": from_label, "to": self.label, "reason": type(exc).__name__},
            )

    def _fresh_convo(self, question: str, tool_log: list[str]) -> list[dict]:
        """Rebuild the message list from scratch: system prompt, plain-text history,
        the question, and a flattened summary of any tool calls already made this
        turn. Used both at turn start and after every provider fallback — different
        providers serialize tool-call/function-call turns differently (e.g. Gemini
        requires a 'thought_signature' field Groq's format doesn't produce), so
        replaying a raw provider-native message list on a different provider can be
        rejected outright. Plain text is universally compatible; nothing is lost
        except the exact wire format, which no provider needs to see again."""
        convo = [{"role": "system", "content": self.system_prompt()}]
        convo += [m for m in self.messages if isinstance(m.get("content"), str)]
        convo.append({"role": "user", "content": question})
        if tool_log:
            convo.append(
                {
                    "role": "user",
                    "content": (
                        "[Tool results already gathered this turn, before a provider "
                        "switch — do not re-fetch these unless you need something "
                        "different:]\n" + "\n".join(tool_log)
                    ),
                }
            )
        return convo

    def _run_turn(self, question: str, on_tool) -> str:
        tool_log: list[str] = []
        convo = self._fresh_convo(question, tool_log)
        self.messages.append({"role": "user", "content": question})

        turns = 0
        while turns < MAX_TURNS:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=convo,
                    tools=self.tools,
                    tool_choice="auto",
                )
            except _UNAVAILABLE as exc:
                self._fallback_or_raise(exc, on_tool)
                convo = self._fresh_convo(question, tool_log)
                continue  # retry on the new provider with a portable message list
            except APIStatusError as exc:
                # 413 "request too large" has no dedicated exception class in the
                # SDK (unlike 429/401/403/5xx) — check status_code directly. A
                # bigger-context provider can usually serve the same request, so
                # this goes through the same fallback path rather than failing the
                # turn outright.
                if exc.status_code != 413:
                    raise
                self._fallback_or_raise(exc, on_tool)
                convo = self._fresh_convo(question, tool_log)
                continue
            except BadRequestError as exc:
                # Some providers validate tool arguments server-side and reject the
                # generation outright. Hand the model its own error so it can retry
                # rather than losing the turn.
                detail = _tool_use_error(exc)
                if detail is None:
                    raise
                convo.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last tool call was rejected: "
                            f"{detail}\nFix the argument types and call the tool again."
                        ),
                    }
                )
                turns += 1
                continue

            message = response.choices[0].message
            convo.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                text = (message.content or "").strip()

                fake_call = _extract_fake_tool_call(text)
                if fake_call is not None:
                    name, arguments = fake_call["name"], fake_call["arguments"]
                    if on_tool:
                        on_tool(
                            "__recovered_text_tool_call__",
                            {"name": name, "arguments": arguments},
                        )
                        on_tool(name, arguments)
                    payload, is_error = tools.execute(name, arguments)
                    tool_log.append(
                        f"{name}({arguments}) -> {'Error: ' + payload if is_error else payload}"
                    )
                    convo.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response wrote a tool call as plain "
                                f"text instead of using the tool-calling mechanism, so "
                                f"it was extracted and run for you. Result of "
                                f"{name}({arguments}): "
                                f"{'Error: ' + payload if is_error else payload}\n"
                                "Give your actual answer now using this data — do not "
                                "write JSON in your response."
                            ),
                        }
                    )
                    turns += 1
                    continue

                if not text:
                    # Small local models can choke on a large tool payload (e.g. a
                    # multi-strategy backtest scan) and return empty content with no
                    # error. Surface that plainly instead of silence.
                    text = (
                        f"[{self.label}/{self.model} returned an empty response — this "
                        "can happen with a large tool result on a small model. Try "
                        "asking a narrower question, or retry once a cloud provider "
                        "is available.]"
                    )
                self.messages.append({"role": "assistant", "content": text})
                return text

            for call in message.tool_calls:
                name = call.function.name
                try:
                    # Some models emit "null" as the arguments string for zero-arg
                    # tools instead of "{}" — treat any non-dict parse as no args.
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        arguments = {}
                except json.JSONDecodeError:
                    convo.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": "Error: arguments were not valid JSON.",
                        }
                    )
                    continue

                if on_tool:
                    on_tool(name, arguments)
                payload, is_error = tools.execute(name, arguments)
                tool_log.append(
                    f"{name}({arguments}) -> {'Error: ' + payload if is_error else payload}"
                )
                convo.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"Error: {payload}" if is_error else payload,
                    }
                )

            turns += 1

        return "[Stopped: hit the tool-call limit for one turn.]"
