"""Anthropic backend plus the provider factory.

`Agent()` returns whichever backend `LLM_PROVIDER` selects, so the UI and scheduler
never know which model is behind the assistant.
"""

from __future__ import annotations

import anthropic

import config
from llm import tools
from llm.base import MAX_TURNS, BaseAgent


class AnthropicAgent(BaseAgent):
    def __init__(self, model: str | None = None, history_turns: int = 20):
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set — add it to .env")
        super().__init__(history_turns)
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = model or config.ANTHROPIC_MODEL

    def _system(self) -> list[dict]:
        return [
            {
                "type": "text",
                "text": self.system_prompt(),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _run_turn(self, question: str, on_tool) -> str:
        self.messages.append({"role": "user", "content": question})

        for _ in range(MAX_TURNS):
            with self.client.messages.stream(
                model=self.model,
                max_tokens=16000,
                system=self._system(),
                tools=tools.TOOLS,
                thinking={"type": "adaptive"},
                output_config={"effort": config.ANTHROPIC_EFFORT},
                messages=self.messages,
            ) as stream:
                response = stream.get_final_message()

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                return "[The model declined to answer this request.]"

            if response.stop_reason != "tool_use":
                return "\n".join(b.text for b in response.content if b.type == "text")

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if on_tool:
                    on_tool(block.name, block.input)
                payload, is_error = tools.execute(block.name, block.input)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": payload,
                        "is_error": is_error,
                    }
                )
            self.messages.append({"role": "user", "content": results})

        return "[Stopped: hit the tool-call limit for one turn.]"


def Agent(provider: str | None = None, **kwargs) -> BaseAgent:
    """Build the agent backend named by config.LLM_PROVIDER."""
    cfg = config.provider_config(provider)

    if cfg["name"] == "anthropic":
        return AnthropicAgent(**kwargs)

    # Ollama runs locally with no key; everything else needs one.
    if not cfg["key"] and cfg["name"] != "ollama":
        raise RuntimeError(
            f"{cfg['key_env']} is not set — add it to .env. Get one at {cfg['signup']}"
        )

    from llm.openai_compat import OpenAICompatAgent

    return OpenAICompatAgent(
        api_key=cfg["key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        label=cfg["name"],
        **kwargs,
    )
