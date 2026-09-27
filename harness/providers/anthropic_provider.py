"""Провайдер для Anthropic Messages API (Claude).

История — родной формат Anthropic: список ``{"role", "content"}``, где content
ассистента — список блоков (text / tool_use), а результаты инструментов — блоки
``tool_result`` внутри user-сообщения.
"""

from __future__ import annotations

from typing import Any

from harness.providers.base import Provider, ToolCall, ToolOutcome, ToolSpec, Turn


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        max_tokens: int = 8000,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from anthropic import AsyncAnthropic  # ленивый импорт

        self.model = model
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url
        # api_key=None -> SDK возьмёт ANTHROPIC_API_KEY / профиль `ant auth login`.
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    def clone(self, model: str | None) -> "AnthropicProvider":
        return AnthropicProvider(
            model or self.model,
            self._max_tokens,
            api_key=self._api_key,
            base_url=self._base_url,
        )

    async def step(
        self, *, system: str, history: list[Any], tool_specs: list[ToolSpec]
    ) -> Turn:
        tools = [
            {
                "name": s["name"],
                "description": s["description"],
                "input_schema": s["input_schema"],
            }
            for s in tool_specs
        ]
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            messages=history,
            tools=tools or [],
        )
        history.append({"role": "assistant", "content": resp.content})

        text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        return Turn(text=text, tool_calls=calls)

    def add_user_message(self, history: list[Any], text: str) -> None:
        history.append({"role": "user", "content": text})

    def add_tool_outcomes(self, history: list[Any], outcomes: list[ToolOutcome]) -> None:
        history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": o.call_id,
                        "content": o.content,
                        "is_error": o.is_error,
                    }
                    for o in outcomes
                ],
            }
        )
