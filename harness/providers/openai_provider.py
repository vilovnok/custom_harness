"""Провайдер для любого OpenAI-совместимого API: OpenAI, DeepSeek, локальные шлюзы.

DeepSeek — это OpenAI-совместимый Chat Completions API:
  base_url = https://api.deepseek.com,  модели deepseek-v4-pro / deepseek-v4-flash,
  function calling в формате OpenAI, thinking-режим через extra_body.

История — родной формат OpenAI: плоский список сообщений с ролями
user / assistant / tool. Ответ ассистента с ``tool_calls`` кладётся как есть,
результат инструмента — отдельное сообщение ``{"role": "tool", "tool_call_id", ...}``.
"""

from __future__ import annotations

import json
from typing import Any

from harness.providers.base import Provider, ToolCall, ToolOutcome, ToolSpec, Turn


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        model: str,
        max_tokens: int = 8000,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        thinking: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI  # ленивый импорт

        self.model = model
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._thinking = thinking
        self._reasoning_effort = reasoning_effort
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def clone(self, model: str | None) -> "OpenAIProvider":
        return OpenAIProvider(
            model or self.model,
            self._max_tokens,
            api_key=self._api_key,
            base_url=self._base_url,
            thinking=self._thinking,
            reasoning_effort=self._reasoning_effort,
        )

    async def step(
        self, *, system: str, history: list[Any], tool_specs: list[ToolSpec]
    ) -> Turn:
        # system у OpenAI — обычное сообщение; храним его отдельно и подставляем
        # в начало на каждый запрос, а в history не кладём.
        messages = ([{"role": "system", "content": system}] if system else []) + history

        tools = [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["input_schema"],
                },
            }
            for s in tool_specs
        ]

        kwargs: dict[str, Any] = {}
        if self._thinking:  # DeepSeek thinking-режим
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            if self._reasoning_effort:
                kwargs["reasoning_effort"] = self._reasoning_effort

        resp = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self._max_tokens,
            messages=messages,
            tools=tools or None,
            **kwargs,
        )
        msg = resp.choices[0].message

        dumped = msg.model_dump(exclude_none=True)
        dumped.pop("reasoning_content", None)  # reasoning обратно в контекст не шлём
        history.append(dumped)

        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return Turn(text=(msg.content or "").strip(), tool_calls=calls)

    def add_user_message(self, history: list[Any], text: str) -> None:
        history.append({"role": "user", "content": text})

    def add_tool_outcomes(self, history: list[Any], outcomes: list[ToolOutcome]) -> None:
        # У OpenAI нет флага is_error — помечаем текстом.
        for o in outcomes:
            content = f"ERROR: {o.content}" if o.is_error else o.content
            history.append(
                {"role": "tool", "tool_call_id": o.call_id, "content": content}
            )
