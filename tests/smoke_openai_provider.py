"""Оффлайн-проверка OpenAIProvider (путь DeepSeek) без сети.

Подменяем HTTP-слой SDK фейком и проверяем, что провайдер:
  * формирует system + messages + tools в формате OpenAI/DeepSeek;
  * разбирает tool_calls (JSON-строка аргументов -> dict);
  * кладёт ответ ассистента и результат инструмента в историю правильных ролей;
  * в loop доводит диалог до финального текста.

Запуск:  python3 tests/smoke_openai_provider.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.loop import run_loop
from harness.providers.openai_provider import OpenAIProvider
from harness.tools import FunctionTool, ToolRegistry


class _FakeMessage:
    def __init__(self, content: str | None, tool_calls: list | None) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = None

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


def _tc(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


class _FakeCompletions:
    def __init__(self, script: list[_FakeMessage]) -> None:
        self._script = script
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        msg = self._script.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


async def main() -> None:
    calls_seen: list[str] = []

    async def _echo(text: str) -> str:
        calls_seen.append(text)
        return f"эхо: {text}"

    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            "echo",
            "Вернуть строку с префиксом.",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            _echo,
        )
    )

    provider = OpenAIProvider("deepseek-v4-flash", api_key="sk-test", base_url="https://api.deepseek.com")
    fake = _FakeCompletions(
        [
            _FakeMessage(None, [_tc("c1", "echo", {"text": "привет"})]),
            _FakeMessage("Инструмент вернул: эхо: привет", None),
        ]
    )
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))

    history: list[Any] = []
    provider.add_user_message(history, "позови echo с 'привет'")
    out = await run_loop(provider, system="ты агент", history=history, registry=registry)

    assert calls_seen == ["привет"], calls_seen
    assert out == "Инструмент вернул: эхо: привет", out

    # первый запрос: system подставлен, tools в формате OpenAI
    first = fake.calls[0]
    assert first["messages"][0] == {"role": "system", "content": "ты агент"}, first["messages"][0]
    assert first["tools"][0]["type"] == "function"
    assert first["tools"][0]["function"]["name"] == "echo"

    # история: user, assistant(tool_calls), tool, assistant(text)
    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant", "tool", "assistant"], roles
    assert history[2]["tool_call_id"] == "c1", history[2]

    print("ok  smoke_openai_provider — путь DeepSeek работает")


if __name__ == "__main__":
    asyncio.run(main())
