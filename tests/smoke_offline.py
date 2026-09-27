"""Оффлайн-проверка харнеса без сети и без SDK провайдеров.

Подменяем провайдер скриптованными `Turn` и убеждаемся, что:
  * loop выполняет tool-call и продолжает диалог;
  * субагент отрабатывает в изолированном контексте через инструмент task;
  * реестр инструментов фильтруется под спеку субагента.

Запуск:  python3 tests/smoke_offline.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.config import SubagentSpec
from harness.loop import run_loop
from harness.providers.base import Provider, ToolCall, ToolOutcome, Turn
from harness.subagents import TaskTool
from harness.tools import ToolRegistry, builtin_tools


class ScriptedProvider(Provider):
    """Отдаёт заранее заданные Turn по очереди. История — плоский список для наглядности."""

    name = "scripted"

    def __init__(self, script: list[Turn]) -> None:
        self.model = "scripted"
        self._script = list(script)
        self.seen_tool_specs: list[str] = []

    async def step(self, *, system, history, tool_specs):
        self.seen_tool_specs = [s["name"] for s in tool_specs]
        turn = self._script.pop(0)
        history.append({"role": "assistant", "turn": turn})
        return turn

    def add_user_message(self, history: list[Any], text: str) -> None:
        history.append({"role": "user", "content": text})

    def add_tool_outcomes(self, history: list[Any], outcomes: list[ToolOutcome]) -> None:
        for o in outcomes:
            history.append({"role": "tool", "name": o.name, "content": o.content})

    def clone(self, model: str | None) -> "ScriptedProvider":
        return self


async def test_loop_runs_tool() -> None:
    workspace = ROOT / "workspace"
    registry = ToolRegistry()
    registry.register_all(builtin_tools(workspace))

    provider = ScriptedProvider(
        [
            Turn(tool_calls=[ToolCall("t1", "write_file", {"path": "note.txt", "content": "hi"})]),
            Turn(text="Готово, файл записан."),
        ]
    )
    history: list[Any] = []
    provider.add_user_message(history, "создай note.txt со словом hi")
    out = await run_loop(provider, system="s", history=history, registry=registry)

    assert out == "Готово, файл записан.", out
    assert (workspace / "note.txt").read_text() == "hi"
    # user, assistant(turn), tool, assistant(turn)
    assert len(history) == 4, history
    print("ok  test_loop_runs_tool")


async def test_subagent_delegation() -> None:
    workspace = ROOT / "workspace"
    parent = ToolRegistry()
    parent.register_all(builtin_tools(workspace))

    spec = SubagentSpec(
        name="reader",
        description="читает файлы",
        system_prompt="ты читатель",
        tools=["read_file"],
    )
    provider = ScriptedProvider(
        [
            Turn(tool_calls=[ToolCall("s1", "read_file", {"path": "note.txt"})]),
            Turn(text="В файле: hi"),
        ]
    )
    task_tool = TaskTool([spec], provider, parent)

    sub_registry = task_tool._subagents["reader"]._registry
    assert sub_registry.names() == ["read_file"], sub_registry.names()

    result = await task_tool.run({"subagent_type": "reader", "description": "что в note.txt?"})
    assert result == "В файле: hi", result
    print("ok  test_subagent_delegation")


async def main() -> None:
    await test_loop_runs_tool()
    await test_subagent_delegation()
    print("\nвсе оффлайн-проверки прошли")


if __name__ == "__main__":
    asyncio.run(main())
