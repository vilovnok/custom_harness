"""Субагент-исполнитель и инструмент ``task`` для делегирования.

Субагент — это тот же ``run_loop``, но:
  * со своей, свежей историей сообщений (изолированный контекст);
  * с урезанным набором инструментов (по ``SubagentSpec.tools``);
  * без инструмента ``task`` (по умолчанию не плодит вложенные субагенты);
  * при желании — с другой моделью (``SubagentSpec.model`` -> provider.clone).
Родителю возвращается только финальный текст.
"""

from __future__ import annotations

from typing import Any

from harness.config import SubagentSpec
from harness.loop import run_loop
from harness.providers.base import Provider
from harness.tools import Tool, ToolRegistry


class Subagent:
    def __init__(self, spec: SubagentSpec, provider: Provider, parent_registry: ToolRegistry) -> None:
        self._spec = spec
        self._provider = provider.clone(spec.model) if spec.model else provider

        allowed = parent_registry.subset(spec.tools)
        self._registry = ToolRegistry()
        for name in allowed.names():
            tool = allowed.get(name)
            if name != "task" and tool is not None:
                self._registry.register(tool)

    async def run(self, task_description: str) -> str:
        history: list[Any] = []
        self._provider.add_user_message(history, task_description)
        return await run_loop(
            self._provider,
            system=self._spec.system_prompt,
            history=history,
            registry=self._registry,
            max_iters=30,
        )


class TaskTool(Tool):
    name = "task"

    def __init__(
        self, specs: list[SubagentSpec], provider: Provider, parent_registry: ToolRegistry
    ) -> None:
        self._subagents = {
            s.name: Subagent(s, provider, parent_registry) for s in specs
        }
        catalog = "\n".join(f"- {s.name}: {s.description}" for s in specs) or "(нет)"
        self.description = (
            "Делегировать подзадачу субагенту. Он работает в отдельном контексте и "
            "возвращает только итог. Субагенты:\n" + catalog
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "subagent_type": {"type": "string"},
                "description": {
                    "type": "string",
                    "description": "Полная самодостаточная постановка задачи для субагента.",
                },
            },
            "required": ["subagent_type", "description"],
        }

    async def run(self, arguments: dict[str, Any]) -> str:
        sub = self._subagents.get(arguments["subagent_type"])
        if sub is None:
            return f"Нет субагента '{arguments['subagent_type']}'. Есть: {', '.join(self._subagents)}"
        return await sub.run(arguments["description"])
