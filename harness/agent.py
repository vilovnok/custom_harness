"""Harness — сборка всех частей и цикл диалога ведущего агента."""

from __future__ import annotations

from typing import Any

from harness.config import Config
from harness.loop import OnText, OnTool, run_loop
from harness.mcp_hub import MCPHub
from harness.providers import build_provider
from harness.skills import LoadSkillTool, Skill, load_skills, skills_catalog
from harness.subagents import TaskTool
from harness.tools import ToolRegistry, builtin_tools

_SYSTEM_TEMPLATE = """{base}

## Инструменты
{tools}

## Скилы (подгружай через load_skill по мере необходимости)
{skills}
"""


class Harness:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.provider = build_provider(config)
        self._mcp = MCPHub(config.mcp_servers)
        self.registry = ToolRegistry()
        self.skills: dict[str, Skill] = {}
        self.history: list[Any] = []  # диалог в формате провайдера
        self._system = ""

    async def __aenter__(self) -> "Harness":
        cfg = self.config

        self.registry.register_all(builtin_tools(cfg.workspace))

        await self._mcp.__aenter__()
        self.registry.register_all(self._mcp.tools)

        self.skills = load_skills(cfg.skills_dir)
        self.registry.register(LoadSkillTool(self.skills))

        if cfg.subagents:
            self.registry.register(
                TaskTool(cfg.subagents, self.provider, self.registry)
            )

        self._system = _SYSTEM_TEMPLATE.format(
            base=cfg.system_prompt,
            tools="\n".join(f"- {n}" for n in self.registry.names()),
            skills=skills_catalog(self.skills),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._mcp.__aexit__(*exc)

    # -- диалог -----------------------------------------------------------------

    async def chat(
        self,
        user_input: str,
        *,
        on_text: OnText | None = None,
        on_tool: OnTool | None = None,
    ) -> str:
        self.provider.add_user_message(self.history, user_input)
        return await run_loop(
            self.provider,
            system=self._system,
            history=self.history,
            registry=self.registry,
            on_text=on_text,
            on_tool=on_tool,
        )

    def inject_skill(self, name: str) -> str:
        """Ручная загрузка скила (CLI-команда /skill) — кладём тело как user-сообщение."""
        skill = self.skills.get(name)
        if skill is None:
            return f"Скила '{name}' нет."
        self.provider.add_user_message(
            self.history, f"[Загружен скил {name}]\n\n{skill.body}"
        )
        return f"Скил '{name}' добавлен в контекст."

    def reset(self) -> None:
        self.history.clear()
