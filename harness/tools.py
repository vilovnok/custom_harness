"""Абстракция инструмента, реестр и набор встроенных инструментов.

Инструмент для харнеса — это объект с ``name`` / ``description`` / ``input_schema``
и корутиной ``run(arguments) -> str``. Откуда он взялся (встроенный, MCP, обёртка
над субагентом) — реестру всё равно.
"""

from __future__ import annotations

import abc
import asyncio
import fnmatch
from pathlib import Path
from typing import Any, Awaitable, Callable

MAX_OUTPUT = 8000  # символов на один результат инструмента — чтобы не раздувать контекст


class Tool(abc.ABC):
    name: str
    description: str
    input_schema: dict[str, Any]

    @abc.abstractmethod
    async def run(self, arguments: dict[str, Any]) -> str: ...

    def to_spec(self) -> dict[str, Any]:
        """Формат, который ждёт Messages API в параметре ``tools``."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class FunctionTool(Tool):
    """Быстро сделать инструмент из обычной корутины."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        func: Callable[..., Awaitable[str]],
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._func = func

    async def run(self, arguments: dict[str, Any]) -> str:
        return await self._func(**arguments)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        return [t.to_spec() for t in self._tools.values()]

    def subset(self, patterns: list[str]) -> "ToolRegistry":
        """Новый реестр только с инструментами, подходящими под шаблоны."""
        if not patterns:
            return self
        picked = ToolRegistry()
        for name, tool in self._tools.items():
            if any(fnmatch.fnmatch(name, p) for p in patterns):
                picked.register(tool)
        return picked

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Вызвать инструмент. Возвращает (текст, is_error)."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Неизвестный инструмент: {name}", True
        try:
            out = await tool.run(arguments)
        except Exception as exc:  # инструмент не должен ронять loop
            return f"Ошибка инструмента {name}: {exc!r}", True
        return out[:MAX_OUTPUT], False


# --------------------------------------------------------------------------- #
# Встроенные инструменты (песочница — рабочая директория)
# --------------------------------------------------------------------------- #


def _safe_path(workspace: Path, rel: str) -> Path:
    target = (workspace / rel).resolve()
    if workspace not in target.parents and target != workspace:
        raise ValueError(f"Путь вне рабочей директории: {rel}")
    return target


class BashTool(Tool):
    name = "bash"
    description = "Выполнить shell-команду в рабочей директории, вернуть stdout+stderr и код возврата."
    input_schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(self, workspace: Path, timeout: float = 60.0) -> None:
        self._workspace = workspace
        self._timeout = timeout

    async def run(self, arguments: dict[str, Any]) -> str:
        proc = await asyncio.create_subprocess_shell(
            arguments["command"],
            cwd=self._workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Таймаут {self._timeout}s"
        return f"(exit {proc.returncode})\n{out.decode(errors='replace')}"


class ReadFileTool(Tool):
    name = "read_file"
    description = "Прочитать текстовый файл из рабочей директории."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    async def run(self, arguments: dict[str, Any]) -> str:
        return _safe_path(self._workspace, arguments["path"]).read_text(encoding="utf-8")


class WriteFileTool(Tool):
    name = "write_file"
    description = "Записать (перезаписать) текстовый файл в рабочей директории."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    async def run(self, arguments: dict[str, Any]) -> str:
        target = _safe_path(self._workspace, arguments["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(arguments["content"], encoding="utf-8")
        return f"Записано {len(arguments['content'])} символов в {arguments['path']}"


class ListDirTool(Tool):
    name = "list_dir"
    description = "Показать содержимое директории внутри рабочей директории."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "default": "."}},
    }

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    async def run(self, arguments: dict[str, Any]) -> str:
        base = _safe_path(self._workspace, arguments.get("path", "."))
        entries = sorted(
            f"{p.name}/" if p.is_dir() else p.name for p in base.iterdir()
        )
        return "\n".join(entries) or "(пусто)"


def builtin_tools(workspace: Path) -> list[Tool]:
    return [
        BashTool(workspace),
        ReadFileTool(workspace),
        WriteFileTool(workspace),
        ListDirTool(workspace),
    ]
