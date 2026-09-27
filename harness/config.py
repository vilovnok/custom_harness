"""Загрузка конфигурации из TOML в типизированные датаклассы."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.10 и ниже: pip install tomli
    import tomli as tomllib


@dataclass(slots=True)
class SubagentSpec:
    """Описание субагента-исполнителя."""

    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    model: str | None = None


@dataclass(slots=True)
class MCPServerSpec:
    """Как запустить один stdio MCP-сервер."""

    prefix: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


@dataclass(slots=True)
class Config:
    provider: str  # anthropic | openai | deepseek
    model: str | None  # None -> дефолт провайдера
    max_tokens: int
    workspace: Path
    skills_dir: Path
    system_prompt: str
    base_url: str | None = None  # переопределить endpoint провайдера
    api_key: str | None = None  # иначе берётся из env
    thinking: bool = False  # DeepSeek thinking-режим
    reasoning_effort: str | None = None  # low | high | max (DeepSeek)
    subagents: list[SubagentSpec] = field(default_factory=list)
    mcp_servers: list[MCPServerSpec] = field(default_factory=list)


def load_config(path: str | Path = "config.toml") -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    base = Path(path).resolve().parent

    subagents = [
        SubagentSpec(
            name=item["name"],
            description=item["description"],
            system_prompt=item.get("system_prompt", ""),
            tools=list(item.get("tools", [])),
            model=item.get("model"),
        )
        for item in raw.get("subagents", [])
    ]

    mcp_servers = [
        MCPServerSpec(
            prefix=prefix,
            command=spec["command"],
            args=list(spec.get("args", [])),
            env=spec.get("env"),
        )
        for prefix, spec in raw.get("mcp", {}).items()
    ]

    workspace = (base / raw.get("workspace", "./workspace")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    return Config(
        provider=raw.get("provider", "anthropic"),
        model=raw.get("model") or None,
        max_tokens=int(raw.get("max_tokens", 8000)),
        workspace=workspace,
        skills_dir=(base / raw.get("skills_dir", "./skills")).resolve(),
        system_prompt=raw.get("system_prompt", "Ты — полезный агент.").strip(),
        base_url=raw.get("base_url") or None,
        api_key=raw.get("api_key") or None,
        thinking=bool(raw.get("thinking", False)),
        reasoning_effort=raw.get("reasoning_effort") or None,
        subagents=subagents,
        mcp_servers=mcp_servers,
    )
