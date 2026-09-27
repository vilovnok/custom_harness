"""Тот же агент, но НА ФРЕЙМВОРКЕ deepagents (LangChain) — для сравнения с самописным.

Что здесь делает библиотека за нас (в самописном варианте это отдельные модули):
  * агентный loop            -> внутри create_deep_agent (LangGraph)
  * планирование (todo-лист) -> встроенный инструмент write_todos
  * виртуальная ФС           -> встроенные ls / read_file / write_file / edit_file
  * субагенты + task()       -> параметр subagents=
  * история / checkpoint     -> checkpointer + thread_id
Мы приносим только: модель (DeepSeek), MCP-инструменты, скилы (как один tool),
пару своих инструментов и REPL.

Запуск:
  pip install deepagents langchain langchain-openai langchain-mcp-adapters
  export DEEPSEEK_API_KEY=sk-...
  python3 deepagents_app.py
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import subprocess
import sys

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent

from harness.config import load_config
from harness.skills import load_skills, skills_catalog


# --- модель: DeepSeek через OpenAI-совместимый клиент LangChain ------------------
def build_model(cfg) -> object:
    if cfg.provider == "anthropic":
        return init_chat_model(f"anthropic:{cfg.model or 'claude-opus-5'}")
    # deepseek / openai-совместимые
    return ChatOpenAI(
        model=cfg.model or "deepseek-v4-flash",
        base_url=cfg.base_url or "https://api.deepseek.com",
        api_key=cfg.api_key or os.environ.get("DEEPSEEK_API_KEY"),
        max_tokens=cfg.max_tokens,
    )


# --- свои инструменты (в формате LangChain @tool) ------------------------------
@tool
def run_bash(command: str) -> str:
    """Выполнить shell-команду в текущей директории и вернуть вывод."""
    proc = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=60
    )
    return f"(exit {proc.returncode})\n{proc.stdout}{proc.stderr}"


def make_load_skill(skills: dict):
    catalog = skills_catalog(skills)

    @tool(description=f"Загрузить полную инструкцию скила по имени. Доступно:\n{catalog}")
    def load_skill(name: str) -> str:
        skill = skills.get(name)
        return skill.body if skill else f"Нет скила '{name}'. Есть: {', '.join(skills)}"

    return load_skill


def mcp_config(cfg) -> dict:
    """config.toml [mcp.*] -> формат MultiServerMCPClient (stdio)."""
    return {
        s.prefix: {
            "transport": "stdio",
            "command": s.command,
            "args": s.args,
            "env": s.env,
        }
        for s in cfg.mcp_servers
    }


async def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.toml")
    skills = load_skills(cfg.skills_dir)

    servers = mcp_config(cfg)
    client = MultiServerMCPClient(servers) if servers else None
    mcp_tools = await client.get_tools() if client else []

    tools = [run_bash, make_load_skill(skills), *mcp_tools]
    by_name = {getattr(t, "name", ""): t for t in tools}

    # config.toml [[subagents]] -> формат deepagents.
    # Если фильтр tools по глобам ничего не выбрал — ключ опускаем, субагент
    # наследует все инструменты родителя.
    subagents = []
    for s in cfg.subagents:
        picked = [
            by_name[n]
            for n in by_name
            if s.tools and any(fnmatch.fnmatch(n, p) for p in s.tools)
        ]
        sub: dict = {
            "name": s.name,
            "description": s.description,
            "system_prompt": s.system_prompt,
        }
        if picked:
            sub["tools"] = picked
        subagents.append(sub)

    system_prompt = (
        cfg.system_prompt
        + "\n\n## Скилы (load_skill по необходимости)\n"
        + skills_catalog(skills)
    )

    agent = create_deep_agent(
        model=build_model(cfg),
        tools=tools,
        subagents=subagents,
        system_prompt=system_prompt,
    )

    print("deepagents-версия. Модель:", cfg.model, "| инструментов:", len(tools))
    print("(loop / todo / ФС / task — из коробки deepagents)\n")
    history: list = []
    while True:
        try:
            line = (await asyncio.to_thread(input, "› ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line in {"/quit", "/exit"}:
            break
        history.append({"role": "user", "content": line})
        result = await agent.ainvoke({"messages": history})
        history = result["messages"]
        print(f"\033[36m{history[-1].content}\033[0m")


if __name__ == "__main__":
    asyncio.run(main())
