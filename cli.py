"""CLI-REPL харнеса: свой цикл ввода, слэш-команды, трассировка инструментов."""

from __future__ import annotations

import asyncio
import sys

from harness import load_config
from harness.agent import Harness

BANNER = """mini-harness — учебный CLI-агент
  /help  /tools  /skills  /skill <name>  /agents  /reset  /quit
"""


def _print_tool_call(name: str, args: dict) -> None:
    preview = ", ".join(f"{k}={v!r}"[:60] for k, v in args.items())
    print(f"  \033[90m→ {name}({preview})\033[0m")


def _print_text(text: str) -> None:
    print(f"\033[36m{text}\033[0m")


async def _handle_command(harness: Harness, line: str) -> bool:
    """Вернёт False, если пора выходить."""
    cmd, _, arg = line[1:].partition(" ")
    arg = arg.strip()

    if cmd in {"quit", "exit", "q"}:
        return False
    if cmd == "help":
        print(BANNER)
    elif cmd == "tools":
        print("\n".join(harness.registry.names()))
    elif cmd == "skills":
        for name, skill in harness.skills.items():
            print(f"{name}: {skill.description}")
    elif cmd == "skill":
        print(harness.inject_skill(arg))
    elif cmd == "agents":
        for spec in harness.config.subagents:
            print(f"{spec.name}: {spec.description}")
    elif cmd == "reset":
        harness.reset()
        print("История очищена.")
    else:
        print(f"Неизвестная команда: /{cmd}")
    return True


async def main() -> None:
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.toml")
    async with Harness(config) as harness:
        print(BANNER)
        print(f"Модель: {config.model} | инструментов: {len(harness.registry.names())}\n")
        while True:
            try:
                line = (await asyncio.to_thread(input, "\033[1m› \033[0m")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.startswith("/"):
                if not await _handle_command(harness, line):
                    break
                continue
            await harness.chat(line, on_text=_print_text, on_tool=_print_tool_call)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
