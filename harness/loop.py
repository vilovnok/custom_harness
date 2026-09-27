"""Сердце харнеса — агентный tool-use loop.

Провайдер-агностичен: работает только с нормализованным `Turn` и владеет
`history` через методы провайдера. Крутит «шаг модели → выполнить инструменты →
вернуть результаты», пока модель не перестанет звать инструменты.
"""

from __future__ import annotations

from typing import Any, Callable

from harness.providers.base import Provider, ToolOutcome
from harness.tools import ToolRegistry

OnText = Callable[[str], None]
OnTool = Callable[[str, dict[str, Any]], None]


async def run_loop(
    provider: Provider,
    *,
    system: str,
    history: list[Any],
    registry: ToolRegistry,
    max_iters: int = 40,
    on_text: OnText | None = None,
    on_tool: OnTool | None = None,
) -> str:
    """Прогнать loop. ``history`` мутируется на месте (это диалог в формате провайдера)."""
    last_text = ""

    for _ in range(max_iters):
        turn = await provider.step(
            system=system, history=history, tool_specs=registry.specs()
        )

        if turn.text:
            last_text = turn.text
            if on_text:
                on_text(turn.text)

        if not turn.tool_calls:
            return last_text

        outcomes: list[ToolOutcome] = []
        for call in turn.tool_calls:
            if on_tool:
                on_tool(call.name, call.arguments)
            output, is_error = await registry.dispatch(call.name, call.arguments)
            outcomes.append(
                ToolOutcome(
                    call_id=call.id, name=call.name, content=output, is_error=is_error
                )
            )
        provider.add_tool_outcomes(history, outcomes)

    return last_text or "Достигнут лимит итераций loop."
