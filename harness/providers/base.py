"""Провайдер-абстракция: единый интерфейс поверх разных LLM API.

Loop и инструменты не знают, чей это API. Провайдер отвечает за две вещи:
  1. вызвать модель и вернуть **нормализованный** результат одного шага (`Turn`);
  2. владеть форматом истории сообщений (у Anthropic и OpenAI он разный) —
     поэтому именно провайдер добавляет в историю ответ ассистента,
     пользовательские сообщения и результаты инструментов.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

# Нейтральная спецификация инструмента, как её отдаёт ToolRegistry.specs():
#   {"name": str, "description": str, "input_schema": <JSON Schema>}
ToolSpec = dict[str, Any]


@dataclass(slots=True)
class ToolCall:
    """Запрос модели вызвать инструмент."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolOutcome:
    """Результат выполнения инструмента, который вернём модели."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(slots=True)
class Turn:
    """Нормализованный ответ модели за один шаг loop."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class Provider(abc.ABC):
    name: str
    model: str

    @abc.abstractmethod
    async def step(
        self, *, system: str, history: list[Any], tool_specs: list[ToolSpec]
    ) -> Turn:
        """Вызвать модель. Реализация ОБЯЗАНА добавить ответ ассистента в ``history``."""

    @abc.abstractmethod
    def add_user_message(self, history: list[Any], text: str) -> None: ...

    @abc.abstractmethod
    def add_tool_outcomes(self, history: list[Any], outcomes: list[ToolOutcome]) -> None: ...

    @abc.abstractmethod
    def clone(self, model: str | None) -> "Provider":
        """Копия провайдера с другой моделью (для субагентов)."""
