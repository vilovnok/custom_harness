"""Скилы = markdown-инструкции с прогрессивным раскрытием.

В системный промпт попадает только каталог (имя + описание). Полный текст скила
модель подтягивает сама инструментом ``load_skill`` — или пользователь командой
``/skill <name>`` в CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.tools import Tool


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Мини-парсер YAML-фронтматтера (только ``key: value``), без зависимости от pyyaml."""
    if not text.startswith("---"):
        return {}, text
    _, front, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in front.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, body.strip()


def load_skills(skills_dir: Path) -> dict[str, Skill]:
    skills: dict[str, Skill] = {}
    if not skills_dir.is_dir():
        return skills
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        meta, body = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        name = meta.get("name", skill_md.parent.name)
        skills[name] = Skill(
            name=name,
            description=meta.get("description", ""),
            body=body,
            path=skill_md.parent,
        )
    return skills


def skills_catalog(skills: dict[str, Skill]) -> str:
    if not skills:
        return "(скилов нет)"
    return "\n".join(f"- {s.name}: {s.description}" for s in skills.values())


class LoadSkillTool(Tool):
    name = "load_skill"

    def __init__(self, skills: dict[str, Skill]) -> None:
        self._skills = skills
        self.description = (
            "Загрузить полную инструкцию скила по имени и следовать ей. Доступные скилы:\n"
            + skills_catalog(skills)
        )
        self.input_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

    async def run(self, arguments: dict[str, Any]) -> str:
        skill = self._skills.get(arguments["name"])
        if skill is None:
            return f"Скила '{arguments['name']}' нет. Доступны: {', '.join(self._skills)}"
        return f"# Скил: {skill.name}\n(каталог ресурсов: {skill.path})\n\n{skill.body}"
