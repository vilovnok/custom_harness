# mini-harness — учебный CLI-харнес для агента

Маленький, читаемый за один присест харнес: **свой агентный loop**, сменный
**LLM-провайдер** (Anthropic / DeepSeek / OpenAI-совместимые), подключение
**MCP**-серверов, **скилы** с прогрессивным раскрытием, реестр **инструментов**
и **исполнитель субагентов** (`task`). Ядро — ~1000 строк с комментариями, Python 3.10+.

**Статус:** оффлайн-тесты (`tests/`) зелёные; путь DeepSeek проверен на замоканном
HTTP-слое; deepagents-вариант собирается на Python 3.12. Живых вызовов к API в репозитории нет.

> Это **самописный** харнес, а не обёртка над пакетом `deepagents`. Фреймворки
> `deepagents` (LangChain) и DeerFlow здесь — **источник идей**, а не зависимость:
> цель — увидеть, из чего собран агентный харнес, а не спрятать loop за `create_deep_agent()`.
> Единственные внешние пакеты — `mcp` и SDK выбранного провайдера (`anthropic` / `openai`).
>
> Для сравнения рядом лежит [`deepagents_app.py`](deepagents_app.py) — **тот же агент на фреймворке
> deepagents** (~120 строк, но loop/todo/ФС/субагенты — из коробки). См. раздел 7.

---

## 1. Что такое «харнес» и что подсмотрено у deepagents и DeerFlow

**Харнес** (agent harness) — это обвязка вокруг голого вызова LLM, которая
превращает «модель отвечает текстом» в «агент делает работу». По формулировке
самих deepagents: *«agent harness that utilizes a core tool-calling loop enhanced
with built-in capabilities»* — то есть **цикл вызова инструментов** + набор
встроенных возможностей поверх него.

### deepagents (LangChain) — что берём как идею

| Возможность deepagents | Здесь |
| --- | --- |
| Core tool-calling loop | [`harness/loop.py`](harness/loop.py) |
| Смена LLM-провайдера / model resolution | [`harness/providers/`](harness/providers/) (Anthropic + OpenAI/DeepSeek) |
| Sub-agents + инструмент `task()` (изолированный контекст, наружу — только итог) | [`harness/subagents.py`](harness/subagents.py) |
| Виртуальная ФС / файловые инструменты | `bash` / `read_file` / `write_file` / `list_dir` в [`harness/tools.py`](harness/tools.py) |
| Подключение инструментов из любого MCP-сервера | [`harness/mcp_hub.py`](harness/mcp_hub.py) |
| Detailed prompts / system prompt как каркас | шаблон в [`harness/agent.py`](harness/agent.py) |
| Planning tool (todo-лист), middleware, human-in-the-loop, checkpointer | **не реализовано** — см. раздел 6 |

### DeerFlow (ByteDance) — что берём как идею

DeerFlow называет себя «super agent harness» на LangGraph. Оттуда:

| Возможность DeerFlow | Здесь |
| --- | --- |
| Runtime = стейт-машина, гоняющая «LLM → инструмент → повтор» | наш `run_loop` (без графа, просто `while`) |
| MCP по stdio **и** HTTP/SSE, имена инструментов неймспейсятся | stdio + префикс сервера (`fs__read_file`); HTTP — TODO |
| Skills = markdown-модули, **прогрессивная загрузка**, активация `/skill-name`, фильтр `allowed-tools` | [`harness/skills.py`](harness/skills.py) + команда `/skill` в CLI |
| Делегирование в субагентов инструментом `task`, у каждого своя модель/контекст/инструменты | `SubagentSpec` + `TaskTool` |
| Config-driven: модели, MCP, скилы в конфиге, а не в коде | [`config.toml`](config.toml) |
| Checkpoint/memory в SQLite/Postgres, sandbox в Docker/K8s, OAuth, мульти-воркер | **вне рамок** учебного примера |

---

## 2. Карта модулей

```
config.toml            конфиг: модель, workspace, MCP-серверы, субагенты
cli.py                 REPL: свой цикл ввода, слэш-команды, трассировка

harness/
  config.py            TOML -> датаклассы (Config, SubagentSpec, MCPServerSpec)
  providers/           сменный LLM-провайдер (единый интерфейс поверх разных API)
    base.py            Provider (ABC) + нормализованные Turn / ToolCall / ToolOutcome
    anthropic_provider.py   Claude Messages API
    openai_provider.py      OpenAI / DeepSeek / любой OpenAI-совместимый
    __init__.py             build_provider(cfg) — фабрика по конфигу
  tools.py             Tool (ABC) + ToolRegistry + встроенные инструменты
  mcp_hub.py           открывает stdio MCP-серверы, оборачивает их tools в Tool
  skills.py            парсинг SKILL.md, каталог, инструмент load_skill
  subagents.py         Subagent (вложенный loop) + инструмент task
  loop.py              агентный цикл tool-use (провайдер-агностичный)
  agent.py             Harness: собирает всё вместе, ведёт историю диалога

deepagents_app.py                тот же агент, но на фреймворке deepagents (раздел 7)
skills/web-research/SKILL.md      пример скила
tests/smoke_offline.py            loop + субагенты на фейковом провайдере
tests/smoke_openai_provider.py    путь DeepSeek на замоканном HTTP-слое
```

Две ключевые абстракции:

* **`Tool`** — объект с `name` / `description` / `input_schema` и корутиной
  `run(arguments) -> str`. Реестру всё равно, откуда инструмент: встроенный, из MCP,
  обёртка над субагентом (`task`) или над скилом (`load_skill`).
* **`Provider`** — единый интерфейс поверх LLM API. Loop работает только с
  нормализованным `Turn`, а форматом истории (у Anthropic — блоки, у OpenAI —
  плоские роли) владеет провайдер. Смена LLM = одна строка в `config.toml`.

---

## 3. Как работает loop

`run_loop` (в [`harness/loop.py`](harness/loop.py)) — это весь «интеллект» харнеса:

```
provider.add_user_message(history, запрос)
повторять (до лимита итераций):
    turn = provider.step(system, history, tools=реестр.specs())  # + кладёт ответ в history
    если turn.text: показать
    если не turn.tool_calls:
        return turn.text                              # агент закончил
    для каждого call в turn.tool_calls:
        out, is_error = реестр.dispatch(call.name, call.arguments)
    provider.add_tool_outcomes(history, [...])         # вернуть все результаты разом
```

Важные детали, которые легко упустить:

* **все результаты инструментов возвращаются разом** — иначе модель
  разучивается вызывать инструменты параллельно;
* ответ ассистента кладётся в историю **целиком** (у Anthropic — с `tool_use`-блоками,
  у OpenAI — с `tool_calls`) — провайдер сам знает формат;
* ошибка инструмента — это не исключение, а результат с флагом ошибки;
  loop не должен падать из-за инструмента;
* результат инструмента обрезается (`MAX_OUTPUT`), чтобы не взорвать контекст;
* «агент закончил» = провайдер вернул `Turn` без `tool_calls` (Anthropic:
  `stop_reason != "tool_use"`; OpenAI/DeepSeek: `message.tool_calls is None`).

Субагент — это **тот же `run_loop`**, но со своей `history`, урезанным реестром,
без `task` и, при желании, с другой моделью (`SubagentSpec.model` → `provider.clone`).
Родитель получает только финальную строку.

---

## 4. Как расширять

### Добавить свой инструмент
```python
from harness.tools import FunctionTool

async def _now(tz: str = "UTC") -> str:
    ...
clock = FunctionTool(
    name="clock",
    description="Текущее время в зоне tz.",
    input_schema={"type": "object", "properties": {"tz": {"type": "string"}}},
    func=_now,
)
harness.registry.register(clock)   # до первого chat()
```
Либо наследуйтесь от `Tool` для инструментов с состоянием.

### Подключить MCP-сервер
В `config.toml` добавьте секцию `[mcp.<префикс>]`:
```toml
[mcp.git]
command = "uvx"
args = ["mcp-server-git", "--repository", "./workspace"]
```
Инструменты появятся как `git__<имя>` и попадут в реестр автоматически.

### Добавить скил
Создайте `skills/<имя>/SKILL.md` с фронтматтером:
```markdown
---
name: pdf-report
description: Когда и как собирать PDF-отчёт
---
# шаги...
```
В системный промпт попадёт только строка каталога. Полный текст модель возьмёт
инструментом `load_skill`, пользователь — командой `/skill pdf-report`.

### Добавить субагента
Секция `[[subagents]]` в `config.toml`: `name`, `description`, `system_prompt`,
опционально `tools` (список имён / glob вроде `fs__*`) и `model`.

---

## 5. Запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mcp tomli                    # tomli только для Python < 3.11
pip install openai                       # для DeepSeek / OpenAI
#   или: pip install anthropic           # для Claude

export DEEPSEEK_API_KEY=sk-...           # ключ по провайдеру из config.toml
#   ANTHROPIC_API_KEY / OPENAI_API_KEY — для других провайдеров

python3 cli.py                           # REPL
python3 tests/smoke_offline.py           # логика loop/субагентов без сети
python3 tests/smoke_openai_provider.py   # путь DeepSeek без сети
```

Команды REPL: `/help` `/tools` `/skills` `/skill <name>` `/agents` `/reset` `/quit`.

### Провайдер и модель — в [`config.toml`](config.toml)

```toml
provider = "deepseek"          # anthropic | openai | deepseek
model    = "deepseek-v4-flash" # пусто -> дефолт провайдера; "deepseek-v4-pro" для агентных задач
thinking = false               # DeepSeek reasoning-режим (+ reasoning_effort = "high")
```

DeepSeek — это OpenAI-совместимый Chat Completions API (`base_url =
https://api.deepseek.com`), поэтому идёт через тот же `OpenAIProvider`. У DeepSeek
есть и Anthropic-совместимый endpoint (`https://api.deepseek.com/anthropic`) —
тогда можно было бы обойтись `AnthropicProvider` с другим `base_url`, но
function calling надёжнее по «родному» OpenAI-пути.

Добавить свой провайдер: новый класс от `Provider` в `harness/providers/` +
строчка в `_PRESETS`. Loop, инструменты, скилы и субагенты не меняются.

---

## 6. Что намеренно упрощено (и куда расти)

| Тема | Здесь | Как в «взрослых» харнесах |
| --- | --- | --- |
| Стриминг ответа | нет, ждём финальный текст | `client.messages.stream()` + рендер токенов |
| Провайдеры | Anthropic + OpenAI/DeepSeek, вручную | litellm (100+ провайдеров) в DeerFlow |
| Планирование | нет | `write_todos` / planning-инструмент (deepagents), граф узлов (DeerFlow) |
| Персистентность диалога | в памяти процесса | checkpointer в SQLite/Postgres, thread_id |
| Human-in-the-loop | нет | `interrupt_on` для опасных инструментов, подтверждение в CLI |
| Сжатие контекста | грубая обрезка вывода | server-side compaction / context editing / memory-инструмент |
| MCP transport | только stdio | + HTTP/SSE, OAuth, таймауты на вызов |
| Ограничение прав | `tools` у субагента | `allowed-tools` у скила, песочницы, политики |
| Рекурсия субагентов | запрещена (нет `task`) | контроль глубины, бюджеты токенов |

Начать расширение проще всего со стриминга и `write_todos` — это самые заметные
по эффекту добавления при минимуме кода.

---

## 7. Вариант на фреймворке deepagents ([`deepagents_app.py`](deepagents_app.py))

Тот же агент с DeepSeek, но loop, планирование (`write_todos`), виртуальная ФС,
`task()` и история — **из коробки** `deepagents`. Мы приносим только модель,
MCP-инструменты, скилы (как один tool) и REPL.

```bash
# требует Python >= 3.11 (deepagents не ставится на 3.10) — отдельный venv
python3.12 -m venv .venv-da && source .venv-da/bin/activate
pip install deepagents langchain langchain-openai langchain-mcp-adapters
export DEEPSEEK_API_KEY=sk-...
python deepagents_app.py
```

Модель DeepSeek подаётся как `ChatOpenAI(base_url="https://api.deepseek.com", ...)`
(deepagents принимает любой LangChain chat-model с поддержкой tool calling).

| | самописный (`cli.py`) | на фреймворке (`deepagents_app.py`) |
| --- | --- | --- |
| Агентный loop | свой, `harness/loop.py` (виден, правится) | внутри `create_deep_agent` (LangGraph) |
| Планирование (todo) | нет | `write_todos` из коробки |
| Файловые инструменты | свои 4 инструмента | `ls/read_file/write_file/edit_file` из коробки |
| Субагенты / `task` | свой `harness/subagents.py` | параметр `subagents=` |
| История / checkpoint | список в памяти | `checkpointer` + `thread_id` |
| Провайдеры | свой `Provider` (2 шт.) | `init_chat_model` / любой LangChain-model |
| MCP | свой `mcp_hub.py` (stdio) | `langchain-mcp-adapters` (stdio + HTTP) |
| Скилы (markdown) | свой загрузчик + `/skill` | свёрнуты в один `@tool load_skill` |
| Строк кода | ~1000 | ~140 + большой стек зависимостей |
| Python | 3.10+ | 3.11+ |

Вывод: фреймворк экономит код и даёт готовые планирование/ФС/persistence, но
loop и контекст-менеджмент становятся «чёрным ящиком». Для ознакомления полезны
оба — сначала собрать руками, потом посмотреть, что за тебя делает библиотека.