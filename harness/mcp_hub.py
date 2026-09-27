"""Подключение stdio MCP-серверов и превращение их инструментов в объекты Tool.

MCP-сессия — это async-контекст, который должен жить всё время работы харнеса,
поэтому все сессии держим открытыми в общем AsyncExitStack.
"""

from __future__ import annotations

import re
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from harness.config import MCPServerSpec
from harness.tools import Tool

if TYPE_CHECKING:
    from mcp import ClientSession

_SANITIZE = re.compile(r"[^a-zA-Z0-9_-]")


class MCPTool(Tool):
    """Инструмент MCP-сервера, проксирующий вызов в ``session.call_tool``."""

    def __init__(self, session: "ClientSession", remote_name: str, prefix: str, mcp_tool: Any) -> None:
        self._session = session
        self._remote_name = remote_name
        self.name = _SANITIZE.sub("_", f"{prefix}__{remote_name}")[:64]
        self.description = (mcp_tool.description or "").strip() or f"MCP-инструмент {remote_name}"
        self.input_schema = mcp_tool.inputSchema or {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any]) -> str:
        result = await self._session.call_tool(self._remote_name, arguments)
        chunks: list[str] = []
        for block in result.content:
            chunks.append(getattr(block, "text", None) or str(block))
        text = "\n".join(chunks)
        return f"[MCP error] {text}" if result.isError else text


class MCPHub:
    """Открывает все сконфигурированные MCP-серверы и собирает их инструменты."""

    def __init__(self, servers: list[MCPServerSpec]) -> None:
        self._servers = servers
        self._stack = AsyncExitStack()
        self.tools: list[Tool] = []

    async def __aenter__(self) -> "MCPHub":
        if not self._servers:
            return self
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        for spec in self._servers:
            params = StdioServerParameters(
                command=spec.command, args=spec.args, env=spec.env
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
            for mcp_tool in listed.tools:
                self.tools.append(MCPTool(session, mcp_tool.name, spec.prefix, mcp_tool))
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()
