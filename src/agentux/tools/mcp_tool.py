"""Exposing MCP server tools to the agent.

Each remote tool becomes an ordinary :class:`~agentux.tools.base.BaseTool`, so
the agent loop cannot tell the difference between a tool Agentux implements and
one that lives in another process. That is the point: adding MCP changes what
the model can call without changing how the loop calls it.

Remote tools are namespaced ``mcp__<server>__<tool>``. Namespacing is not
cosmetic -- a server is free to publish a tool called ``shell`` or ``write``,
and without a prefix it would silently shadow a built-in one.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from ..errors import MCPError
from ..mcp import DEFAULT_TIMEOUT, MCPServer, render_tool_result
from ..mcp_config import MCPServerConfig, load_server_configs
from ..models import ToolDefinition
from .base import BaseTool

#: Separator between the server name and the tool name.
SEPARATOR = "__"


def qualify(server: str, tool: str) -> str:
    """The name the model uses to call a remote tool."""
    return f"mcp{SEPARATOR}{server}{SEPARATOR}{tool}"


def split(name: str) -> Optional[Tuple[str, str]]:
    """Inverse of :func:`qualify`, or ``None`` if the name is not a remote tool."""
    if not name.startswith(f"mcp{SEPARATOR}"):
        return None
    remainder = name[len(f"mcp{SEPARATOR}"):]
    server, separator, tool = remainder.partition(SEPARATOR)
    if not separator or not server or not tool:
        return None
    return server, tool


class MCPTool(BaseTool):
    """One tool provided by an MCP server."""

    def __init__(
        self,
        server: MCPServer,
        remote_name: str,
        description: str,
        schema: Dict[str, Any],
    ) -> None:
        self.server = server
        self.remote_name = remote_name
        self._description = description or "No description provided by the server."
        self._schema = schema or {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return qualify(self.server.name, self.remote_name)

    @property
    def description(self) -> str:
        return f"[MCP server {self.server.name!r}] {self._description}"

    @property
    def parameters(self) -> Dict[str, Any]:
        # Servers send JSON Schema; providers want the same shape. Only the
        # fields that vary are rewritten.
        schema = dict(self._schema)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description, parameters=self.parameters
        )

    async def execute(self, **kwargs: Any) -> str:
        result = await self.server.call_tool(self.remote_name, kwargs)

        text = render_tool_result(result)
        if result.get("isError"):
            # Surfaced as text, not raised: the spec is explicit that these are
            # the errors a model can act on, and aborting the run would deny it
            # the chance to retry with better arguments.
            return f"ERROR (from MCP server {self.server.name!r}): {text}"
        return text


class MCPToolManager:
    """Owns the running MCP servers and the tools they provide."""

    def __init__(
        self,
        working_dir: str,
        timeout: float = DEFAULT_TIMEOUT,
        on_log: Optional[Any] = None,
    ) -> None:
        self.working_dir = working_dir
        self.timeout = timeout
        self.on_log = on_log
        self.servers: Dict[str, MCPServer] = {}
        self.tools: List[MCPTool] = []
        self.warnings: List[str] = []
        self.configs: List[MCPServerConfig] = []
        #: (server name, tool name, reason) for each server that failed to load.
        self.failed: List[Tuple[str, str]] = []

    def _log(self, message: str) -> None:
        if self.on_log:
            try:
                self.on_log(message)
            except Exception:
                pass

    async def load(self, start: bool = True) -> List[MCPTool]:
        """Load configured servers and their tools.

        A server that fails to start is recorded and skipped. One broken server
        must not take down the session -- Agentux's own tools still work.

        Returns a snapshot, not the live list: :meth:`close` clears the
        internal one, and a caller that kept the result should not find it
        emptied out from under it.
        """
        self.configs, self.warnings = load_server_configs(self.working_dir)

        for config in self.configs:
            server = MCPServer(config, timeout=self.timeout, on_log=self._log)
            self.servers[config.name] = server

            if not start:
                continue

            try:
                await server.start()
                remote_tools = await server.list_tools()
            except (MCPError, asyncio.TimeoutError, OSError) as exc:
                self.failed.append((config.name, str(exc)))
                self._log(f"skipping MCP server {config.name!r}: {exc}")
                await server.close()
                continue

            for entry in remote_tools:
                remote_name = entry.get("name")
                if not remote_name:
                    continue
                self.tools.append(
                    MCPTool(
                        server=server,
                        remote_name=str(remote_name),
                        description=str(entry.get("description") or ""),
                        schema=entry.get("inputSchema") or {},
                    )
                )

            self._log(
                f"MCP server {config.name!r} ready: {len(remote_tools)} tool(s)"
            )

        return list(self.tools)

    def describe(self) -> List[str]:
        """One line per configured server, for ``--list-tools`` and ``/mcp``."""
        lines: List[str] = []
        for name, server in self.servers.items():
            info = server.server_info or {}
            label = info.get("name") or config_name(server.config)
            version = info.get("version")
            detail = f"{label}" + (f" {version}" if version else "")
            if name in {f for f, _ in self.failed}:
                reason = next(r for f, r in self.failed if f == name)
                lines.append(f"{name}: unavailable — {reason}")
            else:
                count = sum(1 for t in self.tools if t.server.name == name)
                lines.append(f"{name}: {detail} — {count} tool(s)")
        return lines

    async def close(self) -> None:
        for server in self.servers.values():
            try:
                await server.close()
            except Exception:
                pass
        self.servers.clear()
        self.tools.clear()


def config_name(config: MCPServerConfig) -> str:
    return f"{config.command} {' '.join(config.args)}".strip()


__all__ = ["MCPTool", "MCPToolManager", "qualify", "split", "SEPARATOR"]
