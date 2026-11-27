"""MCP server configuration.

Two files, merged:

* ``~/.config/agentux/mcp.json``      -- servers you always want
* ``<project>/.agentux/mcp.json``     -- servers for one repository

Both use the same ``{"mcpServers": {...}}`` shape that Claude Desktop and the
MCP tooling use, so an existing config can be copied across verbatim::

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
          "env": {"LOG_LEVEL": "error"}
        }
      }
    }

Project entries win over global ones on a name collision, so a repository can
override what your machine does by default. An entry may set ``disabled: true``
to switch off a global server without deleting it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .errors import MCPConfigError

#: Where the machine-wide file lives, relative to the user's home directory.
GLOBAL_CONFIG_RELPATH = os.path.join(".config", "agentux", "mcp.json")

#: The per-project file, relative to the workspace root.
PROJECT_CONFIG_RELPATH = os.path.join(".agentux", "mcp.json")


class MCPServerConfig:
    """How to launch one MCP server."""

    __slots__ = ("name", "command", "args", "env", "cwd", "disabled", "source")

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        disabled: bool = False,
        source: str = "unknown",
    ) -> None:
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd
        self.disabled = disabled
        #: Which file this entry came from, for error messages and ``--list-tools``.
        self.source = source

    def __repr__(self) -> str:
        return (
            f"MCPServerConfig(name={self.name!r}, command={self.command!r}, "
            f"args={self.args!r}, source={self.source!r})"
        )

    @property
    def display(self) -> str:
        parts = [self.command, *self.args]
        return " ".join(parts)


def global_config_path() -> str:
    override = os.environ.get("AGENTUX_MCP_CONFIG")
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), GLOBAL_CONFIG_RELPATH)


def project_config_path(working_dir: str) -> str:
    return os.path.join(working_dir, PROJECT_CONFIG_RELPATH)


def _read_file(path: str) -> Dict[str, Any]:
    """Read one config file. A missing file is normal, not an error."""
    if not os.path.isfile(path):
        return {}

    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise MCPConfigError(
            f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc
    except OSError as exc:
        raise MCPConfigError(f"Could not read {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise MCPConfigError(
            f"{path} must contain a JSON object at the top level, "
            f"got {type(data).__name__}."
        )
    return data


def _parse_servers(data: Dict[str, Any], path: str, source: str) -> Dict[str, MCPServerConfig]:
    servers = data.get("mcpServers")
    if servers is None:
        # Tolerate a bare mapping of name -> entry, which is what a lot of
        # people write by hand.
        if all(isinstance(v, dict) and ("command" in v or "url" in v) for v in data.values()):
            servers = data
        else:
            return {}

    if not isinstance(servers, dict):
        raise MCPConfigError(
            f"'mcpServers' in {path} must be an object mapping names to server "
            f"definitions, got {type(servers).__name__}."
        )

    parsed: Dict[str, MCPServerConfig] = {}
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            raise MCPConfigError(
                f"Server {name!r} in {path} must be an object, got {type(entry).__name__}."
            )

        command = entry.get("command")
        if not command:
            # Remote servers exist in MCP, but this client speaks stdio only.
            # Saying so beats failing later with a confusing spawn error.
            if entry.get("url"):
                raise MCPConfigError(
                    f"Server {name!r} in {path} uses a remote 'url'. This Agentux "
                    "build supports stdio servers only -- give it a 'command'."
                )
            raise MCPConfigError(
                f"Server {name!r} in {path} has no 'command'. A stdio server needs "
                "at least a command, e.g. {\"command\": \"npx\", \"args\": [...]}."
            )

        args = entry.get("args") or []
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise MCPConfigError(
                f"'args' for server {name!r} in {path} must be a list of strings."
            )

        env = entry.get("env") or {}
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            raise MCPConfigError(
                f"'env' for server {name!r} in {path} must map strings to strings."
            )

        cwd = entry.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise MCPConfigError(f"'cwd' for server {name!r} in {path} must be a string.")

        parsed[name] = MCPServerConfig(
            name=name,
            command=str(command),
            args=args,
            env=env,
            cwd=cwd,
            disabled=bool(entry.get("disabled", False)),
            source=source,
        )

    return parsed


def load_server_configs(working_dir: str) -> Tuple[List[MCPServerConfig], List[str]]:
    """Load and merge the global and per-project MCP configs.

    Returns the enabled servers and a list of warnings for anything skipped, so
    the caller can report problems without failing the session.
    """
    warnings: List[str] = []
    merged: Dict[str, MCPServerConfig] = {}

    for path, source in (
        (global_config_path(), "global"),
        (project_config_path(working_dir), "project"),
    ):
        try:
            data = _read_file(path)
        except MCPConfigError as exc:
            warnings.append(str(exc))
            continue
        try:
            servers = _parse_servers(data, path, source)
        except MCPConfigError as exc:
            warnings.append(str(exc))
            continue
        merged.update(servers)  # later sources win on a name collision

    for name, server in merged.items():
        if server.disabled:
            warnings.append(f"MCP server {name!r} is disabled ({server.source} config).")

    return [s for s in merged.values() if not s.disabled], warnings


__all__ = [
    "MCPServerConfig",
    "load_server_configs",
    "global_config_path",
    "project_config_path",
    "GLOBAL_CONFIG_RELPATH",
    "PROJECT_CONFIG_RELPATH",
]
