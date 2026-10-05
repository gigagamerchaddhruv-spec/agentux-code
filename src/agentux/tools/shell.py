"""Shell tool.

Runs commands with a timeout and captures stdout, stderr, and the exit code.
Destructive commands are not blocked outright -- they are routed to a
confirmation callback so the user sees the exact command and decides. That
callback is injected by the CLI, which keeps this module testable and free of
terminal I/O.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, Optional

from .. import safety
from ..errors import SafetyError, ToolError
from .base import BaseTool

#: Signature of the confirmation hook: ``(command, reasons) -> bool``.
ConfirmCallback = Callable[[str, list], bool]

#: Ceiling on captured output, so one noisy command cannot flood the context.
MAX_OUTPUT_CHARS = 30_000


class ShellTool(BaseTool):
    """Execute shell commands in the workspace."""

    def __init__(
        self,
        cwd: Optional[str] = None,
        timeout: float = 120.0,
        confirm: Optional[ConfirmCallback] = None,
        allow_destructive: bool = False,
    ) -> None:
        self.cwd = cwd or os.getcwd()
        self.timeout = timeout
        #: Called for destructive commands. ``None`` means "always deny".
        self.confirm = confirm
        #: When True, destructive commands run without asking. Opt-in only.
        self.allow_destructive = allow_destructive

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command in the workspace and return its stdout, "
            "stderr, and exit code. Use it for tests, builds, git status, and "
            "searching files. Prefer reading files with the filesystem tool when "
            "you only need their contents. Destructive commands require the "
            "user's confirmation."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Working directory relative to the workspace root. "
                        "Defaults to the workspace root."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds. Defaults to the tool's configured timeout.",
                },
            },
            "required": ["command"],
        }

    def _resolve_cwd(self, cwd: Optional[str]) -> str:
        if not cwd:
            return self.cwd
        candidate = cwd if os.path.isabs(cwd) else os.path.join(self.cwd, cwd)
        resolved = os.path.realpath(os.path.expanduser(candidate))
        if not os.path.isdir(resolved):
            raise ToolError(f"Working directory does not exist: {cwd}")
        return resolved

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        half = MAX_OUTPUT_CHARS // 2
        return (
            text[:half]
            + f"\n\n... [truncated {len(text) - MAX_OUTPUT_CHARS} characters] ...\n\n"
            + text[-half:]
        )

    async def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        if not command or not str(command).strip():
            raise ToolError("A non-empty 'command' is required.")

        command = str(command).strip()
        limit = float(timeout) if timeout else self.timeout

        reasons = safety.scan(command)
        if reasons and not self.allow_destructive:
            approved = self._ask_permission(command, reasons)
            if not approved:
                raise SafetyError(
                    f"The user declined to run this command: {command}\n"
                    f"Reason it was flagged: {', '.join(reasons)}.\n"
                    "Do not retry it. Ask the user how to proceed, or find a "
                    "safer alternative."
                )

        workdir = self._resolve_cwd(cwd)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env={**os.environ, "AGENTUX": "1"},
            )
        except OSError as exc:
            raise ToolError(f"Failed to start command: {exc}") from exc

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=limit)
        except asyncio.TimeoutError:
            # Kill the whole process group: a shell pipeline's children can
            # outlive the shell itself and keep writing to the pipes.
            self._terminate(process)
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            raise ToolError(
                f"Command timed out after {limit:g}s and was killed: {command}\n"
                "Use a smaller scope, add a timeout to the command itself, or "
                "raise the tool's timeout."
            ) from None

        out = self._truncate(stdout.decode("utf-8", errors="replace").rstrip())
        err = self._truncate(stderr.decode("utf-8", errors="replace").rstrip())
        code = process.returncode

        parts = []
        if out:
            parts.append(f"STDOUT:\n{out}")
        if err:
            parts.append(f"STDERR:\n{err}")
        if not out and not err:
            parts.append("(no output)")
        parts.append(f"EXIT CODE: {code}")
        return "\n".join(parts)

    def _ask_permission(self, command: str, reasons: list) -> bool:
        if self.confirm is None:
            raise SafetyError(
                f"This command looks destructive ({', '.join(reasons)}) and "
                f"cannot run without confirmation: {command}\n"
                "It was not executed. Tell the user what you wanted to run and "
                "why, and ask them to run it themselves if it is essential."
            )
        return bool(self.confirm(command, reasons))

    @staticmethod
    def _terminate(process) -> None:
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass


__all__ = ["ShellTool", "ConfirmCallback"]
