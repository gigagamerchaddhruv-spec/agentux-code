"""The Agentux agent loop.

This module contains no HTTP and no provider-specific code. It talks to a
:class:`~agentux.providers.base.BaseProvider` and a
:class:`~agentux.tools.registry.ToolRegistry`, which is what makes the
provider layer swappable.

The loop is::

    user message
        |
        v
    provider.chat(history, tools)  <-----------+
        |                                        |
        +-- no tool calls --> return content    |
        |                                        |
        +-- tool calls --> validate, execute,   |
        |                    append results ----+
"""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable, List, Optional

from .config import Config
from .errors import (
    AgentuxError,
    MaxStepsError,
    ToolNotFoundError,
    UnsupportedToolCallingError,
)
from .models import Message, ToolCall
from .providers.base import BaseProvider
from .tools.registry import ToolRegistry

#: Called with each message of the run, for CLI display. ``(role, text)``.
EventCallback = Callable[[str, str], None]

SYSTEM_PROMPT = """\
You are Agentux, a terminal-based coding and automation agent working inside \
the user's project.

Work by observing before acting:
- Call the 'project' tool once at the start to learn what kind of repository \
this is, then use the 'filesystem' and 'shell' tools to gather specifics.
- Read a file before editing it. Use the 'edit' operation for changes to part \
of a file; it replaces only the exact text you target and reports a diff.
- Prefer 'edit' over 'write'. Use 'write' only for new files or full rewrites.
- Run tests or builds with the 'shell' tool when that verifies the work, and \
read what they actually printed before claiming success.

Rules:
- Never invent file contents, test results, or command output. Report only \
what you actually observed.
- If a tool fails, read the error and adapt rather than repeating the same \
call.
- Destructive commands are gated by the user. If one is declined, do not retry \
it; say what you wanted to do instead.
- Stop calling tools once you can answer, and answer concisely.

When you finish, state plainly what you did, what you verified, and anything \
the user still needs to decide.\
"""


class Agent:
    """Runs a multi-step tool-using conversation against a provider."""

    def __init__(
        self,
        config: Config,
        provider: BaseProvider,
        registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        on_event: Optional[EventCallback] = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.registry = registry if registry is not None else ToolRegistry()
        self.on_event = on_event
        self.history: List[Message] = []
        #: Counts tool calls across the whole session, for reporting.
        self.tool_calls_made = 0
        self._system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Conversation state
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the conversation, keeping configuration and tools."""
        self.history = []
        self.tool_calls_made = 0

    def _emit(self, role: str, text: str) -> None:
        if self.on_event and text:
            try:
                self.on_event(role, text)
            except Exception:
                # Display must never break the run.
                pass

    def _ensure_system_prompt(self) -> None:
        if not self.history or self.history[0].role != "system":
            self.history.insert(
                0,
                Message(
                    role="system",
                    content=self._build_system_prompt(),
                ),
            )

    def _build_system_prompt(self) -> str:
        root = self.config.working_dir or os.getcwd()
        tools = self.registry.names
        context = [
            "",
            "Environment:",
            f"- Workspace root: {root}",
            f"- Available tools: {', '.join(tools) if tools else '(none)'}",
        ]
        if not tools:
            context.append(
                "- You have no tools. Answer from the conversation and say so "
                "if the task needs tools."
            )
        return self._system_prompt + "\n".join(context)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    async def _execute_tool(self, call: ToolCall) -> str:
        """Run one tool call and return the text to send back to the model.

        Failures are returned as text rather than raised: the model needs to
        see *why* a call failed so it can recover on the next iteration.
        """
        try:
            tool = self.registry.get(call.name)
        except ToolNotFoundError as exc:
            return f"ERROR: {exc}"

        try:
            result = await tool.execute(**call.arguments)
        except TypeError as exc:
            # Most often an unknown or missing argument from the model.
            return (
                f"ERROR: invalid arguments for tool {call.name!r}: {exc}\n"
                f"Expected arguments matching this schema: "
                f"{tool.parameters.get('properties', {})}"
            )
        except AgentuxError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            return f"ERROR: tool {call.name!r} raised {type(exc).__name__}: {exc}"

        text = "" if result is None else str(result)
        if not text:
            text = "(no output)"
        if len(text) > 100_000:
            text = text[:100_000] + "\n... [tool output truncated]"
        return text

    def _format_tool_call(self, call: ToolCall) -> str:
        """Render a tool call compactly for the terminal."""
        import json

        try:
            args = json.dumps(call.arguments)
        except (TypeError, ValueError):
            args = str(call.arguments)
        if len(args) > 300:
            args = args[:300] + "..."
        return f"{call.name}({args})"

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------
    async def run(self, prompt: str, interactive: bool = False) -> str:
        """Run one user request to completion and return the final answer."""
        self._ensure_system_prompt()
        self.history.append(Message(role="user", content=prompt))
        self._emit("user", prompt)

        tools = self.registry.definitions()
        if tools and not self.provider.supports_tool_calling():
            raise UnsupportedToolCallingError(
                f"Provider {type(self.provider).__name__} cannot use structured "
                "tool calls, but this session has tools enabled. Agentux will "
                "not fake tool calls by asking the model to print JSON, so this "
                "cannot continue. Use a provider with native tool-calling "
                "support, or re-run with --no-tools."
            )

        for step in range(self.config.max_steps):
            message = await self.provider.chat(self.history, tools or None)

            if message.tool_calls:
                # Preserve the assistant turn verbatim: providers reject a
                # conversation whose tool calls are not answered in order.
                self.history.append(message)
                self.tool_calls_made += len(message.tool_calls)

                for call in message.tool_calls:
                    self._emit("tool_call", self._format_tool_call(call))
                    output = await self._execute_tool(call)
                    self._emit("tool_result", output)
                    self.history.append(
                        Message(
                            role="tool",
                            content=output,
                            tool_call_id=call.id,
                        )
                    )
                continue

            self.history.append(message)
            answer = message.content or ""
            self._emit("assistant", answer)
            return answer

        raise MaxStepsError(
            f"Stopped after {self.config.max_steps} steps without a final "
            "answer. Raise AGENTUX_MAX_STEPS if the task genuinely needs more, "
            "or try breaking the task into smaller pieces."
        )

    __all__ = ["Agent", "SYSTEM_PROMPT"]
