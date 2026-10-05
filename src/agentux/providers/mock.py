"""Deterministic in-memory provider for tests and offline development.

It never performs network I/O and never uses randomness, so a test that
scripts a sequence of responses gets the same loop every time.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..models import Message, ToolCall, ToolDefinition
from .base import BaseProvider


class MockProvider(BaseProvider):
    """Replays a scripted list of responses.

    Each call to :meth:`chat` pops the next scripted message. Once the script
    is exhausted the last message repeats, which keeps a runaway loop from
    raising inside the provider (the agent's step ceiling catches it instead).
    """

    def __init__(self, responses: Sequence[Message], config=None) -> None:
        if config is None:
            from ..config import Config

            config = Config(api_key="mock-key")
        super().__init__(config)
        self.responses = list(responses)
        self.calls: List[List[Message]] = []
        self.tools_seen: List[Optional[List[ToolDefinition]]] = []

    def queue(self, *responses: Message) -> "MockProvider":
        """Append more scripted responses and return self, for chaining."""
        self.responses.extend(responses)
        return self

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> Message:
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        if not self.responses:
            return Message(role="assistant", content="")
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return "mock-model"


def tool_call(call_id: str, name: str, **arguments) -> Message:
    """Build an assistant message that requests a single tool call."""
    return Message(
        role="assistant",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def text(content: str) -> Message:
    """Build a plain assistant reply."""
    return Message(role="assistant", content=content)


__all__ = ["MockProvider", "tool_call", "text"]
