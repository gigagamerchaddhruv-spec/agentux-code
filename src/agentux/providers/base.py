"""Provider interface.

The agent loop depends only on :class:`BaseProvider`. It never imports a
concrete provider and never contains HTTP logic, so adding a provider is a
matter of implementing this interface -- the agent loop does not change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Awaitable, Callable, List, Optional

from ..config import Config
from ..models import Message, ToolDefinition

#: Receives text deltas as they arrive. Async so providers can await the
#: consumer (a terminal write) without blocking the read loop.
TextCallback = Callable[[str], Awaitable[None]]


class BaseProvider(ABC):
    """A model backend Agentux can drive."""

    def __init__(self, config: Config) -> None:
        self.config = config
        #: Token usage reported by the most recent response, or ``None``.
        #: Agents read this to show real numbers instead of estimates.
        self.last_usage: Optional[Dict[str, int]] = None

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> Message:
        """Return the model's next message.

        If the model wants tools, the returned message carries ``tool_calls``.
        If it is done, ``tool_calls`` is empty and ``content`` is the answer.
        """
        raise NotImplementedError

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Whether this provider can use native structured tool calls.

        Providers that return ``False`` are rejected when tools are requested:
        Agentux will not fake tool calls by asking for JSON in plain text.
        """
        raise NotImplementedError

    def get_model_name(self) -> str:
        return self.config.model

    async def list_models(self) -> List[str]:
        """Return the model names this provider offers.

        Used by ``agentux --list-models`` so a local user can discover what is
        available instead of guessing.
        """
        return []

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive.

        The default implementation falls back to a single chunk from
        :meth:`chat`, so providers only need to override this to get streaming.
        """
        message = await self.chat(messages, tools)
        if message.content:
            yield message.content

    async def chat_streaming(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        on_text: Optional[TextCallback] = None,
    ) -> Message:
        """Chat, streaming text deltas to ``on_text`` as they arrive.

        Returns a complete :class:`Message`, so a step that turns out to be a
        tool call costs the same single request as a step that is not. That is
        the whole point: streaming a step and *then* discovering it was a tool
        call means paying for the request twice.
        """
        message = await self.chat(messages, tools)
        if on_text and message.content:
            await on_text(message.content)
        return message


__all__ = ["BaseProvider", "TextCallback"]
