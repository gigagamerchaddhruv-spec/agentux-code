"""Tool interface.

A tool is anything the agent can run on the model's behalf. The model only
ever sees a tool's name, description, and JSON Schema -- it addresses tools by
name, and the registry decides what actually runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..models import ToolDefinition


class BaseTool(ABC):
    """Abstract base class for all Agentux tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name the model uses to call this tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """A description of what the tool does, used for the LLM prompt."""

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema defining the tool's input parameters."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Run the tool and return its result as text.

        Raise ``ToolError`` to report failure; the agent turns the message into
        tool output so the model can recover instead of aborting the run.
        """

    def to_definition(self) -> ToolDefinition:
        """Describe this tool in the shape providers expect."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


__all__ = ["BaseTool"]
