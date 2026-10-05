"""Tool registry.

Holds the tools the agent may call and exposes them to providers as tool
definitions. The agent depends on this, not on individual tools.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..errors import ToolNotFoundError
from ..models import ToolDefinition
from .base import BaseTool


class ToolRegistry:
    """Registry for managing and looking up tools."""

    def __init__(self, tools: Optional[List[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Register a tool, replacing any existing tool with the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Return a registered tool, or raise :class:`ToolNotFoundError`."""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "(none registered)"
            raise ToolNotFoundError(
                f"No tool named {name!r}. Available tools: {available}"
            )
        return tool

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    @property
    def names(self) -> List[str]:
        return sorted(self._tools)

    def definitions(self) -> List[ToolDefinition]:
        """Generate the tool schemas sent to the model."""
        return [tool.to_definition() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


__all__ = ["ToolRegistry"]
