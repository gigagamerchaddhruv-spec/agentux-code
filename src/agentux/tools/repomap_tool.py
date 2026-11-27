"""Repomap tool for Agentux."""

from __future__ import annotations

from typing import Any, Dict
from ..models import ToolDefinition
from ..repomap import RepoMap
from .base import BaseTool


class RepomapTool(BaseTool):
    """Tool allowing the agent to inspect the codebase symbol outline."""

    def __init__(self, root: str) -> None:
        self.repomap = RepoMap(root)

    @property
    def name(self) -> str:
        return "repomap"

    @property
    def description(self) -> str:
        return (
            "Generate a compact, token-efficient repository symbol map. "
            "Shows all classes, structs, functions, methods, and line numbers "
            "across the codebase without reading entire files. "
            "Optionally filter by symbol query or path keyword."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional symbol name, keyword, or path to filter the map.",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum token budget for the map (default 600).",
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query")
        max_tokens = int(kwargs.get("max_tokens", 600))
        return self.repomap.generate_map(max_tokens=max_tokens, query=query)
