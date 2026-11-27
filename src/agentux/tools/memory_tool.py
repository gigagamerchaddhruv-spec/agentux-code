"""Memory tools: the model's own read and write access to the store.

``memory_save`` is how the model records something it learned. ``memory_search``
is how it looks for something it might already know. Both are deliberately
plain: the model hands over text, the store does the ranking, and the result
comes back as text the model can read and act on.
"""

from __future__ import annotations

from typing import Any, Dict

from ..errors import ToolError
from ..memory import (
    SCOPE_GLOBAL,
    SCOPE_PROJECT,
    VALID_SCOPES,
    MemoryStore,
)
from .base import BaseTool


class MemorySaveTool(BaseTool):
    """Write one thing the model learned into the store."""

    def __init__(self, store: MemoryStore, project: str = "") -> None:
        self.store = store
        self.project = project

    @property
    def name(self) -> str:
        return "memory_save"

    @property
    def description(self) -> str:
        return (
            "Save something you learned so you can recall it later. Use it for "
            "anything non-obvious: a gotcha, a fact about this project, a "
            "convention, a command that works, a URL, a correction the user "
            "gave you. One fact per call. Be specific enough that the fact is "
            "useful months from now."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "body": {
                    "type": "string",
                    "description": "The fact to remember, in one or two sentences.",
                },
                "scope": {
                    "type": "string",
                    "enum": list(VALID_SCOPES),
                    "description": (
                        "'project' (default) keeps it visible only in this "
                        "workspace. 'global' makes it visible everywhere — use "
                        "that only for lessons that are genuinely general."
                    ),
                },
                "tags": {
                    "type": "string",
                    "description": "Optional comma-separated keywords, e.g. 'go,channels'.",
                },
            },
            "required": ["body"],
        }

    async def execute(
        self,
        body: str,
        scope: str = SCOPE_PROJECT,
        tags: str = "",
        **kwargs: Any,
    ) -> str:
        if body is None or not str(body).strip():
            raise ToolError("'body' is required and must be non-empty.")
        if scope not in VALID_SCOPES:
            raise ToolError(
                f"Scope must be one of {', '.join(VALID_SCOPES)}, got {scope!r}."
            )
        if scope == SCOPE_GLOBAL:
            project = None
        else:
            project = self.project or None

        memory = self.store.save(
            str(body), scope=scope, project=project, source="model", tags=str(tags)
        )
        preview = memory.body if len(memory.body) <= 80 else memory.body[:80] + "..."
        return f"Saved memory {memory.id} ({memory.scope}): {preview}"


class MemorySearchTool(BaseTool):
    """Look for something the model might already know."""

    def __init__(self, store: MemoryStore, project: str = "") -> None:
        self.store = store
        self.project = project

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search what you have learned before. Use this before answering "
            "something you might already know, and whenever a question mentions "
            "a project, a command, or a technology you have seen before."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords describing what you are looking for. A few "
                        "specific words work better than a whole sentence."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum memories to return. Defaults to 5.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, limit: int = 5, **kwargs: Any) -> str:
        if not query or not str(query).strip():
            return "No memories matched."
        try:
            count = max(1, min(int(limit or 5), 20))
        except (TypeError, ValueError):
            count = 5

        hits = self.store.search(str(query), project=self.project or None, limit=count)
        if not hits:
            return "No memories matched."

        lines = [f"{len(hits)} remembered:"]
        for index, (memory, _score) in enumerate(hits, 1):
            scope = memory.scope if memory.scope == SCOPE_GLOBAL else "project"
            lines.append(f"{index}. [{scope}] {memory.body}")
        return "\n".join(lines)


__all__ = ["MemorySaveTool", "MemorySearchTool"]
