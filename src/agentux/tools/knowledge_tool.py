"""Knowledge database tools for Agentux Solo.

The model's read/write access to its persistent knowledge store. These tools
exist so a small model does not have to remember everything: it can look up
what it learned before, save what it just discovered, and inspect the store.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..errors import ToolError
from ..knowledge import (
    KnowledgeStore,
    VERIFICATION_CONFLICTING,
    VERIFICATION_VERIFIED,
    VERIFICATION_UNVERIFIED,
)
from .base import BaseTool

_VALID_VERIFICATION = (VERIFICATION_UNVERIFIED, VERIFICATION_VERIFIED, VERIFICATION_CONFLICTING)


def _format_entry(k) -> str:
    lines = [
        f"[{k.id}] {k.title}",
        f"    Category: {k.category or '(none)'}",
        f"    Confidence: {k.confidence:.0%}  Verification: {k.verification}",
        f"    Tags: {k.tags or '(none)'}",
        f"    Content: {k.content}",
    ]
    if k.sources:
        lines.append("    Sources:")
        for s in k.sources:
            lines.append(f"      - {s.get('url', '')}")
    return "\n".join(lines)


class SearchKnowledgeTool(BaseTool):
    """Search the persistent knowledge database."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "search_knowledge"

    @property
    def description(self) -> str:
        return (
            "Search what you learned before and stored in your knowledge "
            "database. Use this BEFORE answering a factual question you might "
            "already know the answer to. Returns matching entries with their "
            "sources, confidence, and verification status."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing what you are looking for.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category to restrict the search to.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum entries to return (default 10).",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, category: str = "", limit: int = 10, **kwargs: Any) -> str:
        if not query or not str(query).strip():
            return "No query provided."
        try:
            count = max(1, min(int(limit or 10), 50))
        except (TypeError, ValueError):
            count = 10
        hits = self.store.search(str(query), category=category or None, limit=count)
        if not hits:
            return "No matching knowledge entries found."
        header = f"Found {len(hits)} matching knowledge entr(ies):"
        return header + "\n\n" + "\n\n".join(_format_entry(k) for k in hits)


class SaveKnowledgeTool(BaseTool):
    """Store newly discovered information in the knowledge database."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "save_knowledge"

    @property
    def description(self) -> str:
        return (
            "Save something you discovered to your persistent knowledge "
            "database. Use this after researching a topic so you do not have "
            "to look it up again next time. Store one fact per call. Include "
            "the source URLs so you can verify the information later."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title for this fact.",
                },
                "content": {
                    "type": "string",
                    "description": "The fact itself, in one or two sentences.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category, e.g. 'history', 'science', 'programming'.",
                },
                "tags": {
                    "type": "string",
                    "description": "Optional comma-separated keywords.",
                },
                "confidence": {
                    "type": "number",
                    "description": "How confident you are (0.0 to 1.0). Default 0.5.",
                },
                "verification": {
                    "type": "string",
                    "enum": list(_VALID_VERIFICATION),
                    "description": "'unverified' (default), 'verified', or 'conflicting'.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional list of source URLs this fact came from.",
                },
            },
            "required": ["title", "content"],
        }

    async def execute(
        self,
        title: str,
        content: str,
        category: str = "",
        tags: str = "",
        confidence: float = 0.5,
        verification: str = VERIFICATION_UNVERIFIED,
        sources: List[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        if not title or not str(title).strip():
            raise ToolError("'title' is required.")
        if not content or not str(content).strip():
            raise ToolError("'content' is required.")
        if verification not in _VALID_VERIFICATION:
            raise ToolError(
                f"verification must be one of {', '.join(_VALID_VERIFICATION)}"
            )
        entry = self.store.save(
            str(title).strip(),
            str(content).strip(),
            category=str(category).strip(),
            tags=str(tags).strip(),
            confidence=float(confidence),
            verification=verification,
            sources=sources or None,
        )
        return (
            f"Saved knowledge entry {entry.id} ({entry.verification}, "
            f"{entry.confidence:.0%} confidence): {entry.title}"
        )


class GetKnowledgeTool(BaseTool):
    """Retrieve a specific knowledge entry by ID."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "get_knowledge"

    @property
    def description(self) -> str:
        return (
            "Retrieve a specific knowledge entry by its ID. Use this to look "
            "up something you stored earlier when you only have its ID."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The knowledge entry ID.",
                },
            },
            "required": ["id"],
        }

    async def execute(self, id: str, **kwargs: Any) -> str:
        entry = self.store.get(str(id).strip())
        if entry is None:
            return f"No knowledge entry found with ID {id}."
        return _format_entry(entry)


class ListKnowledgeTool(BaseTool):
    """List stored knowledge entries, optionally filtered by category."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "list_knowledge"

    @property
    def description(self) -> str:
        return (
            "List the knowledge entries you have stored, optionally filtered "
            "by category. Use this to see what you already know."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category to filter by.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum entries to list (default 20).",
                },
            },
        }

    async def execute(self, category: str = "", limit: int = 20, **kwargs: Any) -> str:
        try:
            count = max(1, min(int(limit or 20), 100))
        except (TypeError, ValueError):
            count = 20
        entries = self.store.list_all(category=category or None, limit=count)
        if not entries:
            return "No knowledge entries stored yet."
        lines = [f"{len(entries)} knowledge entr(ies):", ""]
        for entry in entries:
            lines.append(f"- [{entry.id}] {entry.title} ({entry.category or 'uncategorized'}, {entry.verification})")
        return "\n".join(lines)


class DeleteKnowledgeTool(BaseTool):
    """Remove an incorrect or obsolete knowledge entry."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "delete_knowledge"

    @property
    def description(self) -> str:
        return (
            "Delete a knowledge entry you stored when it turns out to be "
            "wrong or obsolete. Use with care -- deletion is permanent."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The knowledge entry ID to delete.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this entry is being deleted.",
                },
            },
            "required": ["id"],
        }

    async def execute(self, id: str, reason: str = "", **kwargs: Any) -> str:
        entry = self.store.get(str(id).strip())
        if entry is None:
            return f"No knowledge entry found with ID {id}."
        self.store.delete(entry.id)
        note = f" (reason: {reason})" if reason else ""
        return f"Deleted knowledge entry {entry.id}: {entry.title}{note}"


class UpdateKnowledgeTool(BaseTool):
    """Update an existing knowledge entry with newer or more accurate info."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "update_knowledge"

    @property
    def description(self) -> str:
        return (
            "Update an existing knowledge entry when you discover newer or "
            "more accurate information. Use this instead of saving a duplicate."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The knowledge entry ID to update.",
                },
                "title": {"type": "string", "description": "New title (optional)."},
                "content": {"type": "string", "description": "New content (optional)."},
                "category": {"type": "string", "description": "New category (optional)."},
                "tags": {"type": "string", "description": "New tags (optional)."},
                "confidence": {"type": "number", "description": "New confidence 0.0-1.0 (optional)."},
                "verification": {
                    "type": "string",
                    "enum": list(_VALID_VERIFICATION),
                    "description": "New verification status (optional).",
                },
            },
            "required": ["id"],
        }

    async def execute(self, id: str, **kwargs: Any) -> str:
        entry = self.store.get(str(id).strip())
        if entry is None:
            return f"No knowledge entry found with ID {id}."
        title = kwargs.get("title") or entry.title
        content = kwargs.get("content") or entry.content
        category = kwargs.get("category") or entry.category
        tags = kwargs.get("tags") or entry.tags
        confidence = float(kwargs.get("confidence", entry.confidence))
        verification = kwargs.get("verification") or entry.verification
        if verification not in _VALID_VERIFICATION:
            raise ToolError(
                f"verification must be one of {', '.join(_VALID_VERIFICATION)}"
            )
        updated = self.store.save(
            title, content, category=category, tags=tags,
            confidence=confidence, verification=verification,
            knowledge_id=entry.id,
        )
        return f"Updated knowledge entry {updated.id}: {updated.title}"


class ExtractInformationTool(BaseTool):
    """Extract factual information from research material."""

    def __init__(self, store: Optional[KnowledgeStore] = None) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "extract_information"

    @property
    def description(self) -> str:
        return (
            "Extract the factual information from research material you have "
            "gathered. Returns a structured list of facts with their sources, "
            "ready to be saved to your knowledge database."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "material": {
                    "type": "string",
                    "description": "The research material to extract facts from.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional source URLs the material came from.",
                },
                "question": {
                    "type": "string",
                    "description": "Optional question to focus extraction on.",
                },
            },
            "required": ["material"],
        }

    async def execute(
        self,
        material: str,
        sources: List[Dict[str, Any]] = None,
        question: str = "",
        **kwargs: Any,
    ) -> str:
        material = (material or "").strip()
        if not material:
            raise ToolError("'material' is required.")
        if len(material) > 50_000:
            material = material[:50_000] + "\n... [truncated]"

        # Deterministic extraction: split into sentences and keep the ones
        # that look like factual statements. A small model benefits from a
        # structured output it can act on directly.
        import re

        sentences = re.split(r"(?<=[.!?])\s+", material)
        facts: List[Dict[str, Any]] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 15:
                continue
            if sentence.startswith(("#", "|", "*", "-", ">", "`")):
                continue
            facts.append({
                "fact": sentence,
                "sources": [s.get("url", "") for s in (sources or []) if s.get("url")],
            })

        if not facts:
            return "No factual statements could be extracted from the material."

        lines = [f"Extracted {len(facts)} fact(s):", ""]
        for i, fact in enumerate(facts, 1):
            srcs = ", ".join(fact["sources"]) or "(no source)"
            lines.append(f"{i}. {fact['fact']}")
            lines.append(f"   Source: {srcs}")
        return "\n".join(lines)


class SearchMultipleTool(BaseTool):
    """Search for a topic from multiple angles."""

    def __init__(self, web_search=None) -> None:
        self._web_search = web_search

    @property
    def name(self) -> str:
        return "search_multiple"

    @property
    def description(self) -> str:
        return (
            "Search for a topic from multiple angles at once. Use this when "
            "one query is not enough to understand a subject -- it runs "
            "several related searches and returns all results together."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries to run.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results per query (default 3).",
                },
            },
            "required": ["queries"],
        }

    async def execute(self, queries: List[str], max_results: int = 3, **kwargs: Any) -> str:
        if not queries:
            raise ToolError("'queries' is required and must be non-empty.")
        if self._web_search is None:
            raise ToolError("No web search backend is configured.")
        try:
            count = max(1, min(int(max_results or 3), 10))
        except (TypeError, ValueError):
            count = 3
        blocks: List[str] = []
        for query in queries:
            try:
                result = self._web_search(str(query).strip(), count)
            except Exception as exc:
                result = f"Search failed for '{query}': {exc}"
            blocks.append(f"### {query}\n{result}")
        return "\n\n".join(blocks)


class GetDateTimeTool(BaseTool):
    """Return the current date and time."""

    @property
    def name(self) -> str:
        return "get_current_datetime"

    @property
    def description(self) -> str:
        return (
            "Get the current date and time. Use this when the user asks about "
            "the current date, time, or 'today'."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        import datetime

        now = datetime.datetime.now(datetime.timezone.utc)
        return (
            f"Current date and time (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"ISO 8601: {now.isoformat()}"
        )


class CalculateTool(BaseTool):
    """Perform deterministic mathematical calculations."""

    @property
    def name(self) -> str:
        return "calculate"

    @property
    def description(self) -> str:
        return (
            "Perform a mathematical calculation and return the result. Use "
            "this for arithmetic instead of trying to compute it yourself."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression, e.g. '2 + 3 * 4'.",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, expression: str, **kwargs: Any) -> str:
        import ast
        import operator as op

        expression = str(expression).strip()
        if not expression:
            raise ToolError("'expression' is required.")

        _SAFE_OPS = {
            ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
            ast.Div: op.truediv, ast.Mod: op.mod, ast.Pow: op.pow,
            ast.USub: op.neg, ast.UAdd: op.pos, ast.FloorDiv: op.floordiv,
        }

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.BinOp):
                return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp):
                return _SAFE_OPS[type(node.op)](_eval(node.operand))
            raise ValueError(f"Unsupported operation: {type(node).__name__}")

        try:
            tree = ast.parse(expression, mode="eval")
            result = _eval(tree)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return f"{expression} = {result}"
        except Exception as exc:
            raise ToolError(f"Could not evaluate '{expression}': {exc}")


__all__ = [
    "SearchKnowledgeTool",
    "SaveKnowledgeTool",
    "GetKnowledgeTool",
    "ListKnowledgeTool",
    "DeleteKnowledgeTool",
    "UpdateKnowledgeTool",
    "ExtractInformationTool",
    "SearchMultipleTool",
    "GetDateTimeTool",
    "CalculateTool",
]