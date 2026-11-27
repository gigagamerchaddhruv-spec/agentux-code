"""High-speed symbol outline and code skeleton extraction tool.

Allows the AI model to locate classes, functions, and types in large files
with precise line numbers, minimizing token consumption.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..errors import ToolError
from .base import BaseTool


class OutlineSymbolsTool(BaseTool):
    """Extract symbol outlines (classes, functions, types) from workspace source code."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = os.path.realpath(root or os.getcwd())

    @property
    def name(self) -> str:
        return "outline_symbols"

    @property
    def description(self) -> str:
        return (
            "Extracts a high-level symbol outline (classes, functions, methods, types, "
            "interfaces) with line numbers from source files (Python, Go, JS/TS, Rust). "
            "Far faster and more token-efficient than reading full files."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the source code file (relative to workspace).",
                },
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> str:
        path = path.strip()
        if not path:
            raise ToolError("File path cannot be empty.")

        full_path = path if os.path.isabs(path) else os.path.join(self.root, path)
        full_path = os.path.realpath(full_path)

        if not full_path.startswith(self.root):
            raise ToolError(f"Path escapes workspace directory: {path}")
        if not os.path.isfile(full_path):
            raise ToolError(f"File not found: {path}")

        ext = os.path.splitext(full_path)[1].lower()
        patterns = self._get_patterns_for_ext(ext)

        symbols: List[Tuple[int, str, str]] = []
        total_lines = 0

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                for idx, line in enumerate(f, 1):
                    total_lines = idx
                    stripped = line.strip()
                    if not stripped or stripped.startswith(("//", "#", "/*", "*")):
                        continue

                    for kind, regex in patterns:
                        if regex.search(stripped):
                            symbols.append((idx, kind, stripped))
                            break
        except Exception as exc:
            raise ToolError(f"Failed to read file {path}: {exc}")

        rel_path = os.path.relpath(full_path, self.root)
        header = f"Outline of {rel_path} ({total_lines} lines, {len(symbols)} symbols found):\n"

        if not symbols:
            return header + "\n(no top-level classes or functions detected)"

        out_lines = [header]
        for line_num, kind, raw in symbols:
            out_lines.append(f"L{line_num:<4} [{kind:<7}] {raw}")

        return "\n".join(out_lines)

    def _get_patterns_for_ext(self, ext: str) -> List[Tuple[str, re.Pattern]]:
        if ext == ".py":
            return [
                ("class", re.compile(r"^class\s+[A-Za-z0-9_]+")),
                ("async", re.compile(r"^async\s+def\s+[A-Za-z0-9_]+\s*\(")),
                ("def", re.compile(r"^def\s+[A-Za-z0-9_]+\s*\(")),
            ]
        elif ext == ".go":
            return [
                ("method", re.compile(r"^func\s+\([^)]+\)\s+[A-Za-z0-9_]+\s*\(")),
                ("func", re.compile(r"^func\s+[A-Za-z0-9_]+\s*\(")),
                ("struct", re.compile(r"^type\s+[A-Za-z0-9_]+\s+struct\b")),
                ("interface", re.compile(r"^type\s+[A-Za-z0-9_]+\s+interface\b")),
                ("type", re.compile(r"^type\s+[A-Za-z0-9_]+\s+")),
            ]
        elif ext in (".ts", ".tsx", ".js", ".jsx"):
            return [
                ("class", re.compile(r"^(?:export\s+)?(?:default\s+)?class\s+[A-Za-z0-9_]+")),
                ("func", re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+[A-Za-z0-9_]+")),
                ("type", re.compile(r"^(?:export\s+)?(?:interface|type)\s+[A-Za-z0-9_]+")),
                ("const", re.compile(r"^(?:export\s+)?(?:const|let|var)\s+[A-Za-z0-9_]+\s*=\s*(?:async\s*)?\(")),
            ]
        elif ext == ".rs":
            return [
                ("fn", re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+[A-Za-z0-9_]+")),
                ("struct", re.compile(r"^(?:pub\s+)?struct\s+[A-Za-z0-9_]+")),
                ("enum", re.compile(r"^(?:pub\s+)?enum\s+[A-Za-z0-9_]+")),
                ("impl", re.compile(r"^impl(?:\s*<[^>]+>)?\s+[A-Za-z0-9_]+")),
                ("trait", re.compile(r"^(?:pub\s+)?trait\s+[A-Za-z0-9_]+")),
            ]
        else:
            return [
                ("symbol", re.compile(r"^(?:class|struct|function|def|func)\s+[A-Za-z0-9_]+")),
            ]


__all__ = ["OutlineSymbolsTool"]
