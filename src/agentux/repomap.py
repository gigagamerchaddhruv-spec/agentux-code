"""Token-efficient AST and symbol-based Repository Map generator for Agentux.

Inspired by repomap algorithms in advanced AI pair programming tools. It scans
source files (Python via native AST, plus multi-language extractors for Go, TS/JS,
Rust), extracts definitions (classes, structs, functions, methods, interfaces),
and produces a compact outline that fits into 200-600 tokens.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class SymbolDef:
    name: str
    kind: str  # "class", "function", "method", "struct", "interface"
    line: int
    signature: str = ""


@dataclass
class FileSymbols:
    rel_path: str
    symbols: List[SymbolDef] = field(default_factory=list)


# Standard ignore patterns
IGNORE_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".idea", ".vscode", "dist", "build", ".pytest_cache", ".mypy_cache",
    ".tox", "target", "vendor", "bin", "obj", ".agentux"
}

IGNORE_EXTS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".lock", ".sum",
    ".log", ".tmp", ".gz", ".tar", ".zip", ".wasm", ".map"
}

CODE_EXTS = {
    ".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".java", ".rb", ".php", ".sh"
}


class RepoMap:
    """Extracts and formats a dense, token-aware map of the codebase."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)

    def scan_files(self, limit: int = 200) -> List[str]:
        """Collect source files within the workspace respecting ignores."""
        found: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
            for f in sorted(filenames):
                if f.startswith("."):
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext in CODE_EXTS and ext not in IGNORE_EXTS:
                    rel = os.path.relpath(os.path.join(dirpath, f), self.root)
                    found.append(rel)
                    if len(found) >= limit:
                        return found
        return found

    def extract_symbols(self, rel_path: str) -> FileSymbols:
        """Parse definitions from a single file."""
        abs_path = os.path.join(self.root, rel_path)
        symbols: List[SymbolDef] = []
        ext = os.path.splitext(rel_path)[1].lower()

        if not os.path.isfile(abs_path):
            return FileSymbols(rel_path=rel_path, symbols=[])

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return FileSymbols(rel_path=rel_path, symbols=[])

        if ext == ".py":
            symbols = self._extract_python(content)
        else:
            symbols = self._extract_generic(content, ext)

        return FileSymbols(rel_path=rel_path, symbols=symbols)

    def _extract_python(self, content: str) -> List[SymbolDef]:
        """Extract classes and top-level / method functions using AST."""
        symbols: List[SymbolDef] = []
        try:
            tree = ast.parse(content)
        except Exception:
            return self._extract_generic(content, ".py")

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbols.append(SymbolDef(name=node.name, kind="class", line=node.lineno))
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args_list = [a.arg for a in child.args.args if a.arg != "self"]
                        sig = f"({', '.join(args_list)})"
                        symbols.append(
                            SymbolDef(
                                name=f"{node.name}.{child.name}",
                                kind="method",
                                line=child.lineno,
                                signature=sig,
                            )
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args_list = [a.arg for a in node.args.args]
                sig = f"({', '.join(args_list)})"
                symbols.append(
                    SymbolDef(
                        name=node.name,
                        kind="function",
                        line=node.lineno,
                        signature=sig,
                    )
                )
        return symbols

    def _extract_generic(self, content: str, ext: str) -> List[SymbolDef]:
        """Regex-based symbol extractor for Go, TS/JS, Rust, etc."""
        symbols: List[SymbolDef] = []
        lines = content.split("\n")

        # Patterns for functions, classes, structs
        patterns = [
            # Go: type Foo struct / func (r *R) Method / func Func
            (re.compile(r"^type\s+([A-Za-z0-9_]+)\s+(?:struct|interface)"), "type"),
            (re.compile(r"^func\s+(?:\([^)]+\)\s+)?([A-Za-z0-9_]+)\s*\("), "func"),
            # JS/TS: class Foo / function bar / export function bar / const bar = () =>
            (re.compile(r"^(?:export\s+)?class\s+([A-Za-z0-9_]+)"), "class"),
            (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)"), "function"),
            (re.compile(r"^(?:export\s+)?interface\s+([A-Za-z0-9_]+)"), "interface"),
            # Rust: struct Foo / fn bar / impl Foo
            (re.compile(r"^(?:pub\s+)?(?:struct|enum)\s+([A-Za-z0-9_]+)"), "struct"),
            (re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)"), "fn"),
        ]

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            for pat, kind in patterns:
                m = pat.search(stripped)
                if m:
                    symbols.append(SymbolDef(name=m.group(1), kind=kind, line=idx))
                    break
        return symbols

    def generate_map(self, max_tokens: int = 600, query: Optional[str] = None) -> str:
        """Render a compact repository outline within token budget."""
        files = self.scan_files()
        if not files:
            return "(empty or non-code repository)"

        query_lower = query.lower() if query else None
        lines: List[str] = []
        token_estimate = 0

        for rel in files:
            fs = self.extract_symbols(rel)
            if not fs.symbols and not query_lower:
                continue

            filtered = fs.symbols
            if query_lower:
                filtered = [s for s in fs.symbols if query_lower in s.name.lower() or query_lower in rel.lower()]
                if not filtered and query_lower not in rel.lower():
                    continue

            file_header = f"{rel}:"
            lines.append(file_header)
            token_estimate += len(file_header) // 4 + 1

            for s in filtered:
                sig = f" {s.signature}" if s.signature else ""
                sym_line = f"  {s.kind} {s.name}{sig} (L{s.line})"
                lines.append(sym_line)
                token_estimate += len(sym_line) // 4 + 1

                if token_estimate >= max_tokens:
                    lines.append("  ... (truncated to fit token budget)")
                    return "\n".join(lines)

        return "\n".join(lines) if lines else "(no symbols matched query)"
