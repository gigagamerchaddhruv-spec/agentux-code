"""Content search across the workspace.

Regex-based, the way an agent actually needs to find things: a symbol's
definition and its call sites, a string literal, an error message. Every result
carries enough context to act on without a second read.

Results are capped and files are skipped by extension and directory, because an
unbounded search is how an agent fills its own context window with noise.
"""

from __future__ import annotations

import os
import re

from ..errors import ToolError
from .base import BaseTool

#: Directories that are never worth grepping.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
    ".idea", ".vscode", "target", ".next", ".gradle", ".cache", "site-packages",
    ".eggs", "*.egg-info",
}

#: Binary and generated formats with nothing useful in them as text.
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svgz",
    ".pdf", ".zip", ".gz", ".bz2", ".xz", ".tar", ".7z", ".rar",
    ".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a", ".class", ".jar",
    ".pyc", ".pyo", ".wasm", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".mp4", ".mov", ".avi", ".wav", ".flac", ".ogg",
    ".db", ".sqlite", ".sqlite3", ".lock",
}

#: Files larger than this are reported but not searched: a 4 MB minified bundle
#: matches everything and tells the model nothing.
MAX_FILE_BYTES = 1_000_000

#: Caps that keep a search result readable and the context window survivable.
MAX_MATCHES = 200
MAX_FILES = 2_000
MAX_LINE_CHARS = 400
MAX_FILE_LIST = 100


class SearchTool(BaseTool):
    """Find text in files across the workspace."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = os.path.realpath(root or os.getcwd())

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Search file contents with a regular expression. Returns matching "
            "lines with file:line references and surrounding context. Use this "
            "to find where something is defined or used before reading whole "
            "files. Also supports listing file paths by glob pattern when "
            "'content_pattern' is omitted."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content_pattern": {
                    "type": "string",
                    "description": (
                        "Regular expression to search for. Omit to list files "
                        "whose paths match 'file_pattern' instead."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search under. Defaults to the project root.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": (
                        "Glob restricting which files are searched, e.g. '*.py' "
                        "or 'src/**/*.ts'. Omit to search all text files."
                    ),
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context to show around each match. Defaults to 1.",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Case-insensitive match. Defaults to false.",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum matches to return. Defaults to 200.",
                },
            },
            "required": [],
        }

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------
    def _base(self, path: Optional[str]) -> str:
        if not path:
            return self.root
        candidate = path if os.path.isabs(path) else os.path.join(self.root, path)
        resolved = os.path.realpath(os.path.expanduser(candidate))
        if not resolved.startswith(self.root + os.sep) and resolved != self.root:
            raise ToolError(f"Path {path!r} is outside the workspace root {self.root}.")
        return resolved

    def _walk(self, base: str) -> Iterator[str]:
        """Yield searchable files under ``base``, pruning noise directories."""
        if os.path.isfile(base):
            yield base
            return

        checked = 0
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in SKIP_DIRS and not d.endswith(".egg-info")
            )
            for filename in sorted(filenames):
                if os.path.splitext(filename)[1].lower() in SKIP_SUFFIXES:
                    continue
                full = os.path.join(dirpath, filename)
                try:
                    if os.path.getsize(full) > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                yield full
                checked += 1
                if checked >= MAX_FILES:
                    return

    def _matches_glob(self, path: str, pattern: Optional[str]) -> bool:
        if not pattern:
            return True
        try:
            from fnmatch import fnmatch

            if fnmatch(path, pattern):
                return True
            # A bare '*.py' should also match nested paths.
            return fnmatch(os.path.basename(path), pattern)
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _list_files(self, base: str, file_pattern: Optional[str]) -> str:
        matches = [
            os.path.relpath(p, self.root)
            for p in self._walk(base)
            if self._matches_glob(p, file_pattern)
        ]
        if not matches:
            scope = os.path.relpath(base, self.root) or "."
            return f"No files matching {file_pattern!r} under {scope}."

        shown = matches[:MAX_FILE_LIST]
        lines = "\n".join(shown)
        footer = ""
        if len(matches) > MAX_FILE_LIST:
            footer = f"\n... {len(matches) - MAX_FILE_LIST} more (narrow 'file_pattern')"
        return f"{len(matches)} file(s):\n{lines}{footer}"

    def _grep(
        self,
        base: str,
        pattern: str,
        file_pattern: Optional[str],
        context_lines: int,
        ignore_case: bool,
        max_matches: int,
    ) -> str:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise ToolError(
                f"Invalid regular expression {pattern!r}: {exc}"
            ) from exc

        span = max(0, min(int(context_lines or 0), 5))
        results: List[str] = []
        matched_files: List[str] = []
        truncated_at: Optional[str] = None
        total = 0

        for full in self._walk(base):
            if not self._matches_glob(full, file_pattern):
                continue
            try:
                with open(full, encoding="utf-8", errors="replace") as handle:
                    lines = handle.readlines()
            except (OSError, PermissionError):
                continue

            rel = os.path.relpath(full, self.root)
            hits = [i for i, line in enumerate(lines) if regex.search(line)]
            if not hits:
                continue
            matched_files.append(rel)
            total += len(hits)

            # Collapse a run of nearby hits into one block instead of repeating
            # the same context lines for every match.
            shown_lines: set = set()
            for index in hits:
                if len(results) >= max_matches:
                    truncated_at = rel
                    break
                start, end = max(0, index - span), min(len(lines), index + span + 1)
                if any(j in shown_lines for j in range(start, end)):
                    continue
                shown_lines.update(range(start, end))
                if start > 0:
                    results.append("    ...")
                for j in range(start, end):
                    marker = ">" if j == index else " "
                    text = lines[j].rstrip("\n")
                    if len(text) > MAX_LINE_CHARS:
                        text = text[:MAX_LINE_CHARS] + "..."
                    results.append(f"{marker} {rel}:{j + 1}: {text}")
                if end < len(lines):
                    results.append("    ...")
            if truncated_at:
                break

        if not results:
            scope = os.path.relpath(base, self.root) or "."
            where = f" under {scope}" if scope != "." else ""
            suffix = f" matching {file_pattern!r}" if file_pattern else ""
            return f"No matches for {pattern!r}{where}{suffix}."

        header = f"{total} match(es) for {pattern!r} in {len(matched_files)} file(s)"
        footer = ""
        if truncated_at:
            footer = (
                f"\n\n... stopped at {max_matches} matches (in {truncated_at}). "
                "Narrow the pattern or the file_pattern to see the rest."
            )
        elif total > max_matches:
            footer = "\n\n... output capped."
        return f"{header}\n\n" + "\n".join(results) + footer

    async def execute(
        self,
        content_pattern: Optional[str] = None,
        path: Optional[str] = None,
        file_pattern: Optional[str] = None,
        context_lines: int = 1,
        ignore_case: bool = False,
        max_matches: int = MAX_MATCHES,
        **kwargs: Any,
    ) -> str:
        base = self._base(path)

        if not content_pattern:
            return self._list_files(base, file_pattern)

        return self._grep(
            base=base,
            pattern=content_pattern,
            file_pattern=file_pattern,
            context_lines=context_lines,
            ignore_case=bool(ignore_case),
            max_matches=max(1, min(int(max_matches or MAX_MATCHES), 500)),
        )


__all__ = ["SearchTool"]
