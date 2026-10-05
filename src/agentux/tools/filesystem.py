"""Filesystem tool.

Every path is resolved against a workspace root and rejected if it escapes
it, so a confused or adversarial model cannot wander into ``/etc`` or
``~/.ssh``. Reads are size-capped and writes report a real diff so the agent
does not blindly clobber files.
"""

from __future__ import annotations

import difflib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import ToolError
from .base import BaseTool

#: Reads larger than this are truncated rather than dumped into the context.
MAX_READ_BYTES = 400_000
#: Directory listings are capped so the model gets a summary, not a firehose.
MAX_LIST_ENTRIES = 500
#: Directories that are noise in almost every project.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
    ".idea", ".vscode", "target", ".next", ".gradle",
}

_OPERATIONS = ["ls", "read", "write", "edit", "mkdir", "rm", "mv", "glob"]


class FilesystemTool(BaseTool):
    """Read and modify files inside the workspace."""

    def __init__(self, root: Optional[str] = None, read_only: bool = False) -> None:
        self.root = Path(root or os.getcwd()).resolve()
        self.read_only = read_only

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return (
            "Read and modify files in the workspace. Prefer 'edit' over 'write' "
            "when changing part of an existing file, since edit only replaces "
            "the exact text you target and leaves the rest untouched. "
            "Paths may be absolute or relative to the workspace root."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": _OPERATIONS,
                    "description": "The filesystem operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory, absolute or relative to the root.",
                },
                "content": {
                    "type": "string",
                    "description": "Full new content. Required for 'write'.",
                },
                "old_string": {
                    "type": "string",
                    "description": (
                        "Exact existing text to replace. Required for 'edit'. "
                        "Must appear exactly once."
                    ),
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text for 'edit'. Use an empty string to delete.",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination path. Required for 'mv'.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern such as '**/*.py'. Used by 'glob'.",
                },
            },
            "required": ["operation", "path"],
        }

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------
    def _resolve(self, path: str) -> Path:
        """Resolve ``path`` inside the workspace, or raise.

        Uses ``Path.resolve`` (which collapses ``..`` symlinks and relative
        segments) *before* the containment check, so neither ``../../etc`` nor
        a symlink pointing outside the root can escape.
        """
        if not path or not str(path).strip():
            raise ToolError("A non-empty 'path' is required.")
        candidate = Path(str(path).strip()).expanduser()
        resolved = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ToolError(
                f"Path {str(path)!r} is outside the workspace root {str(self.root)!r}. "
                "Agentux only touches files inside the workspace."
            ) from None
        return resolved

    def _relative(self, path: Path) -> str:
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return str(path)
        return str(rel) if str(rel) != "." else "."

    def _require_writable(self) -> None:
        if self.read_only:
            raise ToolError(
                "This filesystem tool is read-only. Files cannot be created, "
                "modified, moved, or deleted."
            )

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def _ls(self, target: Path, show_all: bool) -> str:
        if not target.exists():
            raise ToolError(f"Path does not exist: {self._relative(target)}")
        if not target.is_dir():
            raise ToolError(f"Path is not a directory: {self._relative(target)}")
        try:
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError as exc:
            raise ToolError(
                f"Permission denied listing {self._relative(target)}: {exc}"
            ) from exc

        lines: List[str] = []
        hidden = 0
        for entry in entries:
            if entry.name.startswith(".") and not show_all:
                hidden += 1
                continue
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            elif entry.is_symlink():
                lines.append(f"{entry.name} -> (symlink)")
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"{entry.name} ({size} bytes)")
            if len(lines) >= MAX_LIST_ENTRIES:
                lines.append(f"... truncated at {MAX_LIST_ENTRIES} entries")
                break

        if hidden:
            lines.append(f"({hidden} hidden entries omitted)")
        header = f"{self._relative(target)}/ - {len(entries)} entries"
        return header + "\n" + ("\n".join(lines) if lines else "(empty)")

    def _read(self, target: Path) -> str:
        if not target.exists():
            raise ToolError(f"File does not exist: {self._relative(target)}")
        if target.is_dir():
            raise ToolError(
                f"{self._relative(target)} is a directory, not a file. "
                "Use the 'ls' operation to list it."
            )
        try:
            size = target.stat().st_size
        except OSError as exc:
            raise ToolError(f"Cannot stat {self._relative(target)}: {exc}") from exc

        try:
            data = target.read_bytes()
        except PermissionError as exc:
            raise ToolError(
                f"Permission denied reading {self._relative(target)}: {exc}"
            ) from exc
        except OSError as exc:
            raise ToolError(f"Failed to read {self._relative(target)}: {exc}") from exc

        text = data.decode("utf-8", errors="replace")
        if size > MAX_READ_BYTES:
            return (
                text[:MAX_READ_BYTES]
                + f"\n\n... [truncated: file is {size} bytes, showing first "
                f"{MAX_READ_BYTES}. Use 'grep' via the shell tool to search it.]"
            )
        return text

    def _write(self, target: Path, content: str) -> str:
        existed = target.exists()
        if target.is_dir():
            raise ToolError(
                f"{self._relative(target)} is a directory, not a file."
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except PermissionError as exc:
            raise ToolError(
                f"Permission denied writing {self._relative(target)}: {exc}"
            ) from exc
        except OSError as exc:
            raise ToolError(f"Failed to write {self._relative(target)}: {exc}") from exc

        verb = "Overwrote" if existed else "Created"
        return f"{verb} {self._relative(target)} ({len(content)} bytes)."

    def _edit(self, target: Path, old: str, new: str) -> str:
        if not target.exists():
            raise ToolError(
                f"Cannot edit {self._relative(target)}: the file does not exist. "
                "Use the 'write' operation to create it."
            )
        if target.is_dir():
            raise ToolError(f"{self._relative(target)} is a directory, not a file.")

        content = self._read(target)
        if not old:
            raise ToolError(
                "'old_string' is required for the 'edit' operation and cannot be empty."
            )

        occurrences = content.count(old)
        if occurrences == 0:
            preview = old if len(old) <= 200 else old[:200] + "..."
            raise ToolError(
                f"'old_string' was not found in {self._relative(target)}. "
                f"Read the file again and copy the exact text to replace.\n"
                f"Searched for: {preview!r}"
            )
        if occurrences > 1:
            raise ToolError(
                f"'old_string' appears {occurrences} times in "
                f"{self._relative(target)}, so the edit would be ambiguous. "
                "Include more surrounding context to make it unique."
            )

        updated = content.replace(old, new, 1)
        try:
            target.write_text(updated, encoding="utf-8")
        except PermissionError as exc:
            raise ToolError(
                f"Permission denied writing {self._relative(target)}: {exc}"
            ) from exc

        diff = list(
            difflib.unified_diff(
                content.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{self._relative(target)}",
                tofile=f"b/{self._relative(target)}",
                n=2,
            )
        )
        report = "".join(diff[:200])
        if len(diff) > 200:
            report += f"\n... [diff truncated, {len(diff)} lines total]"
        return f"Edited {self._relative(target)}.\n{report}".rstrip()

    def _mkdir(self, target: Path) -> str:
        if target.exists():
            if target.is_dir():
                return f"Directory already exists: {self._relative(target)}"
            raise ToolError(
                f"Cannot create directory {self._relative(target)}: a file with "
                "that name already exists."
            )
        try:
            target.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            raise ToolError(
                f"Permission denied creating {self._relative(target)}: {exc}"
            ) from exc
        except OSError as exc:
            raise ToolError(
                f"Failed to create directory {self._relative(target)}: {exc}"
            ) from exc
        return f"Created directory {self._relative(target)}."

    def _rm(self, target: Path) -> str:
        if not target.exists() and not target.is_symlink():
            raise ToolError(f"Path does not exist: {self._relative(target)}")
        if target == self.root:
            raise ToolError("Refusing to delete the workspace root.")
        try:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        except PermissionError as exc:
            raise ToolError(
                f"Permission denied deleting {self._relative(target)}: {exc}"
            ) from exc
        return f"Deleted {self._relative(target)}."

    def _mv(self, source: Path, destination: str) -> str:
        if not source.exists():
            raise ToolError(f"Path does not exist: {self._relative(source)}")
        dest = self._resolve(destination)
        if dest == self.root:
            raise ToolError("Cannot move onto the workspace root.")
        if dest.exists():
            raise ToolError(
                f"Destination already exists: {self._relative(dest)}. "
                "Remove it first or choose another path."
            )
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
        except OSError as exc:
            raise ToolError(
                f"Failed to move {self._relative(source)} to "
                f"{self._relative(dest)}: {exc}"
            ) from exc
        return f"Moved {self._relative(source)} -> {self._relative(dest)}."

    def _glob(self, target: Path, pattern: str) -> str:
        if not target.is_dir():
            raise ToolError(
                f"{self._relative(target)} is not a directory, so 'glob' cannot "
                "search it. Use 'ls'."
            )
        if not pattern:
            raise ToolError("'pattern' is required for the 'glob' operation.")
        matches: List[str] = []
        for path in sorted(target.glob(pattern)):
            rel = self._relative(path)
            if any(part in SKIP_DIRS for part in path.relative_to(self.root).parts[:-1]):
                continue
            matches.append(rel)
            if len(matches) >= MAX_LIST_ENTRIES:
                matches.append(f"... truncated at {MAX_LIST_ENTRIES} matches")
                break
        if not matches:
            return f"No files matching {pattern!r} under {self._relative(target)}."
        return f"{len(matches)} match(es) for {pattern!r}:\n" + "\n".join(matches)

    # ------------------------------------------------------------------
    async def execute(
        self,
        operation: str,
        path: str,
        content: Optional[str] = None,
        old_string: Optional[str] = None,
        new_string: Optional[str] = None,
        destination: Optional[str] = None,
        pattern: Optional[str] = None,
        show_all: bool = False,
        **kwargs: Any,
    ) -> str:
        if operation not in _OPERATIONS:
            raise ToolError(
                f"Unsupported operation {operation!r}. Use one of: {', '.join(_OPERATIONS)}."
            )
        target = self._resolve(path)

        read_only_ops = {"ls", "read", "glob"}
        if operation not in read_only_ops:
            self._require_writable()

        if operation == "ls":
            return self._ls(target, show_all)
        if operation == "read":
            return self._read(target)
        if operation == "write":
            if content is None:
                raise ToolError("'content' is required for the 'write' operation.")
            return self._write(target, content)
        if operation == "edit":
            if old_string is None or new_string is None:
                raise ToolError(
                    "'old_string' and 'new_string' are both required for the "
                    "'edit' operation."
                )
            return self._edit(target, old_string, new_string)
        if operation == "mkdir":
            return self._mkdir(target)
        if operation == "rm":
            return self._rm(target)
        if operation == "mv":
            if not destination:
                raise ToolError("'destination' is required for the 'mv' operation.")
            return self._mv(target, destination)
        return self._glob(target, pattern or "")


__all__ = ["FilesystemTool"]
