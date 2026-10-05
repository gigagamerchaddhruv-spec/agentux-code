"""Project awareness.

Gives the model a compact map of a repository -- what kind of project it is,
which files matter, what the test layout looks like -- without reading the
whole tree. Inspection is targeted and depth-limited on purpose.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..errors import ToolError
from .base import BaseTool

#: Files that identify a project and its dependencies.
MANIFESTS = {
    "pyproject.toml": "Python (pyproject)",
    "setup.py": "Python (setup.py)",
    "requirements.txt": "Python requirements",
    "package.json": "Node/TypeScript",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java (Maven)",
    "build.gradle": "Java/Kotlin (Gradle)",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
    "CMakeLists.txt": "C/C++ (CMake)",
}

#: Other files worth surfacing during orientation.
SIGNALS = [
    "README.md", "README.rst", "README.txt", "Makefile", "Justfile",
    "Dockerfile", "docker-compose.yml", ".gitignore", "LICENSE", "CHANGELOG.md",
]

#: Directories that usually hold the interesting source.
SOURCE_DIRS = ["src", "lib", "app", "pkg", "internal", "cmd", "tests", "test", "spec"]

#: Directories that are noise in essentially every repository.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
    ".idea", ".vscode", "target", ".next", ".gradle", ".cache", "site-packages",
}

MAX_DEPTH = 3
MAX_ENTRIES_PER_DIR = 40


class ProjectTool(BaseTool):
    """Summarize the layout and conventions of the current project."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = os.path.realpath(root or os.getcwd())

    @property
    def name(self) -> str:
        return "project"

    @property
    def description(self) -> str:
        return (
            "Inspect the current repository: identify the language and build "
            "system, list the directory structure, and show entry points, test "
            "locations, and tracked files. Use this once at the start of a task "
            "to orient yourself, then use the filesystem and shell tools for "
            "details. Do not read the whole repository -- this returns a map."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["overview", "tree", "tracked"],
                    "description": (
                        "overview: project type, key files, source and test dirs. "
                        "tree: directory structure to a limited depth. "
                        "tracked: list files tracked by git."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory to inspect. Defaults to the project root.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Depth limit for the 'tree' action. Defaults to 3.",
                },
            },
            "required": ["action"],
        }

    def _base(self, path: Optional[str]) -> str:
        if not path:
            return self.root
        candidate = path if os.path.isabs(path) else os.path.join(self.root, path)
        resolved = os.path.realpath(os.path.expanduser(candidate))
        if not os.path.isdir(resolved):
            raise ToolError(f"Not a directory: {path}")
        return resolved

    def _overview(self, base: str) -> str:
        lines: List[str] = [f"Project root: {self.root}"]

        try:
            entries = sorted(os.listdir(base))
        except PermissionError as exc:
            raise ToolError(f"Permission denied reading {base}: {exc}") from exc

        kinds = [label for name, label in MANIFESTS.items() if name in entries]
        if kinds:
            lines.append(f"Type: {', '.join(sorted(set(kinds)))}")
        else:
            lines.append("Type: unknown (no recognised manifest file at the root)")

        found_signals = [name for name in SIGNALS if name in entries]
        if found_signals:
            lines.append(f"Key files: {', '.join(found_signals)}")

        sources = [
            d for d in SOURCE_DIRS
            if os.path.isdir(os.path.join(base, d)) and d not in SKIP_DIRS
        ]
        if sources:
            lines.append(f"Source/test directories: {', '.join(sources)}")

        script_dirs = []
        package_json = os.path.join(base, "package.json")
        if os.path.isfile(package_json):
            try:
                import json

                with open(package_json, encoding="utf-8") as handle:
                    data = json.load(handle)
                script_dirs = sorted((data.get("scripts") or {}).keys())
            except Exception:
                script_dirs = []
            if script_dirs:
                lines.append(f"npm scripts: {', '.join(script_dirs)}")

        lines.append("")
        lines.append("Top level:")
        for entry in entries[:MAX_ENTRIES_PER_DIR]:
            if entry in SKIP_DIRS:
                continue
            full = os.path.join(base, entry)
            suffix = "/" if os.path.isdir(full) else ""
            lines.append(f"  {entry}{suffix}")
        return "\n".join(lines)

    def _tree(self, base: str, max_depth: int) -> str:
        depth_limit = max(1, min(int(max_depth or MAX_DEPTH), 6))
        lines: List[str] = []
        total = 0

        def walk(directory: str, prefix: str, depth: int) -> None:
            nonlocal total
            if depth > depth_limit or total > 500:
                return
            try:
                entries = sorted(os.listdir(directory))
            except PermissionError:
                lines.append(f"{prefix}[permission denied]")
                return
            visible = [e for e in entries if e not in SKIP_DIRS]
            hidden = len(entries) - len(visible)
            for index, entry in enumerate(visible):
                if total > 500:
                    lines.append(f"{prefix}...")
                    return
                total += 1
                full = os.path.join(directory, entry)
                last = index == len(visible) - 1 and not hidden
                branch = "└── " if last else "├── "
                if os.path.isdir(full):
                    lines.append(f"{prefix}{branch}{entry}/")
                    walk(full, prefix + ("    " if last else "│   "), depth + 1)
                else:
                    lines.append(f"{prefix}{branch}{entry}")
            if hidden:
                lines.append(f"{prefix}└── ... ({hidden} hidden or generated entries omitted)")

        lines.append(f"{os.path.relpath(base, self.root) or '.'}/ (depth {depth_limit})")
        walk(base, "", 1)
        return "\n".join(lines)

    def _tracked(self, base: str) -> str:
        import subprocess

        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=base, capture_output=True, text=True, timeout=15,
            )
        except FileNotFoundError:
            raise ToolError("git is not available on this system.") from None
        except subprocess.TimeoutExpired:
            raise ToolError("git ls-files timed out.") from None

        if result.returncode != 0:
            raise ToolError(
                f"This directory is not a git repository (or git failed): "
                f"{result.stderr.strip() or 'no output'}"
            )
        files = [f for f in result.stdout.splitlines() if f.strip()]
        if not files:
            return "No tracked files (the repository is empty)."
        listing = "\n".join(files[:300])
        if len(files) > 300:
            listing += f"\n... {len(files) - 300} more tracked files"
        return f"{len(files)} tracked file(s):\n{listing}"

    async def execute(
        self,
        action: str,
        path: Optional[str] = None,
        max_depth: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        if action not in ("overview", "tree", "tracked"):
            raise ToolError(
                f"Unsupported action {action!r}. Use 'overview', 'tree', or 'tracked'."
            )
        base = self._base(path)
        if action == "overview":
            return self._overview(base)
        if action == "tree":
            return self._tree(base, max_depth or MAX_DEPTH)
        return self._tracked(base)


__all__ = ["ProjectTool"]
