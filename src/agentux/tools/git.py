"""Git inspection.

Read-only by construction. Knowing what changed and what is dirty is how an
agent avoids overwriting work in progress -- both the user's and its own.

Every subcommand here is a query. Anything that rewrites history or moves
commits is the shell tool's job, where it passes through the safety prompt.
"""

from __future__ import annotations

import os
import subprocess

from ..errors import ToolError
from .base import BaseTool

#: Subcommands this tool will run.
ACTIONS = {"status", "diff", "log", "show", "branch", "blame", "stash_list", "commit"}
READ_ONLY = ACTIONS

DEFAULT_TIMEOUT = 20
MAX_OUTPUT = 40_000


class GitTool(BaseTool):
    """Inspect the repository's state without changing anything."""

    def __init__(self, root: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.root = os.path.realpath(root or os.getcwd())
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return (
            "Inspect the git repository: current status and uncommitted "
            "changes, diffs (staged or working tree), recent commit history, "
            "the contents of a commit, branches, or line authorship. Read-only "
            "-- it cannot stage, commit, or rewrite anything. Check status "
            "before editing so you do not overwrite work already in progress."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "diff", "log", "show", "branch", "blame", "stash_list", "commit"],
                    "description": (
                        "status: branch and modified/untracked files. "
                        "diff: changes (add file= for staged). "
                        "log: recent commits. show: one commit's changes. "
                        "branch: list branches. blame: who wrote each line. "
                        "stash_list: saved stashes. "
                        "commit: stage changes and create a git commit with 'message'."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": "For 'commit', the commit message to record.",
                },
                "stage_all": {
                    "type": "boolean",
                    "description": "For 'commit', whether to run git add -A first. Defaults to true.",
                },
                "path": {
                    "type": "string",
                    "description": "Limit the diff or blame to this path.",
                },
                "staged": {
                    "type": "boolean",
                    "description": "For 'diff', show staged changes only. Defaults to false.",
                },
                "revision": {
                    "type": "string",
                    "description": "For 'show' and 'log', a revision (HEAD, HEAD~2, a sha).",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of context lines in a diff. Defaults to 3.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of commits for 'log'. Defaults to 15.",
                },
            },
            "required": ["action"],
        }

    def _cwd(self) -> str:
        if not os.path.isdir(os.path.join(self.root, ".git")):
            # Worktrees and submodules keep the real git dir elsewhere.
            probe = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.root, capture_output=True, text=True, timeout=self.timeout,
            )
            if probe.returncode != 0 or probe.stdout.strip() != "true":
                raise ToolError(
                    f"{self.root} is not a git repository. The git tool only "
                    "works inside one; use the shell tool for raw git commands."
                )
        return self.root

    def _run(self, args: List[str]) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", self._cwd()] + args,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            raise ToolError("git is not available on this system.") from None
        except subprocess.TimeoutExpired:
            raise ToolError(f"git {' '.join(args[:2])} timed out after {self.timeout}s.") from None

        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise ToolError(f"git {' '.join(args[:2])} failed: {message or 'no output'}")

        output = result.stdout.strip()
        if not output:
            return "(no output)"
        if len(output) > MAX_OUTPUT:
            head = output[: MAX_OUTPUT // 2]
            tail = output[-MAX_OUTPUT // 2 :]
            dropped = len(output) - MAX_OUTPUT
            return f"{head}\n\n... [{dropped} characters omitted] ...\n\n{tail}"
        return output

    def _build(self, action: str, **kwargs: Any) -> List[str]:
        if action not in READ_ONLY:
            raise ToolError(
                f"Unsupported git action {action!r}. This tool is read-only and "
                f"supports: {', '.join(sorted(READ_ONLY))}."
            )

        path = kwargs.get("path")
        staged = bool(kwargs.get("staged"))
        revision = kwargs.get("revision")
        context = max(0, min(int(kwargs.get("lines") or 3), 20))
        limit = max(1, min(int(kwargs.get("limit") or 15), 100))

        if action == "status":
            return ["status", "--short", "--branch"]
        if action == "diff":
            args = ["diff", f"-U{context}"]
            if staged:
                args.append("--cached")
            if revision:
                args.append(revision)
            if path:
                args += ["--", path]
            return args
        if action == "log":
            args = ["log", f"-{limit}", "--no-color"]
            if path:
                args += ["--", path]
            return args
        if action == "show":
            return ["show", "--no-color", f"-U{context}", revision or "HEAD"]
        if action == "branch":
            return ["branch", "-vv", "--all"]
        if action == "blame":
            if not path:
                raise ToolError("'blame' needs a 'path' to inspect.")
            return ["blame", "-w", "--", path]
        return ["stash", "list"]

    async def execute(self, action: str, **kwargs: Any) -> str:
        if not action:
            raise ToolError("The git tool needs an 'action'.")

        if action == "commit":
            message = kwargs.get("message")
            if not message or not str(message).strip():
                raise ToolError("'commit' action requires a non-empty 'message'.")
            stage_all = kwargs.get("stage_all", True)
            if stage_all:
                self._run(["add", "-A"])
            return self._run(["commit", "-m", str(message).strip()])

        return self._run(self._build(action, **kwargs))


__all__ = ["GitTool"]
