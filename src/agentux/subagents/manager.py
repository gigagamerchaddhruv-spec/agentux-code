"""Autonomous subagents and delegation system for Agentux.

Allows hierarchical agent execution: the primary orchestrator can delegate
subtasks to specialized agents (researcher, coder, reviewer, architect) with
scoped tools, independent context windows, and dedicated system prompts.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from ..models import ToolDefinition
from ..tools.base import BaseTool


RESEARCHER_PROMPT = """\
You are an expert Codebase Researcher subagent.
Your goal is to thoroughly investigate the codebase to answer the user's inquiry or find code locations.
Rules:
- You are strictly read-only.
- Use 'search', 'filesystem', 'repomap', and 'outline_symbols' to find relevant code.
- Report exact file paths, line numbers, function/class names, and concise summaries.
- Keep your final report direct and actionable for the parent agent.\
"""

CODER_PROMPT = """\
You are an expert Implementation Coder subagent.
Your goal is to implement the requested code changes with minimal, surgical edits.
Rules:
- Read files before editing.
- Use 'filesystem' operation 'edit' to replace exact target strings.
- Match existing idioms, naming conventions, and style.
- State clearly what files were modified.\
"""

REVIEWER_PROMPT = """\
You are a senior Code Reviewer and Quality Assurance subagent.
Your goal is to inspect current git diffs and code modifications for bugs, regressions, security flaws, and test coverage.
Rules:
- Inspect git status and git diff.
- Run tests via 'shell' when appropriate to verify changes.
- Provide a clear, prioritized list of findings (Critical, Warning, Improvement) or verify that changes are ready to commit.\
"""

ARCHITECT_PROMPT = """\
You are a Software Architect subagent.
Your goal is to inspect high-level project structure, dependencies, and design patterns to create a clear plan.
Rules:
- Use 'project' and 'repomap' to understand layout.
- Decompose the request into ordered, testable steps in 'todo'.
- Deliver an architectural recommendation with file boundaries and interfaces.\
"""

SPECIALIST_CONFIGS: Dict[str, Dict[str, Any]] = {
    "researcher": {
        "prompt": RESEARCHER_PROMPT,
        "tools": ["filesystem", "search", "outline_symbols", "repomap", "project"],
        "read_only": True,
    },
    "coder": {
        "prompt": CODER_PROMPT,
        "tools": ["filesystem", "git", "shell", "search"],
        "read_only": False,
    },
    "reviewer": {
        "prompt": REVIEWER_PROMPT,
        "tools": ["git", "filesystem", "shell", "search"],
        "read_only": True,
    },
    "architect": {
        "prompt": ARCHITECT_PROMPT,
        "tools": ["project", "repomap", "search", "todo"],
        "read_only": True,
    },
}


class SubagentManager:
    """Orchestrates spawning, running, and reporting of specialist subagents."""

    def __init__(self, parent_agent: Any) -> None:
        self.parent_agent = parent_agent
        self.history: List[Dict[str, Any]] = []

    async def run_subagent(self, role: str, task: str) -> str:
        """Run an isolated subagent and return its final answer."""
        from ..agent import Agent
        from ..tools.registry import ToolRegistry
        from ..tools.filesystem import FilesystemTool

        role_lower = role.lower().strip()
        spec = SPECIALIST_CONFIGS.get(role_lower)
        if not spec:
            available = ", ".join(SPECIALIST_CONFIGS.keys())
            return f"Unknown subagent role '{role}'. Available specialist roles: {available}."

        # Build scoped registry for subagent
        scoped_registry = ToolRegistry()
        for name in spec["tools"]:
            try:
                tool = self.parent_agent.registry.get(name)
                # If researcher or reviewer needs read-only filesystem
                if spec["read_only"] and name == "filesystem":
                    tool = FilesystemTool(
                        root=self.parent_agent.config.working_dir,
                        read_only=True
                    )
                scoped_registry.register(tool)
            except Exception:
                pass

        # Create isolated subagent
        subagent = Agent(
            config=self.parent_agent.config,
            provider=self.parent_agent.provider,
            registry=scoped_registry,
            system_prompt=spec["prompt"],
        )

        # Notify parent TUI/listeners of subagent activity
        if hasattr(self.parent_agent, "_emit"):
            self.parent_agent._emit("narration", f"🤖 Spawning '{role_lower}' subagent for: {task}")

        try:
            result = await subagent.run(task, interactive=False)
            self.history.append({
                "role": role_lower,
                "task": task,
                "steps": subagent.last_steps,
                "tool_calls": subagent.tool_calls_made,
                "result": result,
            })
            return f"[{role_lower.upper()} SUBAGENT REPORT]\n{result}"
        except Exception as exc:
            return f"[{role_lower.upper()} SUBAGENT ERROR]: {exc}"


class DelegateTool(BaseTool):
    """Tool allowing the agent to delegate subtasks to autonomous subagents."""

    def __init__(self, manager: SubagentManager) -> None:
        self.manager = manager

    @property
    def name(self) -> str:
        return "delegate"

    @property
    def description(self) -> str:
        return (
            "Delegate a subtask to an autonomous specialist subagent. "
            "Roles: 'researcher' (read-only code investigation), "
            "'coder' (targeted code changes/edits), "
            "'reviewer' (diff inspection, QA, test validation), "
            "'architect' (high-level system design and planning)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["researcher", "coder", "reviewer", "architect"],
                    "description": "The specialist role to delegate to.",
                },
                "task": {
                    "type": "string",
                    "description": "Detailed prompt describing what the subagent should investigate or do.",
                },
            },
            "required": ["role", "task"],
        }

    async def execute(self, **kwargs: Any) -> str:
        role = kwargs.get("role", "researcher")
        task = kwargs.get("task", "")
        if not task:
            return "ERROR: 'task' argument is required for delegation."
        return await self.manager.run_subagent(role, task)
