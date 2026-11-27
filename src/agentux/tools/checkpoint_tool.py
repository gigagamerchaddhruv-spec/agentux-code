"""Checkpoint tool for Agentux."""

from __future__ import annotations

from typing import Any, Dict
from ..checkpoint import CheckpointManager
from ..models import ToolDefinition
from .base import BaseTool


class CheckpointTool(BaseTool):
    """Tool allowing the agent to create checkpoints, view diffs, and inspect state."""

    def __init__(self, manager: CheckpointManager) -> None:
        self.manager = manager

    @property
    def name(self) -> str:
        return "checkpoint"

    @property
    def description(self) -> str:
        return (
            "Manage project checkpoints and inspect session diffs. "
            "Actions: 'create' (create new snapshot), 'diff' (view current unified diff), "
            "'list' (list recent checkpoints)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "diff", "list"],
                    "description": "The action to perform.",
                },
                "description": {
                    "type": "string",
                    "description": "Description for action='create'.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "diff")
        if action == "create":
            desc = kwargs.get("description", "Manual agent snapshot")
            cp = self.manager.create_checkpoint(desc)
            return f"Created checkpoint '{cp.id}' at {cp.formatted_time}: {desc}"
        elif action == "diff":
            return self.manager.get_diff()
        elif action == "list":
            cps = self.manager.list_checkpoints()
            if not cps:
                return "No checkpoints recorded in this session."
            lines = [f"- [{c.id}] ({c.formatted_time}) {c.description} ({len(c.files)} files)" for c in cps]
            return "Session Checkpoints:\n" + "\n".join(lines)
        return f"Unknown action: {action}"
