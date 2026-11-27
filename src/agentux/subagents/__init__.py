"""Autonomous subagent system package for Agentux."""

from .manager import (
    ARCHITECT_PROMPT,
    CODER_PROMPT,
    DelegateTool,
    RESEARCHER_PROMPT,
    REVIEWER_PROMPT,
    SPECIALIST_CONFIGS,
    SubagentManager,
)

__all__ = [
    "SubagentManager",
    "DelegateTool",
    "SPECIALIST_CONFIGS",
    "RESEARCHER_PROMPT",
    "CODER_PROMPT",
    "REVIEWER_PROMPT",
    "ARCHITECT_PROMPT",
]
