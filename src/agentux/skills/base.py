"""Base class for Agentux skills.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..agent import Agent
from ..config import Config
from ..context import ContextManager


class BaseSkill(ABC):
    """Base class for all Agentux skills."""

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None):
        self.agent = agent
        self.config = config or {}
        self.name = self.__class__.__name__
        self.description = getattr(self.__class__, '__doc__', "").strip() or "No description"

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the skill. Called when the skill is loaded."""
        pass

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the skill's main functionality."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get information about this skill."""
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config
        }


class SkillManager:
    """Manages loading and executing skills."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.skills: Dict[str, BaseSkill] = {}
        self._skill_classes: Dict[str, type[BaseSkill]] = {}

    def register_skill_class(self, name: str, skill_class: type[BaseSkill]) -> None:
        """Register a skill class for later instantiation."""
        self._skill_classes[name] = skill_class

    def load_skill(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseSkill:
        """Load and initialize a skill by name."""
        if name in self.skills:
            return self.skills[name]

        if name not in self._skill_classes:
            raise ValueError(f"Skill class '{name}' not registered")

        skill_instance = self._skill_classes[name](self.agent, config or {})
        self.skills[name] = skill_instance
        # Initialize the skill
        import asyncio
        asyncio.create_task(skill_instance.initialize())
        return skill_instance

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """Get a loaded skill by name."""
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """List all loaded skill names."""
        return list(self.skills.keys())

    def list_available_skills(self) -> List[str]:
        """List all registered skill names."""
        return list(self._skill_classes.keys())