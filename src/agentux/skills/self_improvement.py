"""Self-improvement skill for Agentux.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseSkill


class SelfImprovementSkill(BaseSkill):
    """Enables the agent to improve and extend its own capabilities.

    This skill allows Agentux to:
    - Learn from interactions and experiences
    - Create new skills based on patterns observed
    - Modify existing behaviors and capabilities
    - Extend its own functionality safely
    """

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent, config)
        self.improvement_history: List[Dict[str, Any]] = []
        self.learned_patterns: Dict[str, Any] = {}
        self.custom_skills_dir = Path.home() / ".agentux" / "custom_skills"
        self.custom_skills_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the self-improvement skill."""
        # Load any previously learned patterns
        await self._load_learned_patterns()

    async def execute(self, operation: str, **kwargs) -> Any:
        """Execute a self-improvement operation.

        Args:
            operation: The improvement operation to perform
            **kwargs: Additional arguments for the operation

        Returns:
            The result of the improvement operation
        """
        operations = {
            "learn_from_interaction": self._learn_from_interaction,
            "create_skill_from_pattern": self._create_skill_from_pattern,
            "suggest_improvements": self._suggest_improvements,
            "apply_learned_pattern": self._apply_learned_pattern,
            "get_improvement_history": self._get_improvement_history
        }

        if operation not in operations:
            raise ValueError(f"Unknown improvement operation: {operation}")

        return await operations[operation](**kwargs)

    async def _learn_from_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from an interaction or experience.

        Args:
            interaction_data: Data about the interaction to learn from

        Returns:
            Record of what was learned
        """
        interaction_id = str(uuid.uuid4())

        # Extract patterns from the interaction
        patterns = {
            "task_type": interaction_data.get("task_type", "unknown"),
            "tools_used": interaction_data.get("tools_used", []),
            "success_indicators": interaction_data.get("success", False),
            "approach_used": interaction_data.get("approach", ""),
            "lessons_learned": interaction_data.get("lessons", [])
        }

        # Store the learned pattern
        self.learned_patterns[interaction_id] = {
            "timestamp": interaction_data.get("timestamp", ""),
            "patterns": patterns,
            "interaction_data": interaction_data
        }

        # Record in improvement history
        improvement_record = {
            "id": interaction_id,
            "type": "learning",
            "description": f"Learned from {interaction_data.get('task_type', 'an interaction')}",
            "timestamp": interaction_data.get("timestamp", ""),
            "patterns_identified": list(patterns.keys())
        }

        self.improvement_history.append(improvement_record)
        await self._save_improvement_history()

        return {
            "status": "learned",
            "interaction_id": interaction_id,
            "patterns_identified": len(patterns)
        }

    async def _create_skill_from_pattern(self, pattern_name: str,
                                       skill_description: str,
                                       example_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new skill based on a learned pattern.

        Args:
            pattern_name: Name/identifier of the pattern to base the skill on
            skill_description: Description of what the skill should do
            example_usage: Example of how the skill would be used

        Returns:
            Information about the created skill
        """
        skill_id = str(uuid.uuid4())
        skill_name = f"learned_{skill_id[:8]}"

        # Create the skill file
        skill_content = f'''"""Auto-generated skill: {skill_description}

This skill was automatically created by Agentux's self-improvement system
based on learned patterns and user interactions.
"""

from __future__ import annotations

from typing import Any, Dict
from .base import BaseSkill


class {skill_name.replace('_', '').title()}Skill(BaseSkill):
    """{skill_description}"""

    def __init__(self, agent, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent, config or {{}})

    async def initialize(self) -> None:
        """Initialize the skill."""
        # Initialization code would go here based on the learned pattern
        pass

    async def execute(self, operation: str = "run", **kwargs) -> Any:
        """Execute the skill's main functionality.

        Based on learned pattern: {pattern_name}
        Example usage: {example_usage}
        """
        # Implementation would go here based on the learned pattern
        return {{"status": "executed", "skill": "{skill_name}", "operation": operation}}

    def get_info(self) -> Dict[str, Any]:
        """Get information about this skill."""
        return {{
            "name": self.__class__.__name__,
            "description": "{skill_description}",
            "config": self.config
        }}
'''

        # Save the skill file
        skill_file = self.custom_skills_dir / f"{skill_name}.py"
        with open(skill_file, "w") as f:
            f.write(skill_content)

        # Record the improvement
        improvement_record = {
            "id": str(uuid.uuid4()),
            "type": "skill_creation",
            "description": f"Created skill '{skill_name}' from pattern '{pattern_name}'",
            "skill_name": skill_name,
            "skill_file": str(skill_file),
            "timestamp": str(Path().stat().st_mtime)  # Simplified timestamp
        }

        self.improvement_history.append(improvement_record)
        await self._save_improvement_history()

        return {
            "status": "skill_created",
            "skill_name": skill_name,
            "skill_file": str(skill_file),
            "can_be_loaded": True
        }

    async def _suggest_improvements(self, current_capabilities: List[str]) -> Dict[str, Any]:
        """Suggest improvements based on learned patterns and usage.

        Args:
            current_capabilities: List of current agent capabilities

        Returns:
            Suggested improvements and potential new skills
        """
        suggestions = []

        # Analyze learned patterns for improvement opportunities
        if len(self.learned_patterns) > 5:
            suggestions.append({
                "type": "pattern_based_skill",
                "description": "Create a skill that encapsulates frequently used patterns",
                "priority": "medium",
                "estimated_effort": "low"
            })

        # Suggest based on common tool usage patterns
        if "filesystem" in current_capabilities and "shell" in current_capabilities:
            suggestions.append({
                "type": "workflow_automation",
                "description": "Create a skill for automating common file+shell workflows",
                "priority": "high",
                "estimated_effort": "medium"
            })

        # Suggest meta-improvements
        suggestions.append({
            "type": "reasoning_enhancement",
            "description": "Enhance reasoning capabilities with more sophisticated frameworks",
            "priority": "medium",
            "estimated_effort": "medium"
        })

        return {
            "current_learned_patterns": len(self.learned_patterns),
            "improvement_history_count": len(self.improvement_history),
            "suggestions": suggestions,
            "next_steps": [
                "Review learned patterns for skill creation opportunities",
                "Identify common workflows that could be automated",
                "Consider which reasoning frameworks would be most beneficial"
            ]
        }

    async def _apply_learned_pattern(self, pattern_id: str,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a previously learned pattern to a new context.

        Args:
            pattern_id: ID of the learned pattern to apply
            context: Context in which to apply the pattern

        Returns:
            Result of applying the pattern
        """
        if pattern_id not in self.learned_patterns:
            return {"status": "error", "message": f"Pattern {pattern_id} not found"}

        pattern_data = self.learned_patterns[pattern_id]
        # In a full implementation, this would adapt the pattern to the new context

        return {
            "status": "pattern_applied",
            "pattern_id": pattern_id,
            "adapted_to_context": bool(context),
            "pattern_data": pattern_data
        }

    async def _get_improvement_history(self) -> Dict[str, Any]:
        """Get the history of improvements and learning.

        Returns:
            Summary of improvement activities
        """
        # Count by type
        type_counts = {}
        for record in self.improvement_history:
            record_type = record.get("type", "unknown")
            type_counts[record_type] = type_counts.get(record_type, 0) + 1

        return {
            "total_improvements": len(self.improvement_history),
            "total_learned_patterns": len(self.learned_patterns),
            "improvement_breakdown": type_counts,
            "recent_improvements": self.improvement_history[-5:] if self.improvement_history else [],
            "custom_skills_available": len(list(self.custom_skills_dir.glob("*.py"))) if self.custom_skills_dir.exists() else 0
        }

    async def _save_improvement_history(self) -> None:
        """Save improvement history to disk."""
        try:
            history_file = self.custom_skills_dir / "improvement_history.json"
            import json
            with open(history_file, "w") as f:
                json.dump({
                    "improvement_history": self.improvement_history,
                    "learned_patterns": {k: v for k, v in self.learned_patterns.items() if k != "_meta"},
                    "last_updated": str(Path().stat().st_mtime)
                }, f, indent=2)
        except Exception:
            # Fail silently to avoid breaking the agent
            pass

    async def _load_learned_patterns(self) -> None:
        """Load previously learned patterns from disk."""
        try:
            history_file = self.custom_skills_dir / "improvement_history.json"
            if history_file.exists():
                import json
                with open(history_file) as f:
                    data = json.load(f)
                    self.improvement_history = data.get("improvement_history", [])
                    self.learned_patterns = data.get("learned_patterns", {})
        except Exception:
            # Fail silently - start with empty history
            self.improvement_history = []
            self.learned_patterns = {}