"""Marketplace integration skill for Agentux.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from .base import BaseSkill
from ..marketplace import Marketplace, MarketplaceItemType


class MarketplaceIntegrationSkill(BaseSkill):
    """Integrates with the Agentux marketplace to dynamically load skills and MCP servers.

    This skill allows Agentux to:
    - Browse and search the marketplace for available skills
    - Install and enable skills from the marketplace
    - Load installed skills into the agent's skill system
    - Manage the lifecycle of marketplace-based skills
    """

    def __init__(self, agent: Agent, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent, config)
        self.marketplace = Marketplace()
        self.loaded_marketplace_skills: Dict[str, Any] = {}  # Tracks skills loaded from marketplace

    async def initialize(self) -> None:
        """Initialize the marketplace integration skill."""
        # Ensure marketplace has default items
        self.marketplace.populate_default_items()

    async def execute(self, operation: str, **kwargs) -> Any:
        """Execute a marketplace integration operation.

        Args:
            operation: The marketplace operation to perform
            **kwargs: Additional arguments for the operation

        Returns:
            The result of the marketplace operation
        """
        operations = {
            "browse_marketplace": self._browse_marketplace,
            "search_marketplace": self._search_marketplace,
            "install_skill": self._install_skill,
            "load_installed_skills": self._load_installed_skills,
            "get_marketplace_status": self._get_marketplace_status,
            "use_marketplace_skill": self._use_marketplace_skill
        }

        if operation not in operations:
            raise ValueError(f"Unknown marketplace operation: {operation}")

        return await operations[operation](**kwargs)

    async def _browse_marketplace(self, item_type: Optional[str] = None,
                                limit: int = 20) -> Dict[str, Any]:
        """Browse the marketplace for available items.

        Args:
            item_type: Filter by item type ("skill" or "mcp_server"), or None for all
            limit: Maximum number of items to return

        Returns:
            Marketplace browsing results
        """
        marketplace_type = None
        if item_type:
            try:
                marketplace_type = MarketplaceItemType(item_type.lower())
            except ValueError:
                pass  # Ignore invalid types

        if marketplace_type:
            items = self.marketplace.get_items_by_type(marketplace_type)
        else:
            items = self.marketplace.items

        # Apply limit
        items = items[:limit]

        return {
            "total_available": len(self.marketplace.items),
            "filtered_count": len(items),
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "description": item.description,
                    "type": item.type.value,
                    "version": item.version,
                    "installed": item.installed,
                    "enabled": item.enabled,
                    "tags": item.tags
                }
                for item in items
            ]
        }

    async def _search_marketplace(self, query: str,
                                item_type: Optional[str] = None) -> Dict[str, Any]:
        """Search the marketplace for items matching a query.

        Args:
            query: Search query string
            item_type: Filter by item type, or None for all types

        Returns:
            Marketplace search results
        """
        results = self.marketplace.search_items(query)

        # Filter by type if specified
        if item_type:
            try:
                marketplace_type = MarketplaceItemType(item_type.lower())
                results = [item for item in results if item.type == marketplace_type]
            except ValueError:
                pass  # Ignore invalid types

        return {
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "id": item.id,
                    "name": item.name,
                    "description": item.description,
                    "type": item.type.value,
                    "version": item.version,
                    "installed": item.installed,
                    "enabled": item.enabled,
                    "tags": item.tags
                }
                for item in results
            ]
        }

    async def _install_skill(self, skill_id: str) -> Dict[str, Any]:
        """Install a skill from the marketplace.

        Args:
            skill_id: ID of the skill to install

        Returns:
            Installation result
        """
        item = self.marketplace.get_item(skill_id)
        if not item:
            return {"status": "error", "message": f"Skill with ID {skill_id} not found"}

        if item.type != MarketplaceItemType.SKILL:
            return {"status": "error", "message": f"Item {skill_id} is not a skill"}

        # Install in marketplace
        success = self.marketplace.install_item(skill_id)

        if success:
            # Try to load the skill (this would require the skill to actually exist)
            try:
                loaded_skill = await self._attempt_load_marketplace_skill(item)
                return {
                    "status": "installed_and_loaded",
                    "skill_name": item.name,
                    "skill_id": skill_id,
                    "loaded": loaded_skill is not None
                }
            except Exception as e:
                return {
                    "status": "installed_load_failed",
                    "skill_name": item.name,
                    "skill_id": skill_id,
                    "load_error": str(e)
                }
        else:
            return {"status": "error", "message": f"Failed to install skill {skill_id}"}

    async def _load_installed_skills(self) -> Dict[str, Any]:
        """Load all installed skills from the marketplace.

        Returns:
            Result of loading installed skills
        """
        installed_items = self.marketplace.get_installed_items()
        skill_items = [item for item in installed_items if item.type == MarketplaceItemType.SKILL]

        loaded_skills = []
        failed_skills = []

        for item in skill_items:
            try:
                loaded_skill = await self._attempt_load_marketplace_skill(item)
                if loaded_skill:
                    loaded_skills.append({
                        "skill_name": item.name,
                        "skill_id": item.id,
                        "status": "loaded"
                    })
                    self.loaded_marketplace_skills[item.id] = loaded_skill
                else:
                    failed_skills.append({
                        "skill_name": item.name,
                        "skill_id": item.id,
                        "reason": "Skill class not found or failed to load"
                    })
            except Exception as e:
                failed_skills.append({
                    "skill_name": item.name,
                    "skill_id": item.id,
                    "reason": str(e)
                })

        return {
            "total_installed_skills": len(skill_items),
            "successfully_loaded": len(loaded_skills),
            "failed_to_load": len(failed_skills),
            "loaded_skills": loaded_skills,
            "failed_skills": failed_skills
        }

    async def _attempt_load_marketplace_skill(self, item: Any) -> Optional[Any]:
        """Attempt to load a skill from a marketplace item.

        Args:
            item: MarketplaceItem representing the skill

        Returns:
            Loaded skill instance, or None if loading failed
        """
        # This is a placeholder implementation
        # In a real implementation, this would:
        # 1. Parse the entry_point to determine how to load the skill
        # 2. Dynamically import the skill module
        # 3. Instantiate the skill class
        # 4. Initialize and return the skill

        # For now, we'll simulate loading for known default skills
        known_skills = {
            "frontend-design": "FrontendDesignSkill",
            "dataviz": "DataVizSkill",
            "artifact-diagramming": "ArtifactDiagrammingSkill",
            "update-config": "UpdateConfigSkill",
            "keybindings-help": "KeybindingsHelpSkill"
        }

        if item.name in known_skills:
            # Simulate successful loading
            return {
                "name": item.name,
                "type": "marketplace_skill",
                "entry_point": item.entry_point,
                "status": "simulated_loaded"
            }

        # For unknown skills, return None to indicate they need to be created
        return None

    async def _get_marketplace_status(self) -> Dict[str, Any]:
        """Get the current status of the marketplace integration.

        Returns:
            Marketplace status information
        """
        installed_items = self.marketplace.get_installed_items()
        enabled_items = self.marketplace.get_enabled_items()

        return {
            "total_marketplace_items": len(self.marketplace.items),
            "installed_items": len(installed_items),
            "enabled_items": len(enabled_items),
            "available_skills": len(self.marketplace.get_items_by_type(MarketplaceItemType.SKILL)),
            "available_mcp_servers": len(self.marketplace.get_items_by_type(MarketplaceItemType.MCP_SERVER)),
            "installed_skills": len([i for i in installed_items if i.type == MarketplaceItemType.SKILL]),
            "enabled_skills": len([i for i in enabled_items if i.type == MarketplaceItemType.SKILL]),
            "loaded_marketplace_skills": len(self.loaded_marketplace_skills),
            "recently_updated": sorted(
                [item for item in self.marketplace.items if item.updated_at],
                key=lambda x: x.updated_at,
                reverse=True
            )[:3] if self.marketplace.items else []
        }

    async def _use_marketplace_skill(self, skill_id: str, operation: str = "run",
                                  **kwargs) -> Dict[str, Any]:
        """Use a skill that was loaded from the marketplace.

        Args:
            skill_id: ID of the skill to use
            operation: Operation to perform on the skill
            **kwargs: Additional arguments for the operation

        Returns:
            Result of using the skill
        """
        # Check if it's a loaded marketplace skill
        if skill_id in self.loaded_marketplace_skills:
            skill_data = self.loaded_marketplace_skills[skill_id]
            return {
                "status": "used_marketplace_skill",
                "skill_id": skill_id,
                "skill_name": skill_data.get("name", "unknown"),
                "operation": operation,
                "result": f"Simulated execution of {skill_data.get('name', 'marketplace skill')}",
                **kwargs
            }

        # Check if it's installed and try to load it
        item = self.marketplace.get_item(skill_id)
        if item and item.installed:
            try:
                loaded_skill = await self._attempt_load_marketplace_skill(item)
                if loaded_skill:
                    self.loaded_marketplace_skills[skill_id] = loaded_skill
                    return {
                        "status": "loaded_and_used",
                        "skill_id": skill_id,
                        "skill_name": item.name,
                        "operation": operation,
                        "result": f"Loaded and executed {item.name}",
                        **kwargs
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to load skill {skill_id}: {str(e)}"
                }

        return {
            "status": "error",
            "message": f"Skill {skill_id} not found or not installed"
        }