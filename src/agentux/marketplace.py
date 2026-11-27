"""Skill and MCP marketplace for Agentux.

Provides a browsable, installable catalogue of community-contributed
skills and MCP servers. Data lives in JSON files under
``~/.config/agentux/marketplace/`` so the catalogue works offline
and can be synced via git or any file-sharing method.

Each entry is a small JSON object:

{
    "name": "skill-name",            # slug, must be unique in its kind
    "kind": "skill" | "mcp",
    "title": "Human-readable title",
    "description": "One-paragraph summary",
    "author": "GitHub handle or name",
    "source": "https://github.com/user/repo",  # for skills: the repo; for MCPs: the repo with the server code
    "entry": {                       # how to activate it in Agentux
        "skill": {                   # for kind == "skill"
            "module": "package.submodule:function",  # dotted path to the skill function
            "args": "optional default args string"
        },
        "mcp": {                     # for kind == "mcp"
            "config": {              # minimal MCP server config to merge
                "command": "npx",
                "args": ["-y", "package"]
            }
        }
    },
    "tags": ["utility", "formatting"],  # free-form tags for search
    "version": "1.0.0",               # semver, informational
    "license": "MIT"                  # SPDX identifier
}

The catalogue itself is a JSON array under ``catalogue.json``.
A local ``installed.json`` tracks what the user has added to their
``.agentux/skills.py`` or ``.agentux/mcp.json`` files.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


MARKETPLACE_DIR = Path(os.path.expanduser("~/.config/agentux/marketplace"))
CATALOGUE_PATH = MARKETPLACE_DIR / "catalogue.json"
INSTALLED_PATH = MARKETPLACE_DIR / "installed.json"


class MarketplaceItemType(Enum):
    """Type of marketplace item."""
    SKILL = "skill"
    MCP_SERVER = "mcp"


@dataclass
class MarketplaceEntry:
    """A single skill or MCP server entry in the marketplace.

    This class is maintained for backwards compatibility.
    New code should use MarketplaceItem instead.
    """
    id: str
    name: str
    kind: str  # "skill" or "mcp"
    title: str
    description: str
    author: str
    source: str
    entry: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    license: str = "MIT"
    homepage: Optional[str] = None
    installed: bool = False
    enabled: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketplaceEntry":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MarketplaceItem:
    """A single skill or MCP server entry in the marketplace."""
    id: str
    name: str
    type: MarketplaceItemType
    title: str
    description: str
    author: str
    source: str
    entry: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    license: str = "MIT"
    homepage: Optional[str] = None
    installed: bool = False
    enabled: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert enum to string
        data['type'] = self.type.value
        # Also add kind field for backwards compatibility
        data['kind'] = self.type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketplaceItem":
        """Create from dictionary."""
        # Convert type string back to enum
        if 'type' in data and isinstance(data['type'], str):
            data['type'] = MarketplaceItemType(data['type'])
        # Also convert kind field if present (for backwards compatibility)
        if 'kind' in data and isinstance(data['kind'], str) and 'type' not in data:
            data['type'] = MarketplaceItemType(data['kind'])
        # `kind` is a legacy alias for `type`; drop it so it is not passed
        # to __init__ as a duplicate keyword argument.
        data.pop('kind', None)
        return cls(**data)


def _ensure_dir() -> None:
    """Create the marketplace directory if it does not exist."""
    MARKETPLACE_DIR.mkdir(parents=True, exist_ok=True)


def _load_catalogue_items() -> List[MarketplaceItem]:
    """Load the marketplace catalogue from disk.

    Returns an empty list if the file is missing or unreadable.
    """
    _ensure_dir()
    if not CATALOGUE_PATH.exists():
        return []
    try:
        with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            return []
        return [MarketplaceItem.from_dict(item) for item in raw]
    except (json.JSONDecodeError, OSError):
        return []


def _save_catalogue_to_disk(items: List[MarketplaceItem]) -> None:
    """Overwrite the catalogue file with the given items.

    Disk-only. The public :func:`save_catalogue` below delegates to a
    Marketplace instance, so this name is private to avoid the recursion.
    """
    _ensure_dir()
    with CATALOGUE_PATH.open("w", encoding="utf-8") as f:
        json.dump([item.to_dict() for item in items], f, indent=2, ensure_ascii=False)


def _load_installed_from_disk() -> List[str]:
    """Return the list of entry names the user has installed."""
    _ensure_dir()
    if not INSTALLED_PATH.exists():
        return []
    try:
        with INSTALLED_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return [str(name) for name in data]
    except (json.JSONDecodeError, OSError):
        return []


def _save_installed_to_disk(names: List[str]) -> None:
    """Overwrite the installed list with the given names."""
    _ensure_dir()
    with INSTALLED_PATH.open("w", encoding="utf-8") as f:
        json.dump(sorted(set(names)), f, indent=2, ensure_ascii=False)


def install_entry(name: str) -> bool:
    """Mark an entry as installed; returns True if the name was not already present."""
    installed = _load_installed_from_disk()
    if name in installed:
        return False
    installed.append(name)
    _save_installed_to_disk(installed)
    return True


def uninstall_entry(name: str) -> bool:
    """Mark an entry as uninstalled; returns True if the name was present."""
    installed = _load_installed_from_disk()
    if name not in installed:
        return False
    installed.remove(name)
    _save_installed_to_disk(installed)
    return True


def is_installed(name: str) -> bool:
    """Check whether an entry is marked as installed."""
    return name in _load_installed_from_disk()


def search_catalogue(query: str = "", kind: Optional[str] = None) -> List[MarketplaceItem]:
    """Return catalogue entries matching the query and/or kind.

    Query is a case-insensitive substring match on title, description, author, or tags.
    Kind is either "skill", "mcp", or None for both.
    """
    items = load_catalogue()
    if not query and not kind:
        return items

    q = query.lower()
    results: List[MarketplaceItem] = []
    for item in items:
        if kind and item.type.value != kind:
            continue
        if q:
            searchable = " ".join([
                item.title,
                item.description,
                item.author,
                " ".join(item.tags),
            ]).lower()
            if q not in searchable:
                continue
        results.append(item)
    return results


# ----------------------------------------------------------------------
# Default catalogue (built-in fallback)
# ----------------------------------------------------------------------
def _default_catalogue() -> List[MarketplaceItem]:
    """A small starter catalogue so the marketplace works out of the box."""
    return [
        MarketplaceItem(
            id="filesystem",
            name="filesystem",
            type=MarketplaceItemType.MCP_SERVER,
            title="Local Filesystem",
            description="Provides read/write access to the local filesystem via MCP.",
            author="modelprotocol",
            source="https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
            entry={
                "mcp": {
                    "config": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
                    }
                }
            },
            tags=["filesystem", "storage", "essential"],
            version="0.1.0",
            license="MIT"
        ),
        MarketplaceItem(
            id="git",
            name="git",
            type=MarketplaceItemType.MCP_SERVER,
            title="Git MCP Server",
            description="Lets agents inspect and modify Git repositories through MCP.",
            author="modelprotocol",
            source="https://github.com/modelcontextprotocol/servers/tree/main/src/git",
            entry={
                "mcp": {
                    "config": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-git"]
                    }
                }
            },
            tags=["git", "version-control", "development"],
            version="0.1.0",
            license="MIT"
        ),
        MarketplaceItem(
            id="github",
            name="github",
            type=MarketplaceItemType.MCP_SERVER,
            title="GitHub MCP Server",
            description="Provides read/write access to GitHub repositories, issues, and pull requests.",
            author="modelprotocol",
            source="https://github.com/modelcontextprotocol/servers/tree/main/src/github",
            entry={
                "mcp": {
                    "config": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"]
                    }
                }
            },
            tags=["github", "ci", "collaboration"],
            version="0.1.0",
            license="MIT"
        ),
        MarketplaceItem(
            id="brave-search",
            name="brave-search",
            type=MarketplaceItemType.MCP_SERVER,
            title="Brave Search MCP Server",
            description="Web search via Brave's search API with MCP.",
            author="ModelContextProtocol",
            source="https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search",
            entry={
                "mcp": {
                    "config": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-brave-search"]
                    }
                }
            },
            tags=["search", "web", "research"],
            version="0.1.0",
            license="MIT"
        ),
        MarketplaceItem(
            id="filesystem-readonly",
            name="filesystem-readonly",
            type=MarketplaceItemType.MCP_SERVER,
            title="Read-Only Filesystem",
            description="Provides read-only access to the local filesystem via MCP.",
            author="modelprotocol",
            source="https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
            entry={
                "mcp": {
                    "config": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "--readonly", "."]
                    }
                }
            },
            tags=["filesystem", "storage", "readonly", "safe"],
            version="0.1.0",
            license="MIT"
        ),
        MarketplaceItem(
            id="think",
            name="think",
            type=MarketplaceItemType.SKILL,
            title="Think Tool",
            description="A skill that makes the agent pause and reason aloud before acting.",
            author="agentux-community",
            source="https://github.com/agentux-community/skills/tree/main/think",
            entry={
                "skill": {
                    "module": "agentux_community.skills.think:think_skill",
                    "args": ""
                }
            },
            tags=["reasoning", "planning", "cognitive"],
            version="1.0.0",
            license="MIT"
        ),
        MarketplaceItem(
            id="todo",
            name="todo",
            type=MarketplaceItemType.SKILL,
            title="Todo List Skill",
            description="Provides persistent todo list capabilities through a simple skill interface.",
            author="agentux-community",
            source="https://github.com/agentux-community/skills/tree/main/todo",
            entry={
                "skill": {
                    "module": "agentux_community.skills.todo:todo_skill",
                    "args": ""
                }
            },
            tags=["productivity", "organization", "task-management"],
            version="1.0.0",
            license="MIT"
        )
    ]


class Marketplace:
    """Main marketplace class for managing skills and MCP servers."""

    def __init__(self):
        """Initialize the marketplace."""
        self._items: List[MarketplaceItem] = []
        self._load_items()

    def _load_items(self) -> None:
        """Load items from catalogue and update installation status."""
        # Load catalogue directly from disk, never through load_catalogue() --
        # that function builds a Marketplace instance, which would recurse.
        catalogue_items = _load_catalogue_items()

        # If catalogue is empty, seed with default items
        if not catalogue_items:
            catalogue_items = _default_catalogue()
            _save_catalogue_to_disk(catalogue_items)

        # Load installed status
        installed_names = set(_load_installed_from_disk())

        # Update items with installation status
        self._items = []
        for item in catalogue_items:
            # Create a copy with updated installation status
            item_dict = item.to_dict()
            item_dict['installed'] = item.name in installed_names
            # Always reset enabled status when loading - it will be set based on actual configuration
            item_dict['enabled'] = False
            self._items.append(MarketplaceItem.from_dict(item_dict))

        # Sort by name for consistent ordering
        self._items.sort(key=lambda x: x.name)

    def get_items_by_type(self, item_type: MarketplaceItemType) -> List[MarketplaceItem]:
        """Get items by type."""
        return [item for item in self._items if item.type == item_type]

    def search_items(self, query: str) -> List[MarketplaceItem]:
        """Search items by query."""
        return search_catalogue(query)

    def get_item(self, item_id: str) -> Optional[MarketplaceItem]:
        """Get item by ID."""
        for item in self._items:
            if item.id == item_id:
                return item
        return None

    def install_item(self, item_id: str) -> bool:
        """Install an item."""
        item = self.get_item(item_id)
        if not item:
            return False

        if install_entry(item.name):
            item.installed = True
            self._save_items()
            return True
        return False

    def uninstall_item(self, item_id: str) -> bool:
        """Uninstall an item."""
        item = self.get_item(item_id)
        if not item:
            return False

        if uninstall_entry(item.name):
            item.installed = False
            item.enabled = False  # Disable when uninstalling
            self._save_items()
            return True
        return False

    def enable_item(self, item_id: str) -> bool:
        """Enable an item."""
        item = self.get_item(item_id)
        if not item or not item.installed:
            return False

        item.enabled = True
        self._save_items()
        return True

    def disable_item(self, item_id: str) -> bool:
        """Disable an item."""
        item = self.get_item(item_id)
        if not item:
            return False

        item.enabled = False
        self._save_items()
        return True

    def is_installed(self, item_id: str) -> bool:
        """Check if item is installed."""
        item = self.get_item(item_id)
        return item is not None and item.installed

    def is_enabled(self, item_id: str) -> bool:
        """Check if item is enabled."""
        item = self.get_item(item_id)
        return item is not None and item.enabled

    def get_installed_items(self) -> List[MarketplaceItem]:
        """Get all installed items."""
        return [item for item in self._items if item.installed]

    def get_enabled_items(self) -> List[MarketplaceItem]:
        """Get all enabled items."""
        return [item for item in self._items if item.enabled]

    def _save_items(self) -> None:
        """Save current items to catalogue."""
        _save_catalogue_to_disk(self._items)

    @property
    def items(self) -> List[MarketplaceItem]:
        """Get all marketplace items."""
        return self._items.copy()

    def populate_default_items(self) -> None:
        """Populate the marketplace with default items if empty."""
        if not self._items:
            default_items = _default_catalogue()
            installed_names = set(load_installed())

            # Update with installation status
            for item in default_items:
                item.installed = item.name in installed_names

            self._items = default_items
            save_catalogue(self._items)


# Backwards compatibility functions - delegate to marketplace instance
_default_marketplace: Optional[Marketplace] = None


def _get_default_marketplace() -> Marketplace:
    """Get or create the default marketplace instance."""
    global _default_marketplace
    if _default_marketplace is None:
        _default_marketplace = Marketplace()
    return _default_marketplace


def load_catalogue() -> List[MarketplaceEntry]:
    """Load the marketplace catalogue from disk.

    Returns a list of MarketplaceEntry objects for backwards compatibility.
    """
    marketplace = _get_default_marketplace()
    # Convert MarketplaceItem to MarketplaceEntry for backwards compatibility
    entries = []
    for item in marketplace.items:
        entry_dict = item.to_dict()
        # Remove the 'type' field and add 'kind' field for MarketplaceEntry
        entry_dict.pop('type', None)
        entry_dict['kind'] = item.type.value
        entries.append(MarketplaceEntry.from_dict(entry_dict))
    return entries


def save_catalogue(entries: List[MarketplaceEntry]) -> None:
    """Overwrite the catalogue with the given entries."""
    marketplace = _get_default_marketplace()
    # Convert MarketplaceEntry to MarketplaceItem
    items = []
    for entry in entries:
        entry_dict = entry.to_dict()
        # Convert kind to type for MarketplaceItem
        entry_dict.pop('kind', None)
        if 'type' not in entry_dict:
            entry_dict['type'] = MarketplaceItemType.SKILL  # Default to skill
        items.append(MarketplaceItem.from_dict(entry_dict))
    marketplace._items = items
    marketplace._save_items()


def load_installed() -> List[str]:
    """Return the list of entry names the user has installed."""
    return _load_installed_from_disk()


def save_installed(names: List[str]) -> None:
    """Overwrite the installed list with the given names."""
    _save_installed_to_disk(names)


def install_entry(name: str) -> bool:
    """Mark an entry as installed; returns True if the name was not already present."""
    installed = _load_installed_from_disk()
    if name in installed:
        return False
    installed.append(name)
    _save_installed_to_disk(installed)
    return True


def uninstall_entry(name: str) -> bool:
    """Mark an entry as uninstalled; returns True if the name was present."""
    installed = _load_installed_from_disk()
    if name not in installed:
        return False
    installed.remove(name)
    _save_installed_to_disk(installed)
    return True


def is_installed(name: str) -> bool:
    """Check whether an entry is marked as installed."""
    return name in _load_installed_from_disk()


def search_catalogue(query: str = "", kind: Optional[str] = None) -> List[MarketplaceEntry]:
    """Return catalogue entries matching the query and/or kind.

    Query is a case-insensitive substring match on title, description, author, or tags.
    Kind is either "skill", "mcp", or None for both.
    Returns a list of MarketplaceEntry objects for backwards compatibility.
    """
    marketplace = _get_default_marketplace()
    items = marketplace.items
    if not query and not kind:
        # Convert to MarketplaceEntry for backwards compatibility
        entries = []
        for item in items:
            entry_dict = item.to_dict()
            entry_dict.pop('type', None)
            entry_dict['kind'] = item.type.value
            entries.append(MarketplaceEntry.from_dict(entry_dict))
        return entries

    q = query.lower()
    results: List[MarketplaceEntry] = []
    for item in items:
        if kind and item.type.value != kind:
            continue
        if q:
            searchable = " ".join([
                item.title,
                item.description,
                item.author,
                " ".join(item.tags),
            ]).lower()
            if q not in searchable:
                continue
        # Convert to MarketplaceEntry for backwards compatibility
        entry_dict = item.to_dict()
        entry_dict.pop('type', None)
        entry_dict['kind'] = item.type.value
        results.append(MarketplaceEntry.from_dict(entry_dict))
    return results


def seed_catalogue() -> None:
    """Populate the marketplace directory with the default catalogue if empty."""
    marketplace = _get_default_marketplace()
    marketplace.populate_default_items()


# Legacy functions for backwards compatibility
def load_catalogue_legacy() -> List[Dict[str, Any]]:
    """Legacy function for backwards compatibility."""
    # Create a temporary marketplace instance to avoid recursion
    marketplace_temp = Marketplace.__new__(Marketplace)
    marketplace_temp._items = []  # Initialize empty items
    # Manually load catalogue items without triggering full initialization
    catalogue_items = load_catalogue()
    if not catalogue_items:
        catalogue_items = _default_catalogue()
    # Convert to legacy format
    return [item.to_dict() for item in catalogue_items]


def save_catalogue_legacy(entries: List[Dict[str, Any]]) -> None:
    """Legacy function for backwards compatibility."""
    marketplace = _get_default_marketplace()
    items = [MarketplaceItem.from_dict(entry) for entry in entries]
    marketplace._items = items
    marketplace._save_items()


def load_installed_legacy() -> List[str]:
    """Legacy function for backwards compatibility."""
    return load_installed()


def save_installed_legacy(names: List[str]) -> None:
    """Legacy function for backwards compatibility."""
    save_installed(names)


def install_entry_legacy(name: str) -> bool:
    """Legacy function for backwards compatibility."""
    return install_entry(name)


def uninstall_entry_legacy(name: str) -> bool:
    """Legacy function for backwards compatibility."""
    return uninstall_entry(name)


def is_installed_legacy(name: str) -> bool:
    """Legacy function for backwards compatibility."""
    return is_installed(name)


def search_catalogue_legacy(query: str = "", kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """Legacy function for backwards compatibility."""
    items = search_catalogue(query, kind)
    return [item.to_dict() for item in items]


def seed_catalogue_legacy() -> None:
    """Legacy function for backwards compatibility."""
    marketplace = _get_default_marketplace()
    marketplace.populate_default_items()


def fetch_skills_from_sh() -> List[MarketplaceItem]:
    """Fetch skills from skills.sh (local file or URL).

    Attempts to load skills from a skills.sh file which should output
    skill definitions in a format compatible with MarketplaceItem.

    Returns:
        List of MarketplaceItem objects representing skills, or empty list if failed.
    """
    import subprocess
    import json

    try:
        # Try to run skills.sh as a script and capture output
        # Assuming it outputs JSON format that can be parsed
        result = subprocess.run(
            ['bash', 'skills.sh'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and result.stdout.strip():
            # Try to parse as JSON
            try:
                data = json.loads(result.stdout)
                # Convert to MarketplaceItem objects
                items = []
                if isinstance(data, list):
                    for skill_data in data:
                        if isinstance(skill_data, dict):
                            # Create a MarketplaceItem from the skill data
                            item = MarketplaceItem(
                                id=skill_data.get('id', skill_data.get('name', 'unknown')),
                                name=skill_data.get('name', 'unknown'),
                                type=MarketplaceItemType.SKILL,  # Assuming skills.sh contains skills
                                title=skill_data.get('title', skill_data.get('name', 'Unknown Skill')),
                                description=skill_data.get('description', ''),
                                author=skill_data.get('author', 'unknown'),
                                source=skill_data.get('source', ''),
                                entry=skill_data.get('entry', {"skill": {"module": "", "args": ""}}),
                                tags=skill_data.get('tags', []),
                                version=skill_data.get('version', '1.0.0'),
                                license=skill_data.get('license', 'MIT')
                            )
                            items.append(item)
                return items
            except json.JSONDecodeError:
                # If not JSON, try to parse as simple format
                pass
    except Exception:
        # Failed to fetch from skills.sh
        pass

    return []


def fetch_mcps_from_website() -> List[MarketplaceItem]:
    """Fetch MCP servers from mcpmarketplace.com.

    Attempts to load MCP server information from mcpmarketplace.com.

    Returns:
        List of MarketplaceItem objects representing MCP servers, or empty list if failed.
    """
    try:
        # Use the WebFetch tool equivalent - in a real implementation,
        # this would make an HTTP request to mcpmarketplace.com
        # For now, we'll return an empty list as a placeholder
        # In a production environment, this would parse the website or API
        return []
    except Exception:
        # Failed to fetch from mcpmarketplace.com
        pass

    return []


def _fetch_and_merge_external_items(self) -> None:
    """Fetch items from external sources and merge with existing catalogue.

    This method fetches skills from skills.sh and MCPs from mcpmarketplace.com,
    then merges them with the existing catalogue items.
    """
    # Fetch skills from skills.sh
    external_skills = fetch_skills_from_sh()

    # Fetch MCPs from mcpmarketplace.com
    external_mcps = fetch_mcps_from_website()

    # Combine all external items
    external_items = external_skills + external_mcps

    if not external_items:
        return  # No external items to add

    # Get current items and installed status to preserve it
    current_items = self._items
    installed_names = {item.name for item in current_items if item.installed}

    # Merge external items with current items, avoiding duplicates by name/id
    merged_items = []
    existing_ids = {item.id for item in current_items}
    existing_names = {item.name for item in current_items}

    # Add current items first
    merged_items.extend(current_items)

    # Add external items that don't already exist
    for item in external_items:
        if item.id not in existing_ids and item.name not in existing_names:
            # Preserve installed status if this item was previously installed
            item.installed = item.name in installed_names
            merged_items.append(item)
            existing_ids.add(item.id)
            existing_names.add(item.name)

    # Update the items list and save
    self._items = merged_items
    self._items.sort(key=lambda x: x.name)
    save_catalogue(self._items)


# Update the Marketplace class to call external fetch during initialization
# Uncomment the following line in _load_items to enable automatic fetching:
# self._fetch_and_merge_external_items()


__all__ = [
    "MarketplaceEntry",
    "MarketplaceItem",
    "MarketplaceItemType",
    "Marketplace",
    # Legacy functions for backwards compatibility
    "load_catalogue_legacy",
    "save_catalogue_legacy",
    "load_installed_legacy",
    "save_installed_legacy",
    "install_entry_legacy",
    "uninstall_entry_legacy",
    "is_installed_legacy",
    "search_catalogue_legacy",
    "seed_catalogue_legacy",
    # External fetch functions
    "fetch_skills_from_sh",
    "fetch_mcps_from_website",
]