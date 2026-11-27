"""Slash commands for the interactive session.

A command is a name, a one-line summary, and a handler. The registry below is
the single source of truth: ``/help`` renders it, tab-completion offers it, and
the picker shows it -- so a command can never exist without being discoverable.

Handlers return ``True`` to keep the session running and ``False`` to end it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import save_settings
from .errors import AgentuxError
from . import theme
from .marketplace import (
    Marketplace,
    MarketplaceEntry,
    MarketplaceItemType,
    load_catalogue,
    load_installed,
    install_entry,
    is_installed,
    search_catalogue,
    seed_catalogue,
)
from .theme import (
    ACCENT_DIM,
    ACCENT_END_HEX,
    ACCENT_START_HEX,
    BORDER,
    BORDER_SUBTLE,
    CYAN,
    EMERALD,
    Mode,
    PINK,
    PURPLE,
    TEXT_DIM,
    TEXT_LIGHT,
    TEXT_MUTED,
    YELLOW,
)
from .ui import ACCENT_END_HEX, ACCENT_START_HEX, _shorten


class Command(NamedTuple):
    """A slash command. ``name`` is stored bare; users always type the slash."""

    name: str
    summary: str
    handler: Callable[["CommandContext", str], Awaitable[bool]]
    aliases: Tuple[str, ...] = ()


class CommandContext(NamedTuple):
    """Everything a command handler is allowed to touch.

    ``session`` is the CLI's dict of runtime toggles, so ``/set stream off``
    can change something that is not on :class:`~agentux.config.Config`.
    ``rebuild_tools`` re-roots the tools, which a ``/set cwd`` needs.
    """

    agent: Any
    view: Any
    mcp_manager: Any
    console: Console
    args: argparse.Namespace
    session: Dict[str, Any]
    rebuild_tools: Optional[Callable[[], Any]] = None


#: Populated by :func:`register_builtins`. Kept module-level so completion and
#: ``/help`` can read it without importing the CLI.
REGISTRY: Dict[str, Command] = {}
ALIASES: Dict[str, str] = {}


def register(command: Command) -> Command:
    """Add a command under its bare name, plus each of its aliases."""
    name = command.name.lstrip("/")
    if not name:
        raise ValueError("A command needs a name.")
    REGISTRY[name] = command._replace(name=name)
    for alias in command.aliases:
        ALIASES[alias.lstrip("/").lower()] = name
    return command


def resolve(text: str) -> Optional[Command]:
    """Look up a command by name or alias. ``None`` if there is no match.

    Only the first word is considered, so ``/model gpt-4o`` resolves to the
    ``/model`` command and leaves ``gpt-4o`` as the argument.
    """
    if not text:
        return None
    name = text.lstrip("/").split()[0].lower()
    name = ALIASES.get(name, name)
    return REGISTRY.get(name)


def names() -> List[str]:
    return sorted(REGISTRY)


# ----------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------
async def cmd_help(context: CommandContext, argument: str) -> bool:
    """Render every command."""
    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column("command", style=f"bold {ACCENT_START_HEX}", no_wrap=True)
    table.add_column("aliases", style="dim", no_wrap=True)
    table.add_column("what it does", overflow="fold")

    for name in names():
        command = REGISTRY[name]
        aliases = ", ".join(f"/{a}" for a in command.aliases)
        table.add_row(f"/{name}", aliases or "", command.summary)

    context.console.print()
    context.console.print(table)
    context.console.print()
    context.console.print(
        Text(
            "Type / on its own for a menu, or start typing to filter. "
            "Anything not starting with / goes to the agent.",
            style="dim",
        )
    )
    return True


async def cmd_exit(context: CommandContext, argument: str) -> bool:
    context.console.print("[dim]Bye.[/dim]")
    return False


async def cmd_reset(context: CommandContext, argument: str) -> bool:
    context.agent.reset()
    context.console.print("[dim]Conversation cleared. Context is back to empty.[/dim]")
    return True


async def cmd_tools(context: CommandContext, argument: str) -> bool:
    names = context.agent.registry.names
    if not names:
        context.console.print("[dim]No tools are available in this session.[/dim]")
        return True

    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column("tool", style=f"bold {ACCENT_START_HEX}", no_wrap=True)
    table.add_column("source", style=ACCENT_END_HEX, no_wrap=True)
    table.add_column("description", overflow="fold")

    for name in names:
        tool = context.agent.registry.get(name)
        source = "MCP" if name.startswith("mcp__") else "built-in"
        first = tool.description.strip().split("\n")[0]
        table.add_row(name, source, _shorten(first, 96))

    context.console.print()
    context.console.print(table)
    return True


async def cmd_context(context: CommandContext, argument: str) -> bool:
    """Show how full the conversation is and what has been spent."""
    view = context.view
    history = context.agent.history
    system = history[0].content if history and history[0].role == "system" else ""
    tools = context.agent.registry.definitions()

    from .usage import (
        conversation_tokens,
        estimate_tokens,
        format_tokens,
    )

    total = conversation_tokens(history, system or "", tools)

    tool_tokens = sum(
        estimate_tokens(t.name) + estimate_tokens(t.description)
        + estimate_tokens(str(t.parameters))
        for t in tools
    )
    message_tokens = total - estimate_tokens(system) - tool_tokens
    compressed = context.agent.context.compressed
    saved = context.agent.context.saved_chars

    body = Text()
    body.append("context\n")
    # append_text, not append: the stats carry their own styles, and Text.append
    # would take them as literal characters.
    body.append_text(view.context_stats())
    body.append("\n\nbreakdown (estimated)\n")
    body.append(f"  {'system prompt':<18}{format_tokens(estimate_tokens(system))}\n")
    body.append(f"  {'tool schemas':<18}{format_tokens(tool_tokens)}\n")
    body.append(f"  {'conversation':<18}{format_tokens(max(0, message_tokens))}\n")
    body.append(f"  {'total':<18}{format_tokens(total)}\n")

    body.append("\nthis session\n")
    body.append(f"  {'requests':<18}{context.view.usage.requests}\n")
    body.append(
        f"  {'tokens used':<18}"
        f"{format_tokens(context.view.usage.total_tokens)}"
        f"{'' if context.view.usage.reported else '  (estimated)'}\n"
    )
    body.append(f"  {'tool calls':<18}{len(context.agent.step_log)}\n")
    body.append(f"  {'messages':<18}{len(history)}\n")
    if compressed:
        body.append(
            f"  {'compressed':<18}{compressed} tool result(s), "
            f"{format_tokens(saved)} tokens saved\n"
        )

    context.console.print()
    context.console.print(Panel(body, border_style=ACCENT_START_HEX, padding=(1, 2)))
    context.console.print(
        Text(
            "Figures marked estimated are computed from text length; the "
            "provider did not report them.",
            style="dim",
        )
    )
    return True


async def cmd_set(context: CommandContext, argument: str) -> bool:
    """Show or change configuration without leaving Agentux."""
    from .settings import SETTABLE, TOGGLES, apply_setting, describe_settings, resolve_key, savable

    parts = argument.strip().split(None, 1)
    agent = context.agent
    config = agent.config

    if not parts:
        context.console.print()
        context.console.print(
            Panel(describe_settings(config), border_style=ACCENT_START_HEX, padding=(1, 2))
        )
        for name, summary in sorted(TOGGLES.items()):
            state = "on" if context.session.get(name) else "off"
            context.console.print(
                f"  {name:<12}[dim]{summary} — {state}[/dim]"
            )
        context.console.print()
        context.console.print(
            Text(
                "Change one with '/set <name> <value>'. Names: "
                + ", ".join(sorted(set(SETTABLE) | set(TOGGLES)))
                + ".  Keep them with '/set save'.",
                style="dim",
            )
        )
        return True

    key = parts[0].lower()
    value = parts[1].strip() if len(parts) > 1 else ""

    if key in ("save", "persist"):
        if value:
            context.console.print("[dim]Usage: /set save[/dim]")
            return True
        try:
            path = save_settings(savable(config))
        except AgentuxError as exc:
            context.console.print(f"[red]{exc}[/red]")
            return True
        context.console.print(f"[green]Saved preferences to[/green] {path}")
        context.console.print(
            Text(
                "Your API key was not saved. Keep it in AGENTUX_API_KEY, or set "
                "it again with '/set api-key' next time you start.",
                style="dim",
            )
        )
        return True

    resolved = resolve_key(key)
    if resolved is None:
        context.console.print(
            f"[yellow]Unknown setting {key!r}. Try one of: "
            f"{', '.join(sorted(set(SETTABLE) | set(TOGGLES)))}.[/yellow]"
        )
        return True

    if not value:
        context.console.print(
            f"[yellow]'/set {resolved}' needs a value. "
            f"Try '/set {resolved} <value>'.[/yellow]"
        )
        return True

    try:
        message = await apply_setting(
            agent,
            resolved,
            value,
            session=context.session,
            rebuild_tools=context.rebuild_tools,
        )
    except AgentuxError as exc:
        context.console.print(f"[red]{exc}[/red]")
        return True

    context.console.print(f"[green]{message}[/green]")
    return True


async def cmd_model(context: CommandContext, argument: str) -> bool:
    from .modelpicker import handle_model

    return await handle_model(context.agent, context.console, argument)


async def cmd_mcp(context: CommandContext, argument: str) -> bool:
    from .mcp_config import global_config_path

    manager = context.mcp_manager
    body = Text()

    if manager is None or not manager.servers:
        body.append("No MCP servers are configured.\n\n")
    else:
        for line in manager.describe():
            body.append(line + "\n")
        body.append("\n")
        tools = [t for t in (manager.tools or []) if t.server.connected]
        body.append(f"{len(tools)} tool(s) available from MCP\n")

    body.append("\nconfig files\n")
    body.append(f"  global   {global_config_path()}\n")
    body.append(f"  project  {context.agent.config.working_dir}/.agentux/mcp.json\n")

    context.console.print()
    context.console.print(Panel(body, border_style=ACCENT_START_HEX, padding=(1, 2)))

    if manager and manager.warnings:
        for warning in manager.warnings:
            context.console.print(f"[yellow]![/yellow] {warning}")

    context.console.print(
        Text(
            'Add a server like {"mcpServers": {"name": {"command": "npx", '
            '"args": ["-y", "package"]}}}. Agentux works fine without any.',
            style="dim",
        )
    )
    return True


async def cmd_marketplace(context: CommandContext, argument: str) -> bool:
    """Show and interact with the marketplace for skills and MCP servers."""
    marketplace = Marketplace()

    # Simple argument parsing
    if not argument.strip():
        # Show marketplace overview
        await _show_marketplace_overview(context, marketplace)
        return True

    args = argument.strip().split()
    command = args[0].lower() if args else ""

    if command == "search" and len(args) > 1:
        query = " ".join(args[1:])
        await _search_marketplace(context, marketplace, query)
    elif command == "install" and len(args) > 1:
        item_id = args[1]
        await _install_marketplace_item(context, marketplace, item_id)
    elif command == "uninstall" and len(args) > 1:
        item_id = args[1]
        await _uninstall_marketplace_item(context, marketplace, item_id)
    elif command == "enable" and len(args) > 1:
        item_id = args[1]
        await _enable_marketplace_item(context, marketplace, item_id)
    elif command == "disable" and len(args) > 1:
        item_id = args[1]
        await _disable_marketplace_item(context, marketplace, item_id)
    elif command == "list":
        await _list_marketplace_items(context, marketplace)
    elif command == "update" or command == "refresh":
        await _update_marketplace_external(context, marketplace)
    elif command == "info" and len(args) > 1:
        item_id = args[1]
        await _show_marketplace_item_info(context, marketplace, item_id)
    else:
        await _show_marketplace_help(context)

    return True


async def _show_marketplace_overview(context: CommandContext, marketplace: Marketplace) -> None:
    """Show marketplace overview with categories."""
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel

    context.console.print()

    # Create overview table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Item", style="dim", width=20)
    table.add_column("Type", width=12)
    table.add_column("Description", width=40)
    table.add_column("Status", width=12)
    table.add_column("Version", width=8)

    # Add skills
    skills = marketplace.get_items_by_type(MarketplaceItemType.SKILL)
    for skill in skills:
        status = "○ Available"
        if skill.installed:
            status = "✓ Installed"
        if skill.enabled:
            status = "⚡ Enabled"
        table.add_row(
            skill.name,
            "Skill",
            skill.description[:38] + ("..." if len(skill.description) > 38 else ""),
            status,
            skill.version
        )

    # Add MCP servers
    servers = marketplace.get_items_by_type(MarketplaceItemType.MCP_SERVER)
    for server in servers:
        status = "○ Available"
        if server.installed:
            status = "✓ Installed"
        if server.enabled:
            status = "⚡ Enabled"
        table.add_row(
            server.name,
            "MCP Server",
            server.description[:38] + ("..." if len(server.description) > 38 else ""),
            status,
            server.version
        )

    context.console.print(table)
    context.console.print()

    # Show help with better formatting
    help_text = Text()
    help_text.append("Commands: ", style="bold")
    help_text.append("marketplace search <query> - Search for items\n", style="dim")
    help_text.append("marketplace install <id> - Install an item\n", style="dim")
    help_text.append("marketplace uninstall <id> - Uninstall an item\n", style="dim")
    help_text.append("marketplace enable <id> - Enable an installed item\n", style="dim")
    help_text.append("marketplace disable <id> - Disable an installed item\n", style="dim")
    help_text.append("marketplace list - List all items\n", style="dim")
    help_text.append("marketplace update - Fetch latest skills and MCPs from external sources\n", style="dim")
    help_text.append("marketplace refresh - Alias for update\n", style="dim")
    help_text.append("marketplace info <id> - Show detailed item info\n", style="dim")
    help_text.append("marketplace - Show this overview\n", style="dim")

    context.console.print(Panel(help_text, title="Marketplace Help", border_style=BORDER_SUBTLE))


async def _search_marketplace(context: CommandContext, marketplace: Marketplace, query: str) -> None:
    """Search the marketplace for items matching the query."""
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel

    results = marketplace.search_items(query)

    if not results:
        context.console.print(f"[yellow]No items found matching '{query}'[/yellow]")
        return

    context.console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", width=10)
    table.add_column("Name", width=20)
    table.add_column("Type", width=12)
    table.add_column("Description", width=30)
    table.add_column("Installed", width=10)

    for item in results:
        table.add_row(
            item.id[:8],
            item.name,
            item.type.value.title(),
            item.description[:28] + ("..." if len(item.description) > 28 else ""),
            "✓" if item.installed else "○"
        )

    context.console.print(table)
    context.console.print(f"[dim]Found {len(results)} item(s)[/dim]")


async def _install_marketplace_item(context: CommandContext, marketplace: Marketplace, item_id: str) -> None:
    """Install a marketplace item."""
    if marketplace.install_item(item_id):
        item = marketplace.get_item(item_id)
        if item:
            context.console.print(f"[green]✓ Installed {item.name}[/green]")
        else:
            context.console.print("[green]✓ Item installed[/green]")
    else:
        context.console.print(f"[red]✗ Item with ID '{item_id}' not found[/red]")


async def _uninstall_marketplace_item(context: CommandContext, marketplace: Marketplace, item_id: str) -> None:
    """Uninstall a marketplace item."""
    if marketplace.uninstall_item(item_id):
        item = marketplace.get_item(item_id)
        if item:
            context.console.print(f"[yellow]○ Uninstalled {item.name}[/yellow]")
        else:
            context.console.print("[yellow]○ Item uninstalled[/yellow]")
    else:
        context.console.print(f"[red]✗ Item with ID '{item_id}' not found[/red]")


async def _enable_marketplace_item(context: CommandContext, marketplace: Marketplace, item_id: str) -> None:
    """Enable a marketplace item."""
    if marketplace.enable_item(item_id):
        item = marketplace.get_item(item_id)
        if item:
            context.console.print(f"[green]✓ Enabled {item.name}[/green]")
        else:
            context.console.print("[green]✓ Item enabled[/green]")
    else:
        context.console.print(f"[red]✗ Item with ID '{item_id}' not found or not installed[/red]")


async def _disable_marketplace_item(context: CommandContext, marketplace: Marketplace, item_id: str) -> None:
    """Disable a marketplace item."""
    if marketplace.disable_item(item_id):
        item = marketplace.get_item(item_id)
        if item:
            context.console.print(f"[yellow]○ Disabled {item.name}[/yellow]")
        else:
            context.console.print("[yellow]○ Item disabled[/yellow]")
    else:
        context.console.print(f"[red]✗ Item with ID '{item_id}' not found or not installed[/red]")


async def _list_marketplace_items(context: CommandContext, marketplace: Marketplace) -> None:
    """List all marketplace items."""
    from rich.table import Table

    context.console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", width=10)
    table.add_column("Name", width=20)
    table.add_column("Type", width=12)
    table.add_column("Description", width=30)
    table.add_column("Status", width=12)
    table.add_column("Version", width=8)

    for item in marketplace.items:
        status = "○ Available"
        if item.installed:
            status = "✓ Installed"
        if item.enabled:
            status = "⚡ Enabled"
        table.add_row(
            item.id[:8],
            item.name,
            item.type.value.title(),
            item.description[:28] + ("..." if len(item.description) > 28 else ""),
            status,
            item.version
        )

    context.console.print(table)
    context.console.print(f"[dim]Total: {len(marketplace.items)} items[/dim]")


async def _show_marketplace_item_info(context: CommandContext, marketplace: Marketplace, item_id: str) -> None:
    """Show detailed information about a marketplace item."""
    from rich.panel import Panel
    from rich.text import Text

    item = marketplace.get_item(item_id)
    if not item:
        context.console.print(f"[red]✗ Item with ID '{item_id}' not found[/red]")
        return

    context.console.print()

    info_text = Text()
    info_text.append(f"Name: {item.name}\n", style="bold")
    info_text.append(f"Type: {item.type.value.title()}\n", style="dim")
    info_text.append(f"Description: {item.description}\n")
    info_text.append(f"Version: {item.version}\n")
    info_text.append(f"Author: {item.author}\n")
    if item.homepage:
        info_text.append(f"Homepage: {item.homepage}\n")
    if item.tags:
        info_text.append(f"Tags: {', '.join(item.tags)}\n")
    info_text.append(f"Installed: {'Yes' if item.installed else 'No'}\n")
    info_text.append(f"Enabled: {'Yes' if item.enabled else 'No'}\n")
    if item.entry_point:
        info_text.append(f"Entry Point: {item.entry_point}\n")
    if item.config:
        info_text.append(f"Configuration: {json.dumps(item.config, indent=2)}\n")

    # Format timestamps
    import datetime
    created = datetime.datetime.fromtimestamp(item.created_at).strftime("%Y-%m-%d %H:%M:%S")
    updated = datetime.datetime.fromtimestamp(item.updated_at).strftime("%Y-%m-%d %H:%M:%S")
    info_text.append(f"Created: {created}\n")
    info_text.append(f"Updated: {updated}\n")

    context.console.print(Panel(info_text, title=f"Item Info: {item.name}", border_style=BORDER))


async def _show_marketplace_help(context: CommandContext) -> None:
    """Show marketplace help."""
    from rich.text import Text
    from rich.panel import Panel

    help_text = Text()
    help_text.append("Marketplace Commands:\n\n", style="bold")
    help_text.append("marketplace              - Show marketplace overview\n", style="dim")
    help_text.append("marketplace search <q>   - Search for items\n", style="dim")
    help_text.append("marketplace install <id> - Install an item\n", style="dim")
    help_text.append("marketplace uninstall <id> - Uninstall an item\n", style="dim")
    help_text.append("marketplace enable <id>  - Enable an installed item\n", style="dim")
    help_text.append("marketplace disable <id> - Disable an installed item\n", style="dim")
    help_text.append("marketplace list         - List all items\n", style="dim")
    help_text.append("marketplace update       - Fetch latest skills and MCPs from external sources\n", style="dim")
    help_text.append("marketplace refresh      - Alias for update\n", style="dim")
    help_text.append("marketplace info <id>    - Show detailed item info\n", style="dim")
    help_text.append("\nExamples:\n", style="bold")
    help_text.append("  marketplace search dataviz\n", style="dim")
    help_text.append("  marketplace install <skill-id>\n", style="dim")
    help_text.append("  marketplace enable <mcp-server-id>\n", style="dim")
    help_text.append("  marketplace update\n", style="dim")

    context.console.print()
    context.console.print(Panel(help_text, title="Marketplace Help", border_style=BORDER_SUBTLE))


async def _update_marketplace_external(context: CommandContext, marketplace: Marketplace) -> None:
    """Update marketplace with items from external sources."""
    from rich.panel import Panel
    from rich.text import Text

    context.console.print()
    context.console.print("[yellow]Fetching skills from skills.sh and MCPs from mcpmarketplace.com...[/yellow]")

    # Call the external fetch method
    marketplace._fetch_and_merge_external_items()

    # Show results
    skills_count = len(marketplace.get_items_by_type(MarketplaceItemType.SKILL))
    mcps_count = len(marketplace.get_items_by_type(MarketplaceItemType.MCP_SERVER))

    result_text = Text()
    result_text.append("Marketplace updated successfully!\n\n")
    result_text.append(f"Skills available: {skills_count}\n")
    result_text.append(f"MCP servers available: {mcps_count}\n")
    result_text.append("\nUse 'marketplace list' to see all items or 'marketplace search <query>' to search.")

    context.console.print(Panel(result_text, title="Update Complete", border_style=BORDER))


async def cmd_mode(context: CommandContext, argument: str) -> bool:
    """Set or show the current operation mode."""
    tui = context.agent.tui  # Access the TUI instance from the agent

    if not argument.strip():
        # Show current mode with visual indicator
        mode_name = tui.mode.value.upper()
        mode_icons = {
            Mode.CHAT: "►",
            Mode.BUILD: "⚡",
            Mode.PLAN: "📝"
        }
        mode_colors = {
            Mode.CHAT: "green",
            Mode.BUILD: "cyan",
            Mode.PLAN: "blue"
        }
        icon = mode_icons.get(tui.mode, "")
        color = mode_colors.get(tui.mode, "white")

        context.console.print()
        context.console.print(f"[{color} bold]{icon} Current mode: {mode_name}[/{color} bold]")
        context.console.print()
        context.console.print("[dim]Available modes:[/dim]")
        context.console.print("  [green]► chat[/green] - Normal conversational mode")
        context.console.print("  [cyan]⚡ build[/cyan] - Focused on building/creating")
        context.console.print("  [blue]📝 plan[/blue] - Focused on planning/designing")
        context.console.print()
        context.console.print("Usage: /mode <chat|build|plan>")
        return True

    # Parse the mode argument
    mode_str = argument.strip().lower()
    try:
        new_mode = Mode(mode_str)
    except ValueError:
        context.console.print(f"[red]Invalid mode: {mode_str}[/red]")
        context.console.print("[dim]Available modes: chat, build, plan[/dim]")
        return True

    # Set the new mode
    old_mode = tui.mode
    tui.mode = new_mode

    # Update the UI to reflect the new mode with visual feedback
    mode_icons = {
        Mode.CHAT: "►",
        Mode.BUILD: "⚡",
        Mode.PLAN: "📝"
    }
    mode_colors = {
        Mode.CHAT: "green",
        Mode.BUILD: "cyan",
        Mode.PLAN: "blue"
    }
    icon = mode_icons.get(new_mode, "")
    color = mode_colors.get(new_mode, "white")

    context.console.print()
    context.console.print(f"[{color} bold]✓ Switched from {old_mode.value.upper()} to {icon} {new_mode.value.upper()}[/{color} bold]")

    # Trigger a UI redraw to update any mode-dependent styling
    if hasattr(tui, '_redraw'):
        tui._redraw()

    return True


async def cmd_skills(context: CommandContext, argument: str) -> bool:
    """Manage and execute Agentux skills."""
    agent = context.agent

    # Simple argument parsing
    if not argument.strip():
        # Show skills overview
        await _show_skills_overview(context, agent)
        return True

    args = argument.strip().split()
    command = args[0].lower() if args else ""

    if command == "list" and len(args) > 1:
        subcommand = args[1].lower()
        if subcommand == "available":
            await _list_available_skills(context, agent)
        elif subcommand == "loaded":
            await _list_loaded_skills(context, agent)
        else:
            await _show_skills_help(context)
    elif command == "execute" and len(args) > 2:
        skill_name = args[1]
        operation = args[2] if len(args) > 2 else "run"
        # Parse additional arguments as key=value pairs
        kwargs = {}
        for arg in args[3:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                kwargs[key] = value
            else:
                kwargs[arg] = True
        await _execute_skill(context, agent, skill_name, operation, **kwargs)
    elif command == "browse":
        await _browse_marketplace_skills(context, agent)
    elif command == "install" and len(args) > 1:
        skill_id = args[1]
        await _install_marketplace_skill(context, agent, skill_id)
    else:
        await _show_skills_help(context)

    return True


async def _show_skills_overview(context: CommandContext, agent: Any) -> None:
    """Show skills overview."""
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    context.console.print()

    # Get skills information
    available_skills = agent.get_available_skills()
    loaded_skills = agent.get_loaded_skills()

    # Create overview table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Skill", style="dim", width=25)
    table.add_column("Status", width=12)
    table.add_column("Description", width=40)

    # Add built-in skills
    built_in_skills = ["reasoning", "self_improvement", "marketplace_integration"]
    for skill in built_in_skills:
        status = "⚡ Loaded" if skill in loaded_skills else "○ Available"
        description = {
            "reasoning": "Advanced reasoning and planning capabilities",
            "self_improvement": "Self-improvement and learning capabilities",
            "marketplace_integration": "Marketplace integration for dynamic skill loading"
        }.get(skill, "No description available")

        table.add_row(
            skill,
            status,
            description
        )

    # Add pictured skills from marketplace integration if available
    try:
        marketplace_skill = agent.skills.get_skill("marketplace_integration")
        if marketplace_skill:
            # This would normally call the skill to get marketplace status
            # For now, we'll show placeholder info
            table.add_row(
                "marketplace_skills",
                "📦 Available",
                "Skills from Agentux marketplace"
            )
    except Exception:
        pass

    context.console.print(table)
    context.console.print()

    # Show help
    help_text = Text()
    help_text.append("Skills Commands: ", style="bold")
    help_text.append("skills list available - List available skills\n", style="dim")
    help_text.append("skills list loaded - List currently loaded skills\n", style="dim")
    help_text.append("skills execute <skill> <operation> [args] - Execute a skill\n", style="dim")
    help_text.append("skills browse - Browse marketplace for skills\n", style="dim")
    help_text.append("skills install <id> - Install a skill from marketplace\n", style="dim")
    help_text.append("skills - Show this overview\n", style="dim")

    context.console.print(Panel(help_text, title="Skills Help", border_style=BORDER_SUBTLE))


async def _list_available_skills(context: CommandContext, agent: Any) -> None:
    """List available skills."""
    from rich.table import Table
    from rich.panel import Panel

    available_skills = agent.get_available_skills()

    context.console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Skill Name", style="dim", width=25)
    table.add_column("Description", width=50)

    skill_descriptions = {
        "reasoning": "Advanced reasoning and planning capabilities for complex problem solving",
        "self_improvement": "Self-improvement and learning capabilities to extend agent functionality",
        "marketplace_integration": "Integrates with marketplace to dynamically load skills and MCP servers"
    }

    for skill in sorted(available_skills):
        description = skill_descriptions.get(skill, "No description available")
        table.add_row(skill, description)

    context.console.print(table)
    context.console.print(f"[dim]Total: {len(available_skills)} available skills[/dim]")


async def _list_loaded_skills(context: CommandContext, agent: Any) -> None:
    """List loaded skills."""
    from rich.table import Table
    from rich.panel import Panel

    loaded_skills = agent.get_loaded_skills()

    context.console.print()

    if not loaded_skills:
        context.console.print("[yellow]No skills currently loaded[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Skill Name", style="dim", width=25)
    table.add_column("Status", width=15)
    table.add_column("Info", width=40)

    for skill in sorted(loaded_skills):
        table.add_row(
            skill,
            "⚡ Loaded",
            "Skill is loaded and ready for execution"
        )

    context.console.print(table)
    context.console.print(f"[dim]Total: {len(loaded_skills)} loaded skills[/dim]")


async def _execute_skill(context: CommandContext, agent: Any,
                       skill_name: str, operation: str, **kwargs) -> None:
    """Execute a skill operation."""
    from rich.panel import Panel
    from rich.text import Text

    context.console.print()

    try:
        # Execute the skill
        result = await agent.execute_skill(skill_name, operation, **kwargs)

        result_text = Text()
        result_text.append(f"Skill: {skill_name}\n", style="bold")
        result_text.append(f"Operation: {operation}\n")
        if kwargs:
            result_text.append(f"Arguments: {kwargs}\n")
        result_text.append(f"Result: {result}\n")

        context.console.print(Panel(result_text, title=f"Skill Execution Result", border_style=BORDER))

    except Exception as exc:
        from .errors import AgentuxError
        if isinstance(exc, AgentuxError):
            context.console.print(f"[red]Skill error: {exc}[/red]")
        else:
            context.console.print(f"[red]Unexpected error: {exc}[/red]")


async def _browse_marketplace_skills(context: CommandContext, agent: Any) -> None:
    """Browse marketplace for skills."""
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    context.console.print()

    try:
        # Get marketplace integration skill
        marketplace_skill = agent.skills.get_skill("marketplace_integration")
        if marketplace_skill:
            # Browse marketplace for skills
            result = await marketplace_skill.execute("browse_marketplace", item_type="skill")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("ID", width=10)
            table.add_column("Name", width=20)
            table.add_column("Description", width=30)
            table.add_column("Version", width=10)
            table.add_column("Status", width=12)

            for item in result.get("items", [])[:10]:  # Show first 10
                status = "○ Available"
                if item.get("installed"):
                    status = "✓ Installed"
                if item.get("enabled"):
                    status = "⚡ Enabled"

                table.add_row(
                    item.get("id", "")[:8],
                    item.get("name", ""),
                    item.get("description", "")[:28] + ("..." if len(item.get("description", "")) > 28 else ""),
                    item.get("version", ""),
                    status
                )

            context.console.print(table)
            context.console.print(f"[dim]Found {result.get('total_available', 0)} total skills[/dim]")
        else:
            context.console.print("[yellow]Marketplace integration skill not available[/yellow]")

    except Exception as exc:
        context.console.print(f"[red]Error browsing marketplace: {exc}[/red]")


async def _install_marketplace_skill(context: CommandContext, agent: Any, skill_id: str) -> None:
    """Install a skill from the marketplace."""
    from rich.panel import Panel
    from rich.text import Text

    context.console.print()

    try:
        # Get marketplace integration skill
        marketplace_skill = agent.skills.get_skill("marketplace_integration")
        if marketplace_skill:
            # Install the skill
            result = await marketplace_skill.execute("install_skill", skill_id=skill_id)

            if result.get("status") in ["installed_and_loaded", "installed_load_failed"]:
                status_icon = "✓" if "installed" in result.get("status", "") else "⚠"
                status_color = "green" if "installed" in result.get("status", "") else "yellow"

                result_text = Text()
                result_text.append(f"Skill ID: {skill_id}\n", style="bold")
                result_text.append(f"Status: {result.get('status', 'unknown')}\n", style=f"{status_color}")
                if result.get("skill_name"):
                    result_text.append(f"Skill Name: {result.get('skill_name')}\n")
                if result.get("loaded") is not None:
                    result_text.append(f"Loaded: {result.get('loaded')}\n")
                if result.get("load_error"):
                    result_text.append(f"Load Error: {result.get('load_error')}\n")

                context.console.print(Panel(result_text, title="Installation Result", border_style=BORDER))
            else:
                context.console.print(f"[red]Installation failed: {result.get('message', 'Unknown error')}[/red]")
        else:
            context.console.print("[yellow]Marketplace integration skill not available[/yellow]")

    except Exception as exc:
        context.console.print(f"[red]Error installing skill: {exc}[/red]")


async def _show_skills_help(context: CommandContext) -> None:
    """Show skills help."""
    from rich.text import Text
    from rich.panel import Panel

    help_text = Text()
    help_text.append("Skills Commands:\n\n", style="bold")
    help_text.append("skills                    - Show skills overview\n", style="dim")
    help_text.append("skills list available     - List available skills\n", style="dim")
    help_text.append("skills list loaded        - List loaded skills\n", style="dim")
    help_text.append("skills execute <skill> <op> [args] - Execute skill operation\n", style="dim")
    help_text.append("skills browse             - Browse marketplace for skills\n", style="dim")
    help_text.append("skills install <id>       - Install skill from marketplace\n", style="dim")
    help_text.append("\nExamples:\n", style="bold")
    help_text.append("  skills list available\n", style="dim")
    help_text.append("  skills execute reasoning task_decomposition task_description=\"Build a web app\"\n", style="dim")
    help_text.append("  skills browse\n", style="dim")
    help_text.append("  skills install <skill-id-from-browse>\n", style="dim")

    context.console.print()
    context.console.print(Panel(help_text, title="Skills Help", border_style=BORDER_SUBTLE))


async def cmd_commit(context: CommandContext, argument: str) -> bool:
    """Stage changes and record a git commit."""
    import subprocess
    cwd = context.agent.config.working_dir
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        context.console.print(Panel(f"[red]Not a git repository: {cwd}[/red]", border_style="red"))
        return True
    changes = proc.stdout.strip()
    if not changes:
        context.console.print(Panel("[dim]Working tree is clean. Nothing to commit.[/dim]", border_style=BORDER_SUBTLE))
        return True

    msg = argument.strip()
    if not msg:
        changed_lines = changes.splitlines()
        first_file = changed_lines[0][3:].strip()
        scope = os.path.dirname(first_file) or os.path.splitext(first_file)[0]
        msg = f"feat({scope}): update {len(changed_lines)} file(s)\n\n" + "\n".join(f"- {l[3:].strip()}" for l in changed_lines[:8])

    subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
    commit_proc = subprocess.run(["git", "commit", "-m", msg], cwd=cwd, capture_output=True, text=True)
    if commit_proc.returncode == 0:
        context.console.print(Panel(f"[bold emerald]✓ Committed successfully[/bold emerald]\n\n[dim]{commit_proc.stdout.strip()}[/dim]", border_style=EMERALD, title="Git Commit"))
    else:
        context.console.print(Panel(f"[red]Git commit failed:[/red]\n{commit_proc.stderr.strip()}", border_style="red"))
    return True


async def cmd_diff(context: CommandContext, argument: str) -> bool:
    """Show git diff in a syntax-highlighted panel."""
    import subprocess
    cwd = context.agent.config.working_dir
    args = ["git", "diff"]
    if "staged" in argument.lower() or "--cached" in argument.lower():
        args.append("--cached")
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        context.console.print(Panel(f"[red]Git diff failed:[/red] {proc.stderr}", border_style="red"))
        return True
    diff_text = proc.stdout.strip()
    if not diff_text:
        context.console.print(Panel("[dim]No diff. Working tree matches index.[/dim]", border_style=BORDER_SUBTLE))
        return True
    from rich.syntax import Syntax
    syntax = Syntax(diff_text, "diff", theme="monokai", word_wrap=True)
    context.console.print(Panel(syntax, title="Git Diff", border_style=BORDER))
    return True


async def cmd_status(context: CommandContext, argument: str) -> bool:
    """Show git branch and working tree status."""
    import subprocess
    cwd = context.agent.config.working_dir
    branch_proc = subprocess.run(["git", "branch", "--show-current"], cwd=cwd, capture_output=True, text=True)
    branch = branch_proc.stdout.strip() or "detached"
    status_proc = subprocess.run(["git", "status", "--short"], cwd=cwd, capture_output=True, text=True)
    status = status_proc.stdout.strip() or "(working tree clean)"
    body = f"[bold cyan]Branch:[/bold cyan] {branch}\n[bold cyan]Workspace:[/bold cyan] {cwd}\n\n[dim]{status}[/dim]"
    context.console.print(Panel(body, title="Git Status", border_style=BORDER_SUBTLE))
    return True


async def cmd_init(context: CommandContext, argument: str) -> bool:
    """Scan workspace and generate or refresh AGENTUX.md project memory."""
    cwd = context.agent.config.working_dir
    target = os.path.join(cwd, "AGENTUX.md")

    techs = []
    build_cmds = []
    if os.path.isfile(os.path.join(cwd, "pyproject.toml")):
        techs.append("Python (pyproject.toml)")
        build_cmds.append("pytest")
    if os.path.isfile(os.path.join(cwd, "package.json")):
        techs.append("Node/TypeScript (package.json)")
        build_cmds.append("npm test")
    if os.path.isfile(os.path.join(cwd, "go.mod")):
        techs.append("Go (go.mod)")
        build_cmds.append("go test ./...")
    if os.path.isfile(os.path.join(cwd, "Cargo.toml")):
        techs.append("Rust (Cargo.toml)")
        build_cmds.append("cargo test")
    if os.path.isfile(os.path.join(cwd, "Makefile")):
        techs.append("Make (Makefile)")
        build_cmds.append("make test")

    tech_str = ", ".join(techs) or "Generic project"
    test_str = "\n".join(f"- `{cmd}`" for cmd in build_cmds) or "- (configure automated tests here)"

    content = f"""# Agentux Project Memory

## Overview
- **Workspace:** `{os.path.basename(cwd)}`
- **Detected Stack:** {tech_str}

## Build & Test Commands
{test_str}

## Architecture & Code Guidelines
- Targeted modifications: Prefer deterministic edits over wholesale rewrites.
- Verification: Always run tests before claiming success.
- Safety: Strictly preserve access boundaries and existing comments.
"""
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)

    context.agent._system_prompt = context.agent._load_project_memory(context.agent._system_prompt)
    context.console.print(Panel(f"[bold emerald]✓ Initialized AGENTUX.md[/bold emerald]\n\nSaved to [cyan]{target}[/cyan] and loaded into agent memory.", title="Project Memory", border_style=EMERALD))
    return True


async def cmd_compact(context: CommandContext, argument: str) -> bool:
    """Compact conversation history and minimize token consumption."""
    turns = 4
    if argument.strip().isdigit():
        turns = int(argument.strip())
    compacted = context.agent.compact(keep_turns=turns)
    if compacted > 0:
        context.console.print(Panel(f"[bold emerald]✓ Context Compacted[/bold emerald]\n\nSummarized {compacted} older message(s). Preserved last {turns} turn(s).", border_style=EMERALD))
    else:
        context.console.print(Panel("[dim]Context is already minimal. Nothing to compact.[/dim]", border_style=BORDER_SUBTLE))
    return True


async def cmd_fix(context: CommandContext, argument: str) -> bool:
    """Execute test suite and run self-healing agent turn if failures occur."""
    import subprocess
    cwd = context.agent.config.working_dir
    cmd = argument.strip()
    if not cmd:
        if os.path.isfile(os.path.join(cwd, "pyproject.toml")) or os.path.isfile(os.path.join(cwd, "pytest.ini")):
            cmd = "pytest"
        elif os.path.isfile(os.path.join(cwd, "go.mod")):
            cmd = "go test ./..."
        elif os.path.isfile(os.path.join(cwd, "package.json")):
            cmd = "npm test"
        elif os.path.isfile(os.path.join(cwd, "Cargo.toml")):
            cmd = "cargo test"
        elif os.path.isfile(os.path.join(cwd, "Makefile")):
            cmd = "make test"
        else:
            context.console.print(Panel("[yellow]Could not auto-detect test runner. Usage: /fix <test_command>[/yellow]", border_style="yellow"))
            return True

    context.console.print(f"[dim]Running tests: {cmd}...[/dim]")
    proc = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if proc.returncode == 0:
        context.console.print(Panel(f"[bold emerald]✓ All tests green![/bold emerald]\n\n[dim]{proc.stdout.strip()}[/dim]", border_style=EMERALD, title="Test Passed"))
        return True

    error_output = (proc.stdout + "\n" + proc.stderr).strip()
    context.console.print(Panel(f"[bold yellow]Tests failed (exit {proc.returncode}):[/bold yellow]\n\n{error_output[:1000]}\n\n[cyan]Triggering self-healing fix turn...[/cyan]", border_style="yellow", title="Auto-Fix Triggered"))

    prompt = (
        f"The test command `{cmd}` failed with the following output:\n\n"
        f"```\n{error_output[:3000]}\n```\n\n"
        "Diagnose the failure, inspect the affected files, apply a minimal fix with the filesystem tool, "
        f"and run `{cmd}` using the shell tool to verify the fix."
    )
    answer = await context.agent.run(prompt, interactive=True)
    if not context.agent.answer_streamed:
        context.console.print(answer)
    return True


async def cmd_search(context: CommandContext, argument: str) -> bool:
    """Perform a live technical web search."""
    q = argument.strip()
    if not q:
        context.console.print(Panel("[yellow]Usage: /search <query>[/yellow]", border_style="yellow"))
        return True
    from .tools.web import WebSearchTool
    tool = WebSearchTool()
    context.console.print(f"[dim]Searching web for: {q}...[/dim]")
    res = await tool.execute(query=q, max_results=5)
    context.console.print(Panel(res, title=f"Web Search: {q}", border_style=CYAN))
    return True


async def cmd_image(context: CommandContext, argument: str) -> bool:
    """Render an image in the terminal using Chafa."""
    path = argument.strip()
    if not path:
        context.console.print(Panel("[yellow]Usage: /image <path_or_url>[/yellow]", border_style="yellow"))
        return True
    from .tools.image import ViewImageTool
    tool = ViewImageTool(root=context.agent.config.working_dir)
    res = await tool.execute(path=path)
    context.console.print(Panel(res, title=f"Image: {os.path.basename(path)}", border_style=BORDER))
    return True


async def cmd_web(context: CommandContext, argument: str) -> bool:
    """Show information about the embedded Web UI console."""
    msg = (
        "🌐 [bold emerald]Agentux Web Developer Console[/bold emerald]\n\n"
        "Launch the zero-dependency browser console with:\n"
        "  [cyan]agentux --web --open[/cyan]\n\n"
        "Features real-time Server-Sent Events (SSE) token streaming, interactive tool accordions, "
        "and human-in-the-loop tool approvals directly in your browser."
    )
    context.console.print(Panel(msg, title="Web Console", border_style=EMERALD))
    return True


async def cmd_undo(context: CommandContext, argument: str) -> bool:
    """Roll back file modifications made by the last agent turn or to a specific checkpoint."""
    checkpoint_mgr = getattr(context.agent, "checkpoint_manager", None)
    if not checkpoint_mgr:
        context.console.print("[red]Checkpoint manager not initialized on agent.[/red]")
        return True

    target_id = argument.strip() or None
    success, message = checkpoint_mgr.rollback(target_id)
    border = EMERALD if success else PINK
    title = f"Undo Snapshot {'('+target_id+')' if target_id else '(Latest)'}"
    context.console.print(Panel(message, title=title, border_style=border))
    return True


async def cmd_checkpoints(context: CommandContext, argument: str) -> bool:
    """List recorded checkpoints with rollback IDs and timestamps."""
    checkpoint_mgr = getattr(context.agent, "checkpoint_manager", None)
    if not checkpoint_mgr:
        context.console.print("[red]Checkpoint manager not initialized.[/red]")
        return True

    cps = checkpoint_mgr.list_checkpoints()
    if not cps:
        context.console.print(Panel("[dim]No checkpoints recorded in this session yet.[/dim]", title="Checkpoints", border_style=BORDER_SUBTLE))
        return True

    table = Table(title="Session Checkpoints", border_style=BORDER_SUBTLE)
    table.add_column("ID", style="bold cyan")
    table.add_column("Time", style="dim")
    table.add_column("Description", style="white")
    table.add_column("Files", style="green")

    for c in reversed(cps):
        table.add_row(c.id, c.formatted_time, c.description, str(len(c.files)))

    context.console.print(table)
    return True


async def cmd_repomap(context: CommandContext, argument: str) -> bool:
    """Generate and display the codebase repository AST symbol map."""
    repomap = getattr(context.agent, "repomap", None)
    if not repomap:
        from .repomap import RepoMap
        repomap = RepoMap(context.agent.config.working_dir or os.getcwd())

    query = argument.strip() or None
    out = repomap.generate_map(max_tokens=800, query=query)
    from rich.syntax import Syntax
    syntax = Syntax(out, "yaml", theme="monokai", word_wrap=True)
    title = f"Repository Map{' (' + query + ')' if query else ''}"
    context.console.print(Panel(syntax, title=title, border_style=CYAN))
    return True


async def cmd_save(context: CommandContext, argument: str) -> bool:
    """Persist current conversation session to disk."""
    store = getattr(context.agent, "session_store", None)
    if not store:
        from .session_store import SessionStore
        store = SessionStore(context.agent.config.working_dir)

    session_id = argument.strip() or f"session-{int(time.time())}"
    path = store.save(
        session_id=session_id,
        history=context.agent.history,
        model=context.agent.config.model,
        working_dir=context.agent.config.working_dir or os.getcwd(),
        title=argument.strip() or None,
    )
    context.console.print(Panel(
        f"[bold green]Session saved successfully![/bold green]\n"
        f"• Session ID: [bold cyan]{session_id}[/bold cyan]\n"
        f"• Path:       [dim]{path}[/dim]\n"
        f"• Messages:   {len(context.agent.history)}",
        title="Session Saved",
        border_style=EMERALD,
    ))
    return True


async def cmd_load(context: CommandContext, argument: str) -> bool:
    """Restore a saved conversation session."""
    session_id = argument.strip()
    if not session_id:
        context.console.print("[yellow]Usage: /load <session_id>[/yellow]")
        return True

    store = getattr(context.agent, "session_store", None)
    if not store:
        from .session_store import SessionStore
        store = SessionStore(context.agent.config.working_dir)

    data = store.load(session_id)
    if not data:
        context.console.print(f"[red]Could not find session matching '{session_id}'. Run /sessions to list.[/red]")
        return True

    context.agent.history = data["messages"]
    context.console.print(Panel(
        f"[bold green]Loaded session '{data.get('title', session_id)}'![/bold green]\n"
        f"• Model:      {data.get('model', 'unknown')}\n"
        f"• Messages:   {len(context.agent.history)}\n"
        f"• Workspace:  {data.get('working_dir', '')}",
        title="Session Restored",
        border_style=EMERALD,
    ))
    return True


async def cmd_sessions(context: CommandContext, argument: str) -> bool:
    """List all saved conversation sessions."""
    store = getattr(context.agent, "session_store", None)
    if not store:
        from .session_store import SessionStore
        store = SessionStore(context.agent.config.working_dir)

    sessions = store.list_sessions()
    if not sessions:
        context.console.print(Panel("[dim]No saved sessions found. Use '/save [name]' to save current state.[/dim]", title="Sessions", border_style=BORDER_SUBTLE))
        return True

    table = Table(title="Saved Sessions", border_style=BORDER_SUBTLE)
    table.add_column("ID", style="bold cyan")
    table.add_column("Title / First Prompt", style="white")
    table.add_column("Model", style="dim")
    table.add_column("Turns", style="green")

    for s in sessions:
        table.add_row(s["id"], s["title"][:50], s["model"], str(s["turn_count"]))

    context.console.print(table)
    return True


async def cmd_subagent(context: CommandContext, argument: str) -> bool:
    """Delegate a task to a specialist subagent: /subagent <researcher|coder|reviewer|architect> <task>"""
    parts = argument.strip().split(maxsplit=1)
    if not parts:
        msg = (
            "🤖 [bold]Specialist Autonomous Subagents[/bold]\n\n"
            "Usage: [cyan]/subagent <role> <task>[/cyan]\n\n"
            "Roles:\n"
            "  • [bold cyan]researcher[/bold cyan] - Read-only codebase explorer & symbol finder\n"
            "  • [bold emerald]coder[/bold emerald]      - Implementation specialist for targeted code changes\n"
            "  • [bold yellow]reviewer[/bold yellow]   - QA & code review (git diffs, tests, regressions)\n"
            "  • [bold purple]architect[/bold purple]  - Architecture, layout, and task breakdown"
        )
        context.console.print(Panel(msg, title="Subagent Delegation", border_style=CYAN))
        return True

    role = parts[0]
    task = parts[1] if len(parts) > 1 else ""
    if not task:
        context.console.print(f"[yellow]Please provide a task description for subagent '{role}'.[/yellow]")
        return True

    mgr = getattr(context.agent, "subagent_manager", None)
    if not mgr:
        from .subagents.manager import SubagentManager
        mgr = SubagentManager(context.agent)

    context.console.print(f"[dim]Spawning specialist '{role}' subagent...[/dim]")
    report = await mgr.run_subagent(role, task)
    context.console.print(Panel(report, title=f"Subagent Report: {role.upper()}", border_style=EMERALD))
    return True


async def cmd_test(context: CommandContext, argument: str) -> bool:
    """Run tests or execute test-driven self-healing loop: /test [command]"""
    from .workflow import SelfHealingWorkflow
    workflow = SelfHealingWorkflow(context.agent, test_command=argument.strip() or None)

    context.console.print(f"[dim]Executing test suite: {workflow.test_command}...[/dim]")
    code, output = workflow.run_tests_sync()
    if code == 0:
        context.console.print(Panel(
            f"[bold green]✓ All tests passed successfully![/bold green]\n\n[dim]{output[-1000:]}[/dim]",
            title="Tests Passed",
            border_style=EMERALD,
        ))
    else:
        context.console.print(Panel(
            f"[bold red]✗ Tests failed (exit code {code})[/bold red]\n\n{output[-1500:]}\n\n"
            "[yellow]Tip: Run '/fix' to trigger an autonomous self-healing repair cycle.[/yellow]",
            title="Tests Failed",
            border_style=PINK,
        ))
    return True


def register_builtins() -> None:
    """Install the built-in commands.

    Idempotent, and called on import: a command that exists but was never
    registered is a command that silently does nothing, and the failure would
    only show up as "unknown command" for a user who read it in ``/help``.
    """
    register(Command("help", "List every command and what it does", cmd_help, ("h", "?")))
    register(Command("model", "List the provider's models, or switch to one", cmd_model, ("models",)))
    register(Command("set", "Show or change settings for this session", cmd_set, ("settings",)))
    register(Command("tools", "List the tools this agent can call", cmd_tools, ()))
    register(Command("mcp", "Show MCP server status", cmd_mcp, ()))
    register(Command("marketplace", "Browse and install skills and MCP servers", cmd_marketplace, ("market",)))
    register(Command("context", "Show token and context usage", cmd_context, ("stats", "usage")))
    register(Command("mode", "Set or show operation mode", cmd_mode, ()))
    register(Command("reset", "Clear the conversation and start over", cmd_reset, ("clear",)))
    register(Command("exit", "Quit Agentux", cmd_exit, ("quit", "q")))
    register(Command("skills", "Manage and execute Agentux skills", cmd_skills, ()))
    register(Command("commit", "Stage changes and create a git commit", cmd_commit, ()))
    register(Command("diff", "Show unstaged or staged git diff", cmd_diff, ()))
    register(Command("status", "Show git branch and working tree status", cmd_status, ()))
    register(Command("init", "Initialize or refresh AGENTUX.md project memory", cmd_init, ()))
    register(Command("compact", "Compact conversation history to minimize token usage", cmd_compact, ()))
    register(Command("fix", "Run test suite and trigger self-healing fix turn on failure", cmd_fix, ()))
    register(Command("undo", "Rollback file modifications from last turn or to a checkpoint", cmd_undo, ("revert",)))
    register(Command("checkpoints", "List recorded shadow checkpoints", cmd_checkpoints, ()))
    register(Command("repomap", "Generate compact AST repository symbol map", cmd_repomap, ("map",)))
    register(Command("save", "Save conversation session to disk", cmd_save, ()))
    register(Command("load", "Restore saved conversation session", cmd_load, ()))
    register(Command("sessions", "List all saved conversation sessions", cmd_sessions, ()))
    register(Command("subagent", "Delegate task to specialist autonomous subagent", cmd_subagent, ("delegate",)))
    register(Command("test", "Run project test suite", cmd_test, ()))
    register(Command("search", "Perform live technical web search", cmd_search, ()))
    register(Command("image", "Render image in terminal using Chafa", cmd_image, ()))
    register(Command("web", "Information and launch for embedded Web UI console", cmd_web, ()))


register_builtins()


__all__ = ["Command", "CommandContext", "REGISTRY", "register", "resolve", "names",
           "register_builtins"]