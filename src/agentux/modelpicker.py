"""Discovering and switching models.

The provider's ``/models`` endpoint is the only source of truth about what is
available, so Agentux asks rather than guessing. A provider that does not
implement it is common -- gateways usually do not -- and that is not an error:
it just means switching is by name.

Three ways to pick, in descending order of certainty:

1. ``/model`` opens a picker when the terminal can support one, and falls back
   to a numbered table otherwise.
2. ``/model <name>`` takes the name directly, with unique-prefix matching so
   ``/model qwen`` works when there is exactly one qwen.
3. ``/model <number>`` takes an index into the same sorted list the table shows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from .errors import AgentuxError

if TYPE_CHECKING:  # pragma: no cover
    from .agent import Agent
    from .config import Config

#: Capability flags some providers report, and what we do with them.
_TOOL_CAPABILITY = "tools"


async def discover_models(config: "Config") -> Tuple[List[dict], Optional[BaseException]]:
    """Ask the provider what models it has.

    Returns ``(models, None)`` or ``([], error)``. A provider with no model
    listing returns an empty list with no error -- that is normal, not a
    failure, and the caller falls back to free-text entry.
    """
    from .providers.openai_compatible import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(config)
    lister = getattr(provider, "list_models_detailed", None)
    try:
        if callable(lister):
            models = await lister()
        else:
            models = [{"name": name} for name in await provider.list_models()]
    except AgentuxError as exc:
        return [], exc
    except Exception as exc:  # pragma: no cover - defensive
        return [], exc
    return list(models), None


def model_name(model: dict) -> str:
    return model.get("name") or "?"


def has_tools(model: dict) -> Optional[bool]:
    """Whether a model can call tools, or ``None`` if it did not say."""
    capabilities = model.get("capabilities")
    if capabilities is None:
        return None
    return _TOOL_CAPABILITY in capabilities


def sort_models(models: List[dict]) -> List[dict]:
    """The one order every listing uses, so a number means the same thing twice."""
    return sorted(models, key=model_name)


def match_model(models: List[dict], query: str) -> Optional[str]:
    """Resolve a partial, exact, or numeric reference to a real model name.

    Ambiguity resolves to ``None`` rather than to a guess: picking the wrong
    model silently is worse than asking.
    """
    query = query.strip()
    if not query:
        return None

    # The same order the listing printed. A number that indexed an unsorted
    # list would pick a different model than the row the user was looking at.
    names = [model_name(model) for model in sort_models(models)]

    # A number picks from the sorted order the listing printed.
    if query.isdigit():
        index = int(query) - 1
        return names[index] if 0 <= index < len(names) else None

    if query in names:
        return query

    lowered = query.lower()
    exact = [name for name in names if name.lower() == lowered]
    if exact:
        return exact[0]

    partial = [name for name in names if lowered in name.lower()]
    return partial[0] if len(partial) == 1 else None


def render_model_table(
    console: Console, models: List[dict], current: Optional[str] = None
) -> None:
    """Print discovered models, marking tool support and the current one."""
    if not models:
        console.print("[yellow]The provider returned no models.[/yellow]")
        return

    models = sort_models(models)
    # Capabilities are only shown when at least one model reports them; a
    # column of "unknown" is worse than no column.
    detail = any(model.get("capabilities") is not None for model in models)

    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column("", width=1, no_wrap=True)
    table.add_column("#", style="dim", justify="right", no_wrap=True)
    table.add_column("model", style="bold", no_wrap=True)
    if detail:
        table.add_column("tools", no_wrap=True)
        table.add_column("", no_wrap=True)

    for index, model in enumerate(models, start=1):
        name = model_name(model)
        marker = "[bold green]●[/bold green]" if name == current else " "
        if detail:
            supported = has_tools(model)
            if supported is True:
                tools = "[green]yes[/green]"
            elif supported is False:
                tools = "[red]no[/red]"
            else:
                tools = "[dim]?[/dim]"
            notes = [note for note in ("thinking", "vision") if note in (model.get("capabilities") or [])]
            table.add_row(marker, str(index), name, tools, "[dim]" + " ".join(notes) + "[/dim]")
        else:
            table.add_row(marker, str(index), name)

    console.print()
    console.print(table)

    if current:
        console.print(f"\n[dim]{len(models)} model(s). Current: {current}[/dim]")
    else:
        console.print(f"\n[dim]{len(models)} model(s) available.[/dim]")

    if not detail:
        console.print(
            "[dim]This provider does not report capabilities, so tool support "
            "is unknown.[/dim]"
        )
        return

    usable = [model for model in models if has_tools(model) is True]
    if usable:
        console.print(
            "[dim]Models marked 'no' cannot run Agentux's tools; use --no-tools "
            "with those.[/dim]"
        )
    else:
        console.print(
            "[yellow]None of these models report tool support. Agentux needs a "
            "tool-capable model.[/yellow]"
        )


#: The picker value that means "switch to Agentux Solo".
SOLO_CHOICE = "solo"


def switch_to_solo(agent: "Agent", console: Console) -> None:
    """Point the session at Agentux Solo.

    Rebuilds nothing: the Solo preset changes the model and base URL on the
    config, and the provider reads both from there on every request.
    """
    from .solo import SOLO_MODEL, apply_solo, solo_preset

    apply_solo(agent, solo_preset(agent.config.working_dir))
    console.print(
        f"[green]Switched to[/green] [bold]Agentux Solo[/bold] "
        f"([dim]{SOLO_MODEL} + memory[/dim])"
    )


def _picker_choices(models: List[dict], current: Optional[str]) -> List[Dict[str, Any]]:
    from .solo import SOLO_MODEL

    choices = [
        {
            "name": "Agentux Solo  (qwen2.5:0.5b + memory)",
            "value": SOLO_CHOICE,
            "enabled": current != SOLO_MODEL,
        }
    ]
    for model in sort_models(models):
        name = model_name(model)
        supported = has_tools(model)
        if supported is True:
            suffix = "  (tools)"
        elif supported is False:
            suffix = "  (no tools)"
        else:
            suffix = ""
        choices.append(
            {
                "name": f"{name}{suffix}",
                "value": name,
                "enabled": name != current,  # already selected: not a valid choice
            }
        )
    return choices


#: Picker outcomes. Kept distinct because "cancelled" and "unavailable" call
#: for different behaviour: a cancel should leave the user alone, while an
#: unavailable picker should fall back to the numbered table.
PICKED = "picked"
CANCELLED = "cancelled"
UNAVAILABLE = "unavailable"


def pick_model(
    console: Console, models: List[dict], current: Optional[str]
) -> Tuple[str, Optional[str]]:
    """An interactive picker. Returns ``(status, model_name)``.

    Any failure -- no InquirerPy, not a terminal, the user pressing ctrl-c --
    is reported as a status rather than raised. A picker is a convenience and
    must never become the only way to do something.
    """
    try:
        from InquirerPy import inquirer
    except Exception:
        return UNAVAILABLE, None

    choices = [choice for choice in _picker_choices(models, current) if choice["enabled"]]
    if not choices:
        return CANCELLED, None  # Only the current model exists; nothing to pick.

    try:
        answer = inquirer.select(
            message="Switch to which model?",
            choices=choices,
            default=next(
                (c["value"] for c in choices if c["value"] == current), choices[0]["value"]
            ),
            vim_mode=True,
        )
    except (KeyboardInterrupt, EOFError):
        return CANCELLED, None
    except Exception:
        return UNAVAILABLE, None

    return (PICKED, str(answer)) if answer else (CANCELLED, None)


def switch_model(agent: "Agent", target: str, console: Console) -> None:
    """Point the session at a different model.

    The provider is rebuilt rather than reconfigured: the HTTP client holds
    the model name, so mutating the config alone would leave the old model in
    place and quietly ignore the change.
    """
    from .providers.openai_compatible import OpenAICompatibleProvider

    current = agent.provider.get_model_name()
    if target == current:
        console.print(f"[dim]Already on {target}.[/dim]")
        return

    agent.config.model = target
    agent.provider = OpenAICompatibleProvider(agent.config)
    console.print(f"[green]Switched to[/green] [bold]{target}[/bold]")


async def handle_model(agent: "Agent", console: Console, argument: str = "") -> bool:
    """``/model`` with no argument lists; with one, switches.

    Returns ``True`` -- the command never ends the session, including when the
    provider cannot be reached.
    """
    query = argument.strip()
    if query.lower() == SOLO_CHOICE:
        switch_to_solo(agent, console)
        return True

    models, error = await discover_models(agent.config)

    if error is not None:
        console.print(f"[yellow]Could not list models: {error}[/yellow]")
        console.print(
            f"[dim]Staying on {agent.provider.get_model_name()}. "
            f"Switch with '/model <name>' or --model.[/dim]"
        )
        if query:
            # An explicit request still works when discovery failed.
            switch_model(agent, query, console)
        return True

    if not models:
        if query:
            switch_model(agent, query, console)
        else:
            console.print("[yellow]The provider returned no models.[/yellow]")
            console.print("[dim]Name one directly with '/model <name>'.[/dim]")
        return True

    current = agent.provider.get_model_name()

    if not query:
        if console.is_terminal:
            status, chosen = pick_model(console, models, current)
            if status == PICKED and chosen:
                if chosen == SOLO_CHOICE:
                    switch_to_solo(agent, console)
                else:
                    switch_model(agent, chosen, console)
                return True
            # A cancel means the user said no. Dumping the table on them
            # anyway is the kind of small rudeness that makes a tool feel
            # unresponsive.
            if status == CANCELLED:
                return True
        render_model_table(console, models, current=current)
        console.print(
            f"[dim]Switch with '/model <name>' or '/model <number>'. "
            f"Current: {current}[/dim]"
        )
        return True

    target = match_model(models, query)
    if target is None:
        console.print(
            f"[yellow]No model matching {query!r}. Run '/model' to see the list.[/yellow]"
        )
        return True

    switch_model(agent, target, console)
    return True


__all__ = [
    "discover_models",
    "match_model",
    "render_model_table",
    "handle_model",
    "sort_models",
    "has_tools",
]
