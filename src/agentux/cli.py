"""Command-line interface.

Presentation only: argument parsing, terminal output, and the confirmation
prompt live here. Everything substantive is in :mod:`agentux.agent`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import __version__
from .agent import Agent
from .config import Config, load_config
from .errors import (
    AgentuxError,
    AuthenticationError,
    MaxStepsError,
    NetworkError,
    ProviderError,
    RateLimitError,
    SafetyError,
    UnsupportedToolCallingError,
    UserAbort,
)
from .providers.openai_compatible import OpenAICompatibleProvider
from .tools.filesystem import FilesystemTool
from .tools.project import ProjectTool
from .tools.registry import ToolRegistry
from .tools.shell import ShellTool

console = Console()
error_console = Console(stderr=True)

#: Commands run directly in the shell that are safe without asking.
SAFE_COMMANDS = {
    "ls", "pwd", "cd", "echo", "cat", "head", "tail", "wc", "which", "whoami",
    "date", "git", "grep", "find", "diff", "tree", "stat", "file", "env",
    "python", "python3", "pip", "pytest", "npm", "node", "make", "cargo", "go",
    "ruff", "mypy", "black", "flake8", "sort", "uniq", "du", "df",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentux",
        description=(
            "Agentux - a terminal-native AI agent. Works with any "
            "OpenAI-compatible provider (OpenAI, Ollama, LiteLLM, vLLM, ...)."
        ),
        epilog=(
            "Examples:\n"
            "  agentux\n"
            '  agentux "explain the architecture of this project"\n'
            '  agentux "fix the failing tests"\n'
            "  agentux --list-models\n"
            "  agentux --no-tools 'explain this shell script'\n"
            "\n"
            "Ollama (local, no API key needed):\n"
            '  export AGENTUX_BASE_URL="http://localhost:11434/v1"\n'
            '  export AGENTUX_MODEL="granite4.1:3b"\n'
            "  agentux\n"
            "\n"
            "Configuration (environment variables):\n"
            "  AGENTUX_API_KEY   API key (required for remote providers)\n"
            "  AGENTUX_BASE_URL  API root (default: https://api.openai.com/v1)\n"
            "  AGENTUX_MODEL     Model name (default: gpt-4o-mini)\n"
            "  AGENTUX_MAX_STEPS Maximum agent iterations (default: 25)\n"
            "  AGENTUX_TIMEOUT   Request/command timeout in seconds (default: 120)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt", nargs="?", help="Task for the agent. Omit to start an interactive session."
    )
    parser.add_argument("--version", action="version", version=f"agentux {__version__}")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List the models the configured provider offers, then exit.",
    )
    parser.add_argument("--model", help="Override AGENTUX_MODEL.")
    parser.add_argument(
        "--base-url", help="Override AGENTUX_BASE_URL (the API root, ending in /v1)."
    )
    parser.add_argument("--api-key", help="Override AGENTUX_API_KEY. Prefer the environment variable.")
    parser.add_argument(
        "--cwd", default=None, help="Workspace root for file and shell tools (default: current directory)."
    )
    parser.add_argument(
        "--max-steps", type=int, help="Maximum agent iterations before giving up."
    )
    parser.add_argument(
        "--no-tools", action="store_true", help="Disable all tools (plain conversation)."
    )
    parser.add_argument(
        "--read-only", action="store_true", help="Give the agent a read-only filesystem."
    )
    parser.add_argument(
        "--allow-destructive",
        action="store_true",
        help="Run destructive shell commands without asking. Use with care.",
    )
    parser.add_argument(
        "--timeout", type=float, help="Shell command timeout in seconds (default: 120)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show tool arguments in full."
    )
    parser.add_argument(
        "--system-prompt", default=None, help="Override the built-in system prompt."
    )
    return parser


def resolve_config(args: argparse.Namespace) -> Config:
    """Build the config, letting explicit flags win over the environment."""
    # argparse handles --version/--help before we get here, so an API key is
    # always required by the time this runs.
    config = load_config()
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.timeout is not None:
        config.timeout = args.timeout
    if args.cwd:
        config.working_dir = os.path.abspath(os.path.expanduser(args.cwd))
    return config.validate()


def _shorten(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more characters]"


def make_reporter(verbose: bool):
    """Render agent activity to the terminal as it happens."""

    def report(kind: str, text: str) -> None:
        if kind == "tool_call":
            console.print(f"  [dim]›[/dim] [bold cyan]{text}[/bold cyan]")
        elif kind == "tool_result":
            if text.startswith("ERROR:"):
                console.print(
                    f"  [red]✗[/red] [red]{_shorten(text, 400 if not verbose else 4000)}[/red]"
                )
            else:
                body = _shorten(text, 1200 if not verbose else 8000)
                console.print(
                    Panel(body, title="result", border_style="dim", padding=(0, 1))
                )

    return report


def make_confirmer():
    """Ask the user before running a destructive command.

    Returning False denies the command. EOF and Ctrl-C deny too: an
    unattended run must never escalate on a technicality.
    """

    def confirm(command: str, reasons: List[str]) -> bool:
        reason = ", ".join(reasons)
        console.print()
        console.print(
            Panel(
                f"[bold red]Destructive command[/bold red]\n\n"
                f"  [bold]{command}[/bold]\n\n"
                f"Flagged as: [yellow]{reason}[/yellow]",
                border_style="red",
            )
        )
        try:
            answer = console.input("[bold]Run it? [y/N] [/bold]").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False
        return answer in ("y", "yes")

    return confirm


async def list_models(config: Config) -> int:
    """Print the models the provider offers, flagging tool support."""
    provider = OpenAICompatibleProvider(config)
    lister = getattr(provider, "list_models_detailed", None)
    try:
        models = (
            await lister() if callable(lister) else
            [{"name": name} for name in await provider.list_models()]
        )
    except AgentuxError as exc:
        explain_error(exc)
        return 1

    if not models:
        console.print("[yellow]The provider returned no models.[/yellow]")
        return 1

    usable = [m for m in models if "tools" in (m.get("capabilities") or [])]
    detail = bool(models and models[0].get("capabilities") is not None)

    for model in sorted(models, key=lambda m: m["name"]):
        caps = model.get("capabilities")
        if detail:
            can_tools = "tools" in (caps or [])
            marker = "[green]✓ tools[/green]" if can_tools else "[red]✗ no tools[/red]"
            thinking = " [dim](thinking)[/dim]" if "thinking" in (caps or []) else ""
            console.print(f"  {model['name']:<40} {marker}{thinking}")
        else:
            console.print(f"  {model['name']}")

    console.print(f"\n[dim]{len(models)} model(s) available.[/dim]")
    if detail and usable:
        console.print(
            f"[dim]Models marked 'no tools' cannot run Agentux's tools; use "
            f"--no-tools with those.[/dim]"
        )
    elif detail:
        console.print(
            "[yellow]None of these models report tool support. Agentux needs a "
            "tool-capable model.[/yellow]"
        )
    return 0


def build_agent(config: Config, args: argparse.Namespace) -> Agent:
    """Wire up provider, tools, and agent. Kept separate so tests can call it."""
    registry = ToolRegistry()
    if not args.no_tools:
        if args.allow_destructive:
            registry.register(ShellTool(
                cwd=config.working_dir, timeout=config.timeout, allow_destructive=True
            ))
        else:
            registry.register(ShellTool(
                cwd=config.working_dir,
                timeout=config.timeout,
                confirm=make_confirmer(),
            ))
        registry.register(FilesystemTool(root=config.working_dir, read_only=args.read_only))
        registry.register(ProjectTool(root=config.working_dir))

    provider = OpenAICompatibleProvider(config)
    return Agent(
        config=config,
        provider=provider,
        registry=registry,
        system_prompt=args.system_prompt,
        on_event=make_reporter(args.verbose),
    )


def print_answer(answer: str) -> None:
    """Render the final response as markdown when it looks like markdown."""
    if any(marker in answer for marker in ("```", "\n- ", "\n#", "**")):
        console.print()
        console.print(Markdown(answer))
    else:
        console.print()
        console.print(answer)


def explain_error(exc: BaseException) -> None:
    """Print an actionable error message."""
    if isinstance(exc, AuthenticationError):
        error_console.print(f"[bold red]Authentication failed[/bold red]\n{exc}")
    elif isinstance(exc, RateLimitError):
        error_console.print(f"[bold yellow]Rate limited[/bold yellow]\n{exc}")
    elif isinstance(exc, NetworkError):
        error_console.print(f"[bold red]Network error[/bold red]\n{exc}")
    elif isinstance(exc, MaxStepsError):
        error_console.print(f"[bold yellow]Stopped early[/bold yellow]\n{exc}")
    elif isinstance(exc, SafetyError):
        error_console.print(f"[bold red]Command blocked[/bold red]\n{exc}")
    elif isinstance(exc, UnsupportedToolCallingError):
        error_console.print(f"[bold red]Tools unavailable[/bold red]\n{exc}")
    elif isinstance(exc, ProviderError):
        error_console.print(f"[bold red]Provider error[/bold red]\n{exc}")
    elif isinstance(exc, AgentuxError):
        error_console.print(f"[bold red]Error[/bold red]\n{exc}")
    else:
        error_console.print(f"[bold red]Unexpected error[/bold red]\n{type(exc).__name__}: {exc}")


async def run_once(agent: Agent, prompt: str) -> int:
    """Run a single request. Returns a process exit code."""
    try:
        with console.status("[bold blue]thinking...", spinner="dots"):
            answer = await agent.run(prompt)
    except AgentuxError as exc:
        explain_error(exc)
        return 1
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return 130
    except Exception as exc:
        # A bug in Agentux should read as a message, not a traceback dump.
        explain_error(exc)
        return 1
    print_answer(answer)
    return 0


async def run_interactive(agent: Agent) -> int:
    """Run a REPL, keeping history across turns."""
    console.print(
        Panel(
            f"[bold blue]Agentux {__version__}[/bold blue]\n"
            f"Model: [cyan]{agent.provider.get_model_name()}[/cyan]  |  "
            f"Workspace: [cyan]{agent.config.working_dir}[/cyan]\n"
            f"Tools: [cyan]{', '.join(agent.registry.names) or 'none'}[/cyan]\n\n"
            "[dim]/exit to quit, /reset to clear history, /tools to list tools[/dim]",
            border_style="blue",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold green]❯[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            return 0

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            return 0
        if user_input == "/reset":
            agent.reset()
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        if user_input == "/tools":
            console.print(
                "[dim]Tools:[/dim] "
                + (", ".join(agent.registry.names) or "none")
            )
            continue

        try:
            with console.status("[bold blue]thinking...", spinner="dots"):
                answer = await agent.run(user_input, interactive=True)
            print_answer(answer)
        except UserAbort:
            console.print("[yellow]Interrupted.[/yellow]")
        except AgentuxError as exc:
            explain_error(exc)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as exc:
            # Keep the session alive rather than dropping the user into a
            # traceback and losing their conversation.
            explain_error(exc)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``agentux`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = resolve_config(args)
    except AgentuxError as exc:
        error_console.print(f"[bold red]Configuration error[/bold red]\n{exc}")
        return 1

    if args.list_models:
        return asyncio.run(list_models(config))

    try:
        agent = build_agent(config, args)
    except AgentuxError as exc:
        explain_error(exc)
        return 1

    try:
        if args.prompt:
            return asyncio.run(run_once(agent, args.prompt))
        return asyncio.run(run_interactive(agent))
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return 130


def cli() -> None:
    """Console-script wrapper that propagates the exit code."""
    sys.exit(main())


if __name__ == "__main__":
    cli()
