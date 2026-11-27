"""Command-line interface.

Presentation only: argument parsing, terminal output, and the confirmation
prompt live here. Everything substantive is in :mod:`agentux.agent`.

MCP is optional. If no servers are configured -- or every one of them fails to
start -- Agentux runs exactly as it did before, with its built-in tools.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .agent import Agent
from .config import Config, load_config
from .context import ContextManager
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
from .mcp_config import global_config_path, project_config_path
from .modelpicker import discover_models, handle_model, render_model_table
from .providers.openai_compatible import OpenAICompatibleProvider
from .tools.filesystem import FilesystemTool
from .tools.git import GitTool
from .tools.image import ViewImageTool
from .tools.project import ProjectTool
from .tools.registry import ToolRegistry
from .tools.search import SearchTool
from .tools.shell import ShellTool
from .tools.symbols import OutlineSymbolsTool
from .tools.todo import TodoTool
from .tools.web import FetchWebPageTool, WebSearchTool
from .tools.repomap_tool import RepomapTool
from .ui import SessionView, StreamRenderer

console = Console()
error_console = Console(stderr=True)

#: Commands run directly in the shell that are safe without asking.
SAFE_COMMANDS = {
    "ls", "pwd", "cd", "echo", "cat", "head", "tail", "wc", "which", "whoami",
    "date", "git", "grep", "find", "diff", "tree", "stat", "file", "env",
    "python", "python3", "pip", "pytest", "npm", "node", "make", "cargo", "go",
    "ruff", "mypy", "black", "flake8", "sort", "uniq", "du", "df",
}


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
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
            "  agentux --list-tools\n"
            "  agentux --no-tools 'explain this shell script'\n"
            "\n"
            "In an interactive session (tab completes, / alone lists them):\n"
            "  /help             every command\n"
            "  /model            discover models; /model <name> switches\n"
            "  /set              show settings; /set model x changes one\n"
            "  /set save         keep preferences for next time\n"
            "  /tools  /mcp      what is available\n"
            "  /context          token and context usage\n"
            "  /reset  /exit     clear the conversation, quit\n"
            "\n"
            "Ollama (local, no API key needed):\n"
            '  export AGENTUX_BASE_URL="http://localhost:11434/v1"\n'
            '  export AGENTUX_MODEL="qwen3:14b"\n'
            "  agentux\n"
            "\n"
            "Configuration, highest precedence first:\n"
            "  AGENTUX_API_KEY   API key (required for remote providers)\n"
            "  AGENTUX_BASE_URL  API root (default: https://api.openai.com/v1)\n"
            "  AGENTUX_MODEL     Model name (default: gpt-4o-mini)\n"
            "  AGENTUX_MAX_STEPS Maximum agent iterations (default: 25)\n"
            "  AGENTUX_TIMEOUT   Request/command timeout in seconds (default: 120)\n"
            "  then ~/.config/agentux/settings.json, written by '/set save'\n"
            "\n"
            "MCP servers are read from ~/.config/agentux/mcp.json and\n"
            "<project>/.agentux/mcp.json. They are optional; Agentux works\n"
            "with none configured. See README.md for the format.\n"
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
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List the built-in and MCP tools available, then exit.",
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
        "--timeout", type=float, help="Shell command timeout in seconds (default: 120)."
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
        "--no-mcp",
        action="store_true",
        help="Skip MCP servers entirely, even if configured.",
    )
    parser.add_argument(
        "--no-lsp",
        action="store_true",
        help="Disable semantic code navigation tools (go_to_definition, find_references, get_symbol_info).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not print the answer as it is generated (nicer for piping).",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Use the scrolling prompt instead of the full-screen interface.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show tool arguments in full."
    )
    parser.add_argument(
        "--system-prompt", default=None, help="Override the built-in system prompt."
    )
    parser.add_argument(
        "--text-tools",
        action="store_true",
        default=False,
        help="Enable text-based tool execution format for local models (e.g. qwen2.5-coder:3b).",
    )
    parser.add_argument(
        "--lean",
        action="store_true",
        default=False,
        help="Use a compact, lightweight system prompt (10x faster for local models like granite4.1:3b).",
    )
    parser.add_argument(
        "--solo",
        action="store_true",
        default=False,
        help=(
            "Run Agentux Solo: qwen2.5:0.5b with a persistent memory it reads "
            "every turn and writes when it learns."
        ),
    )
    parser.add_argument(
        "--solo-engine",
        action="store_true",
        default=False,
        help=(
            "Run Agentux Solo through the autonomous research engine. Solo "
            "searches its knowledge database first, researches the web when it "
            "does not know, stores what it learns, and verifies it was saved."
        ),
    )
    return parser


def resolve_config(args: argparse.Namespace) -> Config:
    """Build the config, letting explicit flags win over the environment."""
    # Solo runs against local Ollama, which needs no credential. The flag has
    # to be honoured before the key check, which otherwise reads the configured
    # base URL and demands a key for it.
    solo = getattr(args, "solo", False)
    config = load_config(require_api_key=not solo)
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
    if getattr(args, "text_tools", False):
        config.allow_text_tools = True
    if getattr(args, "lean", False):
        config.lean = True
    return config.validate()


def _clip(text: str, limit: int) -> str:
    """Truncate on a word boundary, with no marker left dangling."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


# ----------------------------------------------------------------------
# Terminal presentation
# ----------------------------------------------------------------------
def make_reporter(view: SessionView, verbose: bool, status: Optional["StatusLine"] = None):
    """Route agent events to the session view.

    The status line is refreshed after each event so its tool count and token
    total track the run instead of freezing at whatever they were when the
    turn began.
    """

    def report(kind: str, text: str) -> None:
        if kind == "tool_call":
            view.tool_call(text)
        elif kind == "tool_result":
            if text.startswith("ERROR:"):
                view.tool_error(text)
            else:
                view.tool_output(text, verbose)
        elif kind == "narration":
            view.narration(text)
        elif kind == "reflection":
            view.reflection(text)
        if status is not None:
            status.refresh()

    return report


class AnswerStream:
    """Owns the one live region the answer streams into.

    Markdown cannot be appended to -- a half-typed code fence is not the same
    document as a closed one -- so the accumulated text is re-rendered on every
    delta.

    The region is opened lazily, on the first delta, and the status line is
    stopped immediately before that happens. Rich permits exactly one live
    display at a time, and two of them do not merely overlap: the second
    raises, and the run dies mid-answer. So the handover is explicit rather
    than a matter of ordering luck.
    """

    def __init__(self, console: Console, enabled: bool) -> None:
        self.console = console
        self.enabled = enabled
        self._renderer: Optional[StreamRenderer] = None
        self._owned = False
        self.text = ""
        #: Called just before the live region opens, to release the terminal.
        self.handover: Optional[Callable[[], None]] = None

    @property
    def active(self) -> bool:
        """Whether the answer currently owns the terminal's live region."""
        return self._owned

    def start(self) -> None:
        """Open the region eagerly. Used when the answer is already known."""
        if not self.enabled or self._owned:
            return
        if self.handover is not None:
            self.handover()
        self.text = ""
        self._renderer = StreamRenderer(self.console, markdown=True)
        self._renderer.start()
        self._owned = True

    async def feed(self, piece: str) -> None:
        if not self.enabled or not piece:
            return
        if not self._owned:
            self.start()  # Lazy: the status line held the terminal until now.
        self.text += piece
        if self._renderer is not None:
            self._renderer.feed(piece)

    def stop(self) -> None:
        if not self._owned or self._renderer is None:
            self._owned = False
            return
        self.text = self._renderer.stop() or self.text
        self._renderer = None
        self._owned = False

    def set_enabled(self, enabled: bool) -> None:
        self.stop()
        self.enabled = enabled
        if not enabled:
            self.text = ""

    def finish(self) -> str:
        """Release the region and return the complete answer."""
        self.stop()
        return self.text


def make_stream_printer(stream: AnswerStream):
    """The ``on_text`` callback the agent streams answer deltas into."""

    async def print_text(piece: str) -> None:
        await stream.feed(piece)

    return print_text


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


def print_answer(answer: str) -> None:
    """Render the final response as markdown when it looks like markdown."""
    if any(marker in answer for marker in ("```", "\n- ", "\n#", "**")):
        console.print()
        console.print(Markdown(answer))
    else:
        console.print()
        console.print(answer)


# ----------------------------------------------------------------------
# Discovery commands
# ----------------------------------------------------------------------
async def list_models(config: Config) -> int:
    """Print the models the provider offers, flagging tool support."""
    models, error = await discover_models(config)
    if error is not None:
        explain_error(error)
        return 1
    if not models:
        console.print("[yellow]The provider returned no models.[/yellow]")
        return 1
    render_model_table(console, models, current=None)
    return 0


async def list_tools(config: Config, args: argparse.Namespace) -> int:
    """Print every tool the agent could call, built-in and MCP."""
    from .tools.mcp_tool import MCPToolManager

    registry = ToolRegistry()
    if not args.no_tools:
        _register_builtin_tools(registry, config, args)

    manager: Optional[MCPToolManager] = None
    if not args.no_tools and not args.no_mcp:
        manager = MCPToolManager(
            working_dir=config.working_dir,
            on_log=lambda message: console.print(f"[dim]{message}[/dim]"),
        )
        try:
            await manager.load()
            for tool in manager.tools:
                registry.register(tool)
        finally:
            await manager.close()

    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column("tool", style="bold cyan", no_wrap=True)
    table.add_column("source", style="magenta", no_wrap=True)
    table.add_column("description", style="", overflow="fold")

    for name in registry.names:
        tool = registry.get(name)
        source = "MCP" if name.startswith("mcp__") else "built-in"
        first = tool.description.strip().split("\n")[0]
        table.add_row(name, source, _clip(first, 120))

    console.print()
    console.print(table)

    if manager and manager.warnings:
        console.print()
        for warning in manager.warnings:
            console.print(f"[yellow]![/yellow] {warning}")

    console.print()
    console.print(
        f"[dim]{len(registry)} tool(s) · config: "
        f"{global_config_path()} and {project_config_path(config.working_dir)}[/dim]"
    )
    return 0


# ----------------------------------------------------------------------
# Agent construction
# ----------------------------------------------------------------------
def _register_builtin_tools(
    registry: ToolRegistry, config: Config, args, confirm=None
) -> None:
    if getattr(args, "allow_destructive", False):
        registry.register(ShellTool(
            cwd=config.working_dir, timeout=config.timeout, allow_destructive=True
        ))
    else:
        registry.register(ShellTool(
            cwd=config.working_dir,
            timeout=config.timeout,
            confirm=confirm if confirm is not None else make_confirmer(),
        ))
    registry.register(FilesystemTool(root=config.working_dir, read_only=args.read_only))
    registry.register(ProjectTool(root=config.working_dir))
    registry.register(SearchTool(root=config.working_dir))
    registry.register(OutlineSymbolsTool(root=config.working_dir))
    registry.register(GitTool(root=config.working_dir, timeout=config.timeout))
    registry.register(TodoTool())
    registry.register(ViewImageTool(root=config.working_dir))
    registry.register(WebSearchTool())
    registry.register(FetchWebPageTool())
    registry.register(RepomapTool(root=config.working_dir))

    # Semantic code navigation via LSP (gopls, pyright, rust-analyzer, ...)
    if not getattr(args, "no_lsp", False):
        from .tools.lsp import (
            FindReferencesTool,
            GetSymbolInfoTool,
            GoToDefinitionTool,
        )
        registry.register(GoToDefinitionTool(root=config.working_dir))
        registry.register(FindReferencesTool(root=config.working_dir))
        registry.register(GetSymbolInfoTool(root=config.working_dir))


def build_registry(config: Config, args: argparse.Namespace, mcp_manager=None,
                   confirm=None) -> ToolRegistry:
    """Build the tool registry for a session.

    Separate from :func:`build_agent` so ``/set cwd`` can re-root the tools
    without rebuilding the provider or losing the conversation.
    """
    registry = ToolRegistry()
    if getattr(args, "no_tools", False):
        return registry
    _register_builtin_tools(registry, config, args, confirm=confirm)
    if mcp_manager is not None:
        for tool in mcp_manager.tools:
            registry.register(tool)
    return registry


def enable_solo(agent, config: Config, allow_tools: bool = True, args=None) -> None:
    """Apply the Agentux Solo preset to an agent that has already been built.

    Called from :func:`build_agent` when ``--solo`` is passed, so every run
    mode (prompt, interactive, TUI, web) gets the same treatment.

    An explicitly passed ``--model`` flag takes precedence over the Solo
    preset's default model, so ``agentux --solo --model granite4.1:3b`` uses
    granite4.1:3b rather than qwen2.5:0.5b.
    """
    from .solo import apply_solo, solo_preset

    preset = solo_preset(config.working_dir)
    apply_solo(agent, preset)

    # Honor --model over the preset default.
    if args is not None and getattr(args, "model", None):
        config.model = args.model
        agent.config.model = args.model

    if not allow_tools:
        # '--no-tools --solo' means what it says: the memory still works
        # through automatic recall, but the model gets no explicit tools.
        for name in ("memory_save", "memory_search"):
            agent.registry.deregister(name)


def build_agent(
    config: Config,
    args: argparse.Namespace,
    mcp_manager=None,
    stream: Optional[AnswerStream] = None,
    view: Optional[SessionView] = None,
) -> Agent:
    """Wire up provider, tools, and agent. Kept separate so tests can call it."""
    registry = build_registry(config, args, mcp_manager)
    provider = OpenAICompatibleProvider(config)

    if stream is None:
        stream = AnswerStream(console, enabled=False)
    if view is None:
        view = SessionView(console, None, mcp_manager)

    agent = Agent(
        config=config,
        provider=provider,
        registry=registry,
        system_prompt=args.system_prompt,
        on_event=make_reporter(view, args.verbose),
        on_text=make_stream_printer(stream),
        context_manager=ContextManager(),
    )
    if getattr(args, "solo", False):
        enable_solo(agent, config, allow_tools=not getattr(args, "no_tools", False), args=args)
    return agent


async def attach_mcp(config: Config, args: argparse.Namespace):
    """Start the configured MCP servers, tolerating any that fail.

    Returns ``(manager, warnings)``. The manager may hold no working servers at
    all -- Agentux is fully functional on its built-in tools.
    """
    from .tools.mcp_tool import MCPToolManager

    manager = MCPToolManager(
        working_dir=config.working_dir,
        timeout=60.0,
        on_log=lambda message: console.print(f"  [dim]{message}[/dim]"),
    )

    if args.no_tools or args.no_mcp:
        return manager, []

    if not manager.configs and not os.path.isfile(global_config_path()):
        # Nothing configured anywhere: do not pay for two failed file reads
        # on every single invocation.
        return manager, []

    try:
        await manager.load()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        manager.warnings.append(f"MCP could not be started: {exc}")
    return manager, manager.warnings


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Run modes
# ----------------------------------------------------------------------
#: The prompt marker. A character, not an ASCII arrow, so it survives a font
#: that lacks the box-drawing glyphs.
PROMPT_MARK = "❯"


class StatusLine:
    """The one-line live status shown while the agent works.

    It has to hand the terminal over the moment the answer starts streaming,
    because two live regions on one terminal fight each other and the loser
    leaves a corrupted line behind. That handover is the whole reason this is
    a class and not a ``console.status`` context manager.
    """

    def __init__(self, view: SessionView, target: Console, enabled: bool = True) -> None:
        self.view = view
        self.console = target
        self.enabled = enabled and target.is_terminal
        self._live = None

    def start(self) -> None:
        if not self.enabled or self._live is not None:
            return
        from rich.live import Live

        # get_renderable, not a pre-rendered string: the elapsed time, the
        # tool count and the token total all change while this is up, and a
        # string captured at start would repaint the same stale line forever.
        self._live = Live(
            console=self.console,
            refresh_per_second=8,
            transient=True,
            get_renderable=self.view.status,
        )
        self._live.start()

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self.view.status())

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    @property
    def active(self) -> bool:
        return self._live is not None


def _make_usage_recorder(view: SessionView, agent: Agent):
    """A callback that keeps the session's token totals honest.

    Providers that report nothing still cost tokens, so an estimate is
    recorded too -- and the display is left saying which it is showing.
    """
    from .usage import estimate_tokens

    def record(usage: Optional[dict]) -> None:
        if usage:
            view.usage.record(usage)
            return
        # No numbers from the provider: fall back to a size-based estimate so
        # the counter still moves, and leave `reported` False so the UI marks
        # the figure as guessed rather than billed.
        prompt = sum(estimate_tokens(m.content) for m in agent.history)
        view.usage.record(None, fallback_prompt=prompt)

    return record


async def _run_turn(
    agent: Agent,
    prompt: str,
    view: SessionView,
    stream: "AnswerStream",
    status: StatusLine,
    verbose: bool = False,
    echo_prompt: bool = True,
) -> int:
    """Run one request with live output. Returns a process exit code."""
    if echo_prompt:
        console.print()
        console.print(f"[bold green]{PROMPT_MARK}[/bold green] {prompt.strip()[:200]}")
        console.print()

    # The status line owns the terminal for now. The answer takes it over
    # lazily, on the first delta, via the handover below.
    stream.handover = status.stop
    agent.on_event = make_reporter(view, verbose, status)
    status.start()

    try:
        answer = await agent.run(prompt, interactive=True)
    except UserAbort:
        status.stop()
        stream.stop()
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
    except KeyboardInterrupt:
        status.stop()
        stream.stop()
        console.print("\n[dim]Interrupted.[/dim]")
        return 130
    except AgentuxError as exc:
        status.stop()
        stream.stop()
        explain_error(exc)
        return 1
    except Exception as exc:
        status.stop()
        stream.stop()
        # A bug in Agentux should read as a message, not a traceback dump.
        explain_error(exc)
        return 1
    else:
        # The status goes first: it is transient, and leaving it up would
        # overwrite the answer that just streamed into the region above it.
        status.stop()
        streamed = stream.finish()
        # Only print what was not already shown live. Printing the answer
        # twice is the single most obvious way to look broken.
        if not agent.answer_streamed or not streamed:
            console.print()
            print_answer(answer)
        view.summary(agent.last_steps)
        return 0


async def run_once(
    agent: Agent,
    prompt: str,
    view: Optional[SessionView] = None,
    stream: Optional["AnswerStream"] = None,
) -> int:
    """Run a single request. Returns a process exit code."""
    if view is None:
        view = SessionView(console, agent)
    if stream is None:
        stream = AnswerStream(console, enabled=False)

    agent.on_usage = _make_usage_recorder(view, agent)
    agent.on_text = make_stream_printer(stream)

    status = StatusLine(view, console, enabled=stream.enabled)
    return await _run_turn(
        agent, prompt, view, stream, status, echo_prompt=stream.enabled
    )


async def run_tui(
    agent: Agent,
    mcp_manager=None,
    args: Optional[argparse.Namespace] = None,
    console: Optional[Console] = None,
) -> int:
    """Run the full-screen terminal interface.

    The state built here -- the view, the usage recorder, the command context
    and the tool rebuilder -- is the same set :func:`run_interactive` builds.
    The two differ only in how a turn is presented: there, output scrolls into
    the terminal behind a status line; here it is rendered into the transcript
    and redrawn in place.

    The agent's callbacks are pointed at the TUI rather than at a ``Stream``,
    because in a full-screen application there is no live region to hand over.
    One application owns every part of the screen, so nothing has to take the
    terminal from anything else -- which is what the two-``Live`` crash in the
    scrolling REPL came from.
    """
    from . import commands
    from .tui import Tui, _CaptureConsole

    target = console if console is not None else globals()["console"]
    view = SessionView(target, agent, mcp_manager)
    agent.on_usage = _make_usage_recorder(view, agent)

    if args is None:
        args = build_parser().parse_args([])
    session: Dict[str, Any] = {"stream": True, "verbose": args.verbose}

    def rebuild_tools():
        return build_registry(agent.config, args, mcp_manager, confirm=make_confirmer())

    # Commands write to ``context.console``. In the full-screen app that has to
    # be a capture, not stdout -- the application owns the terminal, and a
    # handler printing to it would land behind the UI and be invisible.
    capture = _CaptureConsole()
    context = commands.CommandContext(
        agent=agent,
        view=view,
        mcp_manager=mcp_manager,
        console=capture.console,
        args=args,
        session=session,
        rebuild_tools=rebuild_tools,
    )

    tui = Tui(agent, view, context, args, mcp_manager=mcp_manager)
    agent.tui = tui  # So /mode and other commands can reach the TUI
    tui.adopt_capture(capture)

    async def run_turn(prompt: str) -> None:
        """One turn, rendered into the transcript instead of the terminal."""
        tui.begin_answer()
        agent.on_event = _tui_reporter(tui, view, bool(session.get("verbose", False)))
        agent.on_text = _tui_printer(tui)

        try:
            answer = await agent.run(prompt, interactive=True)
        finally:
            tui.end_answer()

        # ``agent.answer_streamed`` reports whether a *callback* was installed,
        # not whether anything came through it -- a provider that returns the
        # whole answer in one non-streaming response sends no deltas, and
        # trusting that flag silently drops the answer off the screen.
        if answer and not tui.answer_streamed:
            tui.add_answer(answer)

    tui.run_turn = run_turn

    for warning in (mcp_manager.warnings if mcp_manager else []):
        tui.transcript.add_text(f"! {warning}", "warn")

    await tui.run_async()
    return 0


def _tui_reporter(tui, view: SessionView, verbose: bool):
    """``on_event`` for the TUI: writes into the transcript, refreshes chrome."""
    from .ui import render_tool_ansi

    def report(kind: str, text: str) -> None:
        if kind == "tool_call":
            name = text.split("(", 1)[0]
            arguments = text[len(name):]
            tui.add_tool(name, arguments)
        elif kind == "tool_result":
            if text.startswith("ERROR:"):
                tui.add_text(_clip(text, 400), "error")
            elif verbose:
                tui.add_ansi(render_tool_ansi(text))
        elif kind == "narration":
            tui.add_text(_clip(text, 300), "footer")
        elif kind == "reflection":
            tui.add_text("⟳ " + _clip(text, 200), "warn")
        tui.note_tool()

    return report


def _tui_printer(tui):
    """``on_text``: feed the streaming answer into the transcript."""

    async def print_text(piece: str) -> None:
        tui.feed_answer(piece)

    return print_text


async def run_interactive(
    agent: Agent,
    mcp_manager=None,
    stream: Optional["AnswerStream"] = None,
    args: Optional[argparse.Namespace] = None,
) -> int:
    """Run a REPL, keeping history across turns."""
    from . import commands

    if stream is None:
        stream = AnswerStream(console, enabled=False)
    if args is None:
        args = build_parser().parse_args([])

    view = SessionView(console, agent, mcp_manager)
    agent.on_usage = _make_usage_recorder(view, agent)
    agent.on_text = make_stream_printer(stream)

    # Runtime toggles that are not on Config. /set writes them, the loop reads
    # them, and nothing persists them -- they describe this session only.
    session: Dict[str, Any] = {"stream": stream.enabled, "verbose": args.verbose}

    def rebuild_tools() -> ToolRegistry:
        """Re-root the tools after ``/set cwd``, keeping the MCP ones."""
        return build_registry(agent.config, args, mcp_manager, confirm=make_confirmer())

    context = commands.CommandContext(
        agent=agent,
        view=view,
        mcp_manager=mcp_manager,
        console=console,
        args=args,
        session=session,
        rebuild_tools=rebuild_tools,
    )

    view.banner()
    if mcp_manager and mcp_manager.warnings:
        for warning in mcp_manager.warnings:
            console.print(f"[yellow]![/yellow] {warning}")

    while True:
        try:
            user_input = await read_input(console, commands)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            return 0

        if not user_input:
            continue

        # A slash command is a request to Agentux, not to the model. Nothing
        # typed after a leading slash ever reaches the provider.
        if user_input.startswith("/"):
            keep_going = await dispatch_command(context, user_input)
            if not keep_going:
                return 0
            continue

        # /set may have changed either of these since the last turn, so they
        # are read fresh every time rather than captured at startup.
        stream.set_enabled(bool(session.get("stream", stream.enabled)))
        status = StatusLine(view, console, enabled=stream.enabled)
        await _run_turn(
            agent,
            user_input,
            view,
            stream,
            status,
            verbose=bool(session.get("verbose", args.verbose)),
            echo_prompt=stream.enabled,
        )


async def read_input(console: Console, commands_module) -> str:
    """Read one line, with tab-completion for slash commands.

    Falls back to a plain prompt whenever the terminal cannot support a real
    readline editor -- a pipe, a dumb terminal, no prompt_toolkit -- so the
    REPL still works everywhere. Completion is a nicety; it must never be the
    reason the REPL stops working.
    """
    plain = f"\n[bold green]{PROMPT_MARK}[/bold green] "

    try:
        from prompt_toolkit import HTML, PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import InMemoryHistory
    except Exception:
        return console.input(plain).strip()

    class _SlashCompleter(Completer):
        """Completes command names and aliases, and nothing else.

        A bare ``/`` should not turn every word in the session into a
        suggestion, so completion is limited to text that starts with a slash.
        """

        def get_completions(inner_self, document, complete_event):  # noqa: N805
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            prefix = text[1:].split(" ")[0]
            start = -len(prefix)
            for name in commands_module.names():
                if name.startswith(prefix):
                    yield Completion("/" + name, start_position=start)
            for alias, target in commands_module.ALIASES.items():
                if alias.startswith(prefix):
                    yield Completion(
                        "/" + alias, start_position=start, display_meta=f"= /{target}"
                    )

    try:
        session = PromptSession(history=InMemoryHistory(), completer=_SlashCompleter())
        return session.prompt(
            HTML("<ansigreen><b>❯ </b></ansigreen>"), complete_while_typing=True
        ).strip()
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception:
        return console.input(plain).strip()


async def dispatch_command(context, text: str) -> bool:
    """Run one slash command. Returns ``False`` to end the session.

    An unknown ``/foo`` is reported rather than guessed at, and never silently
    forwarded to the model: a user who typed a slash expects a command, and
    sending it as a prompt produces a baffling answer about a slash.
    """
    from . import commands

    stripped = text.strip()
    if stripped == "/":
        # A bare slash is a request for the menu, not for the model.
        await commands.cmd_help(context, "")
        return True

    command = commands.resolve(stripped)
    if command is None:
        name = stripped.split()[0]
        console.print(
            f"[yellow]No command named {name!r}. Try[/yellow] [bold]/help[/bold]"
            f"[yellow], or drop the slash to talk to the agent.[/yellow]"
        )
        return True

    # Everything after the first word is the argument, whatever spelling of the
    # command was typed: "/model x", "/models x" and "/MODEL x" all pass "x".
    argument = " ".join(stripped.split()[1:]).strip()

    try:
        return await command.handler(context, argument)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return True
    except AgentuxError as exc:
        explain_error(exc)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        # A broken command must not end the session or lose the conversation.
        explain_error(exc)
        return True


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
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

    if args.list_tools:
        return asyncio.run(list_tools(config, args))

    # Streaming needs a terminal to stream into. Piped output gets the whole
    # answer at once, which is what a file or a pipe actually wants.
    enabled = console.is_terminal and not args.no_stream
    stream = AnswerStream(console, enabled=enabled)

    try:
        manager, warnings = asyncio.run(attach_mcp(config, args))
        try:
            if args.prompt:
                if getattr(args, "solo_engine", False):
                    return asyncio.run(
                        run_solo_engine(config, args, args.prompt, console=console)
                    )
                view = SessionView(console, None, manager)
                agent = build_agent(config, args, mcp_manager=manager, stream=stream, view=view)
                view.agent = agent  # the view needs the agent it is describing
                for warning in warnings:
                    console.print(f"[yellow]![/yellow] {warning}")
                return asyncio.run(run_once(agent, args.prompt, view=view, stream=stream))

            agent = build_agent(config, args, mcp_manager=manager, stream=stream)
            if console.is_terminal and not getattr(args, "no_tui", False):
                return asyncio.run(
                    run_tui(agent, mcp_manager=manager, args=args, console=console)
                )
            return asyncio.run(
                run_interactive(agent, mcp_manager=manager, stream=stream, args=args)
            )
        finally:
            asyncio.run(manager.close())
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return 130


def cli() -> None:
    """Console-script wrapper that propagates the exit code."""
    sys.exit(main())


def solo_main(argv: Optional[List[str]] = None) -> int:
    """Entry point for ``agentux-solo``: Agentux Solo, with no extra flags."""
    return main(["--solo", *(argv or [])])


async def run_solo_engine(
    config: Config,
    args: argparse.Namespace,
    prompt: str,
    *,
    console=None,
) -> int:
    """Run a prompt through the autonomous Agentux Solo research engine.

    The engine enforces the research cycle deterministically rather than
    trusting the model to remember to look things up. It prints each step as
    it happens so the user can see whether Solo is searching memory, doing
    web research, or saving what it learned.
    """
    from .knowledge import KnowledgeStore
    from .providers.openai_compatible import OpenAICompatibleProvider
    from .solo import solo_preset
    from .solo_engine import SoloEngine

    if console is None:
        console = globals().get("console") or Console()

    preset = solo_preset(config.working_dir)
    config.model = preset.model
    config.base_url = preset.base_url
    provider = OpenAICompatibleProvider(config)
    knowledge = KnowledgeStore(path=preset.memory_path)

    engine = SoloEngine(
        provider,
        knowledge,
        system_prompt=preset.system_prompt,
        project=config.working_dir,
    )

    def on_event(event):
        kind = event.kind
        msg = event.message
        if kind == "start":
            console.print(f"[bold #00ff88]⌋ Solo[/bold #00ff88] {msg}")
        elif kind == "memory_search":
            console.print(f"[dim]⌋ memory search[/dim] {msg}")
        elif kind == "memory_hit":
            console.print(f"[bold cyan]⌋ remembered[/bold cyan] {msg}")
        elif kind == "memory_miss":
            console.print(f"[dim]⌋ nothing remembered[/dim] {msg}")
        elif kind == "tool_call":
            console.print(f"[bold yellow]⌋ tool[/bold yellow] {msg}")
        elif kind == "tool_result":
            console.print(f"[dim]   ↳ {msg}[/dim]")
        elif kind == "nudge":
            console.print(f"[bold magenta]⌋ enforced[/bold magenta] {msg}")
        elif kind == "answer":
            console.print(f"\n[bold #00ff88]{msg}[/bold #00ff88]")
        elif kind == "done":
            console.print(f"[dim]⌋ {msg}[/dim]")
        elif kind == "error":
            console.print(f"[bold red]⌋ error[/bold red] {msg}")
        elif kind == "warning":
            console.print(f"[bold yellow]⌋ warning[/bold yellow] {msg}")

    console.print(f"[bold #00ff88]⌋ Agentux Solo engine[/bold #00ff88]  model={config.model}")
    result = await engine.run(prompt, on_event=on_event)
    knowledge.close()
    return 0 if result.answer else 1


if __name__ == "__main__":
    cli()
