"""The interactive terminal interface.

A full-screen application rather than a scrolling transcript. It owns the whole
terminal while it runs: a header with the session's identity, a scrolling body
that holds the conversation, a footer showing what the agent is doing right now,
and an input line at the bottom.

Two decisions shape the code here.

**Rich renders, prompt_toolkit displays.** Rich is very good at markdown and
tables and already a dependency, so text is still rendered through it. But rich
owns the terminal when it prints, which would fight the full-screen app for
control. So rich renders into an in-memory console whose output is captured as
ANSI and handed to prompt_toolkit as styled text. Rich never touches stdout.

**One live region, always.** The scrolling REPL had to hand the terminal back
and forth between a status line and a streaming answer -- the source of the
``LiveError`` that killed runs mid-answer. Here there is nothing to hand over:
the body, the footer and the input are all part of one prompt_toolkit
application that redraws itself. Every part updates in place.

The layout is::

    ┌ agentux  ~/agentux  ·  model ────────────────────────┐
    │  ░░░░░░░░░░░░░░░░░░░░░  1%  1.5k / 128.0k              │   header
    ├──────────────────────────────────────────────────────┤
    │  ████    ██████  ████████ ██    ██ ████████ ...      │
    │   ██████  ████████ ████████ ...                      │   welcome
    │  ████████ ███    ██ ████████ ██ ...                  │   (scrolls)
    │                                                          │
    │  ❯ what does this module do                          │   body
    │    ⏺ filesystem({"operation": "read", ...})           │
    │      It parses CLI arguments ...                      │
    │  ❯ now add a --verbose flag                           │
    │    ⠋ working · 2.1s · 1 tool · ~1.4k tok             │   footer
    └──────────────────────────────────────────────────────┘
      ❯ _

The header is deliberately two rows -- brand, workspace, model, then the
context bar. The wordmark scrolls with the transcript rather than sitting in
a fixed pane, where it would be decoration the user can never scroll past.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import textwrap
import time
from typing import Any, Callable, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style

from . import commands as commands_module
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
    STYLE_CLASSES,
    TEXT_DIM,
    TEXT_LIGHT,
    TEXT_MUTED,
    YELLOW,
)
from .ui import interpolate, render_markdown_ansi, wordmark_to_ansi

#: The accent hex used for the input marker and focused chrome.
ACCENT = EMERALD

#: How often the footer repaints while the agent works. Fast enough to feel
#: live, slow enough that a spinner does not consume a core.
FOOTER_REFRESH_HZ = 8

#: Marks the start of the input region in the styled-text buffer.
PROMPT_MARK = "❯"

STYLE = Style.from_dict(
    {
        **STYLE_CLASSES,
        # Emphasis, not a colour of its own: the brand's bold white.
        "bold": f"bold {TEXT_LIGHT}",
    }
)


class TuiLexer(Lexer):
    """Colours the input line: commands, quoted text, and paths.

    Highlight only. The fragments must concatenate back to the original line
    exactly -- a lexer that drops or reorders characters corrupts what the user
    typed, which is worse than no colour at all.
    """

    @staticmethod
    def _lex_arguments(rest: str) -> List[Tuple[str, str]]:
        """Split the remainder into fragments, preserving every space."""
        fragments: List[Tuple[str, str]] = []
        for index, chunk in enumerate(rest.split(" ")):
            if index:
                fragments.append(("", " "))
            if not chunk:
                continue
            if chunk[0] in "\"'":
                fragments.append(("class:ok", chunk))
            elif "/" in chunk and "." in chunk:
                fragments.append(("class:tool-arg", chunk))
            else:
                fragments.append(("", chunk))
        return fragments

    def lex_document(self, document: Document) -> Callable[[int], List[Tuple[str, str]]]:
        lines = document.lines

        def get_line(line_number: int) -> List[Tuple[str, str]]:
            # prompt_toolkit asks about the line just past the end while
            # measuring; an empty line is a normal answer, not an error.
            if line_number >= len(lines):
                return [("", "")]
            line = lines[line_number]
            stripped = line.lstrip()
            if not stripped:
                return [("", "")]

            if stripped.startswith("/"):
                # partition drops the separator, so take the first word and
                # slice the rest explicitly. Losing that space would render
                # "/set model" as "/setmodel" -- the user would be reading
                # back text they never typed.
                head = line.split(" ", 1)[0]
                remainder = line[len(head):]
                fragments: List[Tuple[str, str]] = [("class:prompt", head)]
                if remainder:
                    fragments.extend(self._lex_arguments(remainder))
                return fragments
            if stripped[0] in "#-*":
                return [("class:warn", line)]
            return [("", line)]

        return get_line


def _word_wrap(text: str, width: int) -> str:
    """Wrap text cleanly on whitespace word boundaries, preserving existing indentation and newlines."""
    if width <= 10 or not text:
        return text
    result_lines: List[str] = []
    for line in text.split("\n"):
        if not line:
            result_lines.append("")
            continue
        stripped = line.lstrip(" ")
        indent_len = len(line) - len(stripped)
        indent = " " * indent_len
        if len(line) <= width:
            result_lines.append(line)
            continue
        wrapped = textwrap.wrap(
            stripped,
            width=max(10, width - indent_len),
            initial_indent=indent,
            subsequent_indent=indent,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if wrapped:
            result_lines.extend(wrapped)
        else:
            result_lines.append(line)
    return "\n".join(result_lines)


class Transcript:
    """The conversation so far, as styled text prompt_toolkit can draw.

    Entries are kept as structured records rather than one growing string so a
    live streaming answer can be re-rendered from its accumulated text without
    re-rendering everything above it.
    """

    def __init__(self) -> None:
        self._entries: List[Tuple[str, Any]] = []
        #: Index of the entry currently streaming, if any. Rendering replaces
        #: this entry in place rather than appending.
        self._streaming: Optional[int] = None

    # -- appending ------------------------------------------------------
    def add_user(self, text: str) -> None:
        self._close_stream()
        self._entries.append(("user", text))

    def add_tool(self, name: str, arguments: str) -> None:
        self._close_stream()
        self._entries.append(("tool", (name, arguments)))

    def add_assistant(self, text: str) -> None:
        self._close_stream()
        self._entries.append(("assistant", text))

    def add_text(self, text: str, style: str = "") -> None:
        self._close_stream()
        self._entries.append(("text", (text, style)))

    def add_ansi(self, ansi: str) -> None:
        """Append output already rendered by rich, keeping its colours."""
        self._close_stream()
        self._entries.append(("ansi", ansi))

    def begin_stream(self) -> None:
        self._close_stream()
        self._streaming = len(self._entries)
        self._entries.append(("stream", ""))

    def feed_stream(self, text: str) -> None:
        if self._streaming is None:
            self.begin_stream()
        kind, existing = self._entries[self._streaming]
        self._entries[self._streaming] = (kind, existing + text)

    def _close_stream(self) -> None:
        self._streaming = None

    def clear(self) -> None:
        self._entries = []
        self._streaming = None

    @property
    def empty(self) -> bool:
        return not self._entries

    def renderable(self, width: int, tui: 'Tui') -> Any:
        """Build the styled fragment list for the body window with clean word wrapping."""
        from prompt_toolkit.formatted_text import ANSI, to_formatted_text

        content_width = max(20, width - 2)
        card_width = min(max(24, width - 2), 92)
        inner_width = max(18, card_width - 4)

        fragments: List[Tuple[str, str]] = []
        for kind, payload in self._entries:
            if kind == "user":
                mode = getattr(tui, "mode", Mode.CHAT)
                mode_color_class = {
                    Mode.CHAT: "class:brand",
                    Mode.BUILD: "class:tool",
                    Mode.PLAN: "class:accent"
                }.get(mode, "class:brand")

                u_bar = "─" * max(4, card_width - 8)
                fragments.append(("class:prompt", f"{PROMPT_MARK} "))
                fragments.append((mode_color_class, "You "))
                fragments.append(("class:rule", u_bar))
                fragments.append(("", "\n"))

                wrapped_user = _word_wrap(payload, inner_width)
                for line in wrapped_user.split("\n"):
                    fragments.append(("class:rule", "│ "))
                    fragments.append(("class:bold", line.rstrip()))
                    fragments.append(("", "\n"))

                fragments.append(("class:rule", "╰─" + "─" * max(0, card_width - 2)))
                fragments.append(("", "\n\n"))

            elif kind in ("stream", "assistant"):
                model_name = ""
                if tui and getattr(tui, "agent", None):
                    model_name = getattr(tui.agent.config, "model", "")
                model_tag = f"· {model_name} " if model_name else ""
                a_bar = "─" * max(4, card_width - 12 - len(model_tag))

                fragments.append(("class:tool", "◆ Agentux "))
                if model_tag:
                    fragments.append(("class:footer", model_tag))
                fragments.append(("class:rule", a_bar))
                fragments.append(("", "\n"))

                if not payload or not str(payload).strip():
                    fragments.append(("class:rule", "│ "))
                    fragments.append(("class:footer", "…"))
                    fragments.append(("", "\n"))
                else:
                    ansi = render_markdown_ansi(payload, inner_width)
                    for line in ansi.split("\n"):
                        cleaned = line.rstrip()
                        fragments.append(("class:rule", "│ "))
                        if cleaned:
                            fragments.extend(to_formatted_text(ANSI(cleaned)))
                        fragments.append(("", "\n"))

                fragments.append(("class:rule", "╰─" + "─" * max(0, card_width - 2)))
                fragments.append(("", "\n\n"))

            elif kind == "tool":
                name, arguments = payload
                if name == "delegate":
                    tool_title = "🤖 Subagent Delegation "
                    rule_len = max(4, card_width - len(tool_title) - 2)
                    fragments.append(("class:brand", tool_title))
                    fragments.append(("class:rule", "─" * rule_len))
                    fragments.append(("", "\n"))
                elif name == "checkpoint":
                    tool_title = "⎌ Checkpoint "
                    rule_len = max(4, card_width - len(tool_title) - 2)
                    fragments.append(("class:ok", tool_title))
                    fragments.append(("class:rule", "─" * rule_len))
                    fragments.append(("", "\n"))
                else:
                    tool_title = f"⚙ {name} "
                    rule_len = max(4, card_width - len(tool_title) - 2)
                    fragments.append(("class:tool", tool_title))
                    fragments.append(("class:rule", "─" * rule_len))
                    fragments.append(("", "\n"))

                wrapped_args = _word_wrap(arguments.strip(), inner_width)
                for line in wrapped_args.split("\n"):
                    fragments.append(("class:rule", "│ "))
                    fragments.append(("class:tool-arg", line.rstrip()))
                    fragments.append(("", "\n"))

                fragments.append(("class:rule", "╰─" + "─" * max(0, card_width - 2)))
                fragments.append(("", "\n\n"))

            elif kind == "text":
                text, style = payload
                if style == "error":
                    e_bar = "─" * max(4, card_width - 10)
                    fragments.append(("class:error", "⚠ Error "))
                    fragments.append(("class:rule", e_bar))
                    fragments.append(("", "\n"))

                    wrapped_err = _word_wrap(text, inner_width)
                    for line in wrapped_err.split("\n"):
                        fragments.append(("class:error", "│ "))
                        fragments.append(("class:error", line.rstrip()))
                        fragments.append(("", "\n"))

                    fragments.append(("class:error", "╰─" + "─" * max(0, card_width - 2)))
                    fragments.append(("", "\n\n"))
                elif style == "warn":
                    w_bar = "─" * max(4, card_width - 12)
                    fragments.append(("class:warn", "! Warning "))
                    fragments.append(("class:rule", w_bar))
                    fragments.append(("", "\n"))

                    wrapped_warn = _word_wrap(text, inner_width)
                    for line in wrapped_warn.split("\n"):
                        fragments.append(("class:warn", "│ "))
                        fragments.append(("class:warn", line.rstrip()))
                        fragments.append(("", "\n"))

                    fragments.append(("class:warn", "╰─" + "─" * max(0, card_width - 2)))
                    fragments.append(("", "\n\n"))
                else:
                    wrapped_text = _word_wrap(text, content_width)
                    fragments.append((f"class:{style}" if style else "", wrapped_text))
                    fragments.append(("", "\n"))

            elif kind == "ansi":
                fragments.extend(to_formatted_text(ANSI(payload)))
                fragments.append(("", "\n"))

        return fragments


class _CaptureConsole:
    """A ``Console`` that keeps what it is given instead of printing it.

    Every command handler writes to ``context.console``. In the full-screen app
    that console must not reach stdout -- the application owns the terminal --
    so it is pointed here and drained into the transcript after each command.
    This is what lets every existing handler work unmodified inside the TUI.
    """

    def __init__(self, width: int = 100) -> None:
        from rich.console import Console

        self._buffer: List[str] = []
        self._console = Console(
            file=self,
            width=width,
            force_terminal=True,
            color_system="truecolor",
            legacy_windows=False,
            highlight=False,
        )

    # The Console writes through these two methods.
    def write(self, text: str) -> int:
        self._buffer.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def drain(self) -> str:
        text = "".join(self._buffer)
        self._buffer = []
        return text

    @property
    def console(self):
        return self._console

    def set_width(self, width: int) -> None:
        try:
            self._console.width = width
        except Exception:
            pass


class StatusState:
    """What the footer shows, and the timer behind it."""

    def __init__(self, view: Any) -> None:
        self.view = view
        self.active = False
        self.started = 0.0
        self.label = ""
        self.tool_name = ""
        self._tick: Optional[asyncio.Task] = None
        self._on_change: Optional[Callable[[], None]] = None

    def begin(self, on_change: Callable[[], None]) -> None:
        self.active = True
        self.started = time.monotonic()
        self._on_change = on_change
        self._tick = asyncio.ensure_future(self._run())
        self._emit()

    def note(self, tool_name: str = "") -> None:
        self.tool_name = tool_name or self.tool_name
        self._emit()

    def end(self) -> None:
        self.active = False
        self.tool_name = ""
        if self._tick is not None:
            self._tick.cancel()
            self._tick = None
        self._emit()

    async def _run(self) -> None:
        """Repaint the footer while work is happening.

        A task rather than a rich ``Live``: rich would take the terminal, and
        the terminal belongs to the application.
        """
        try:
            while self.active:
                self._emit()
                await asyncio.sleep(1.0 / FOOTER_REFRESH_HZ)
        except asyncio.CancelledError:
            pass

    def _emit(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:
                pass  # A redraw failure must never end the run.

    def fragments(self) -> List[Tuple[str, str]]:
        if not self.active:
            return []
        spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        frame = spinner[int(time.monotonic() * 10) % len(spinner)]
        elapsed = time.monotonic() - self.started
        usage = getattr(self.view, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) if usage else 0

        from .usage import format_duration, format_tokens

        pieces = [
            ("class:accent", frame),
            ("", " "),
            ("class:bold", "working"),
            ("class:footer", f"  {format_duration(elapsed)}"),
        ]
        tools = len(getattr(self.view.agent, "step_log", []) or [])
        if tools:
            pieces.append(("class:footer", f"  {tools} tool{'s' if tools != 1 else ''}"))
        if tokens:
            reported = getattr(usage, "reported", False)
            mark = "" if reported else "~"
            pieces.append(("class:footer", f"  {mark}{format_tokens(tokens)} tok"))
        if self.tool_name:
            pieces.append(("class:tool-arg", f"  {self.tool_name}"))
        return pieces


class Tui:
    """The full-screen session."""

    def __init__(
        self,
        agent: Any,
        view: Any,
        context: Any,
        args: Any,
        mcp_manager: Any = None,
        run_turn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.agent = agent
        self.view = view
        self.context = context
        self.args = args
        self.mcp_manager = mcp_manager
        self.run_turn = run_turn
        self.mode = Mode.CHAT  # Default mode

        self.transcript = Transcript()
        self.status = StatusState(view)
        self._capture = _CaptureConsole()
        self.input_buffer = Buffer(multiline=False, complete_while_typing=True)
        self.history: List[str] = []
        self._history_index = 0
        self._should_exit = False
        self._busy = False
        self._streamed = False
        self._app: Optional[Application] = None
        self._pane = _Pane(self)

    # ------------------------------------------------------------------
    # chrome
    # ------------------------------------------------------------------
    def header_fragments(self) -> List[Tuple[str, str]]:
        """The fixed top bar: who you are talking to, and where.

        Deliberately two lines. The ASCII banner belongs to the welcome below,
        which scrolls away -- a six-row wordmark parked above a scrolling
        transcript would spend a sixth of the screen on decoration forever.
        """
        model = self.agent.provider.get_model_name()
        workspace = _shorten_path(self.agent.config.working_dir or "")
        stats = self.view.context_stats()

        # Mode-specific styling
        mode = getattr(self, 'mode', Mode.CHAT)
        mode_styles = {
            Mode.CHAT: "class:brand",
            Mode.BUILD: "class:tool",
            Mode.PLAN: "class:accent"
        }
        mode_colors = {
            Mode.CHAT: EMERALD,
            Mode.BUILD: CYAN,
            Mode.PLAN: PURPLE
        }
        mode_icons = {
            Mode.CHAT: "►",
            Mode.BUILD: "⚡",
            Mode.PLAN: "📝"
        }

        fragments: List[Tuple[str, str]] = [
            ("class:brand", f"{PROMPT_MARK} agentux"),
            (mode_styles[mode], f" {mode_icons[mode]} {mode.value.upper()} "),
            ("class:label", f"  Skills: {len(self.agent.skills.list_skills())}/{len(self.agent.skills.list_available_skills())} "),
            ("class:label", f"  {_ellipsis(workspace)}  ·  {model}\n"),
        ]
        fragments.append(("class:label", "  "))
        fragments.extend(self._plain(stats))
        return fragments

    @staticmethod
    def _plain(text: Any) -> List[Tuple[str, str]]:
        """Flatten a rich ``Text`` into prompt_toolkit styled tuples.

        A rich ``Span`` is a (start, end, style) slice of the plain string; it
        carries no text of its own, so the characters have to be sliced out.
        """
        out: List[Tuple[str, str]] = []
        plain = getattr(text, "plain", "")
        for span in getattr(text, "spans", []) or []:
            out.append((_style_name(span.style), plain[span.start:span.end]))
        if not out and plain:
            out.append(("", plain))
        return out

    def footer_fragments(self) -> List[Tuple[str, str]]:
        active = self.status.fragments()
        if active:
            return [("class:footer", "  ")] + active

        usage = getattr(self.view, "usage", None)
        parts: List[Tuple[str, str]] = []
        steps = getattr(self.agent, "last_steps", 0)
        if steps:
            parts.append(("class:footer", f"{steps} step{'s' if steps != 1 else ''}"))
        if usage and getattr(usage, "total_tokens", 0):
            mark = "" if getattr(usage, "reported", False) else "~"
            from .usage import format_tokens

            parts.append(("class:footer", f"{mark}{format_tokens(usage.total_tokens)} tokens"))

        # Add skills information
        skills_loaded = len(self.agent.skills.list_skills())
        skills_total = len(self.agent.skills.list_available_skills())
        if skills_total > 0:
            parts.append(("class:footer", f" Skills: {skills_loaded}/{skills_total} "))
        if not parts:
            return []

        # Add mode indicator to footer
        mode_indicators = {
            Mode.CHAT: [("class:brand", " CHAT ")],
            Mode.BUILD: [("class:tool", " BUILD ")],
            Mode.PLAN: [("class:accent", " PLAN ")]
        }

        return [("class:footer", "  ")] + parts + [("class:footer", "  ")] + mode_indicators[self.mode]

    # ------------------------------------------------------------------
    # frames
    # ------------------------------------------------------------------
    # Rounded, not square. The brand's panels have generous radii, and a
    # square box next to a rounded wordmark reads as two different programs.
    # These are all box-drawing characters, which are East Asian Ambiguous
    # width: narrow in a normal terminal, two cells wide in CJK mode.
    TOP_LEFT, TOP_RIGHT = "╭", "╮"
    BOTTOM_LEFT, BOTTOM_RIGHT = "╰", "╯"
    HORIZONTAL, VERTICAL = "─", "│"

    def _rule(self, width: int) -> List[Tuple[str, str]]:
        return [("class:rule", self.HORIZONTAL * max(0, width))]

    @property
    def _focused(self) -> bool:
        """Whether the input holds focus, which tints the box emerald.

        Checked through the buffer's own control rather than a flag: focus can
        move without the flag changing, and a box that lies about being
        focused is worse than no box.
        """
        control = getattr(self.input_buffer, "control", None)
        if control is None:
            return False
        try:
            return bool(control.is_focused())
        except Exception:
            return False

    @property
    def _frame_class(self) -> str:
        # Mode-specific frame styling
        mode_styles = {
            Mode.CHAT: "class:frame-focus",
            Mode.BUILD: "class:frame-focus tool",
            Mode.PLAN: "class:frame-focus accent"
        }
        base_class = "class:frame-focus" if self._focused else "class:frame"
        mode_class = mode_styles.get(self.mode, "class:frame-focus")
        return f"{base_class} {mode_class}" if self._focused else mode_class

    def header_rule(self) -> List[Tuple[str, str]]:
        return self._rule(self._pane._width())

    def footer_rule(self) -> List[Tuple[str, str]]:
        return self._rule(self._pane._width())

    def input_top(self) -> List[Tuple[str, str]]:
        width = self._pane._width()
        return [
            (
                self._frame_class,
                f"{self.TOP_LEFT}{self.HORIZONTAL * max(0, width - 2)}{self.TOP_RIGHT}",
            )
        ]

    def input_bottom(self) -> List[Tuple[str, str]]:
        width = self._pane._width()
        return [
            (
                self._frame_class,
                f"{self.BOTTOM_LEFT}{self.HORIZONTAL * max(0, width - 2)}{self.BOTTOM_RIGHT}",
            )
        ]

    def input_left(self) -> List[Tuple[str, str]]:
        return [(self._frame_class, self.VERTICAL + " " + PROMPT_MARK + " ")]

    def input_right(self) -> List[Tuple[str, str]]:
        return [(self._frame_class, " " + self.VERTICAL)]

    # ------------------------------------------------------------------
    # input handling
    # ------------------------------------------------------------------
    def on_submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.history.append(text)
        self._history_index = len(self.history)

        self.transcript.add_user(text)
        if text.startswith("/"):
            # A command is ours, not the model's. Nothing typed after a leading
            # slash is ever sent to a provider.
            asyncio.ensure_future(self._run_command(text))
            self._pane.scroll_to_end()
            self._redraw()
            return

        if self.run_turn is None:
            self.transcript.add_text("(no runner wired)", "error")
            self._pane.scroll_to_end()
            self._redraw()
            return

        asyncio.ensure_future(self._run_turn_tui(text))
        self._pane.scroll_to_end()
        self._redraw()

    async def _run_command(self, text: str) -> None:
        stripped = text.strip()
        if stripped == "/":
            await commands_module.cmd_help(self.context, "")
        else:
            command = commands_module.resolve(stripped)
            if command is None:
                name = stripped.lstrip("/").split()[0]
                self.transcript.add_text(
                    f"No command named '/{name}'. Try /help.", "warn"
                )
            else:
                argument = " ".join(stripped.split()[1:]).strip()
                try:
                    keep_going = await command.handler(self.context, argument)
                except Exception as exc:  # A broken command must not end the session.
                    self.transcript.add_text(f"{type(exc).__name__}: {exc}", "error")
                    keep_going = True
                if not keep_going:
                    self._should_exit = True
                    if self._app is not None:
                        self._app.exit()
        self._drain_console()
        self._pane.scroll_to_end()
        self._redraw()

    async def _run_turn_tui(self, text: str) -> None:
        self._busy = True
        self.input_buffer.text = ""
        self.status.begin(self._redraw)
        self._redraw()
        try:
            result = self.run_turn(text)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            self.transcript.add_text(f"{type(exc).__name__}: {exc}", "error")
        finally:
            self.status.end()
            self._busy = False
            self._pane.scroll_to_end()
            self._redraw()

    # ------------------------------------------------------------------
    # console capture
    # ------------------------------------------------------------------
    def adopt_capture(self, capture: "_CaptureConsole") -> None:
        """Use the console the command context was built with."""
        self._capture = capture

    def _drain_console(self) -> None:
        """Move anything the command printed into the transcript."""
        try:
            self._capture.set_width(max(20, self._pane._width() - 2))
        except Exception:
            pass
        captured = self._capture.drain()
        if captured:
            self.transcript.add_ansi(captured)

    def _redraw(self) -> None:
        if self._app is not None:
            try:
                self._app.invalidate()
            except Exception:
                pass

    def welcome(self) -> None:
        """The banner, in the transcript so it scrolls away.

        The full wordmark lives here rather than in the fixed header: this is
        decoration you read once, and the header is not. Model, workspace and
        context are in that header, so they are not repeated -- two copies of
        the same line on one screen is a layout bug, not emphasis.
        """
        agent = self.agent
        mcp_tools = len(self.mcp_manager.tools) if self.mcp_manager else 0
        built_in = len(agent.registry.names) - mcp_tools
        servers = len(self.mcp_manager.servers) if self.mcp_manager else 0

        tools = f"{built_in} built-in"
        if servers:
            tools += f", {mcp_tools} from {servers} MCP server{'s' if servers != 1 else ''}"

        self.transcript.add_ansi(wordmark_to_ansi())
        self.transcript.add_text("")

        # Mode-specific welcome message
        mode_messages = {
            Mode.CHAT: "  Normal conversational mode",
            Mode.BUILD: "  Focused on building/creating",
            Mode.PLAN: "  Focused on planning/designing"
        }

        mode_colors = {
            Mode.CHAT: "footer",
            Mode.BUILD: "tool",
            Mode.PLAN: "accent"
        }

        self.transcript.add_text(mode_messages[self.mode], mode_colors[self.mode])
        self.transcript.add_text("")
        self.transcript.add_text(f"  {tools}", "footer")
        self.transcript.add_text("")
        self.transcript.add_text(
            "  /help for commands · PgUp/PgDn, Shift+↑/↓ or wheel scrolls · Ctrl-L clears · Ctrl-C interrupts · Ctrl-D quits",
            "footer",
        )
        self.transcript.add_text("")

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def run_async(self) -> None:
        """Take over the terminal until the user leaves.

        ``run_async`` and not ``run``: ``Application.run()`` calls
        ``asyncio.run()`` itself, which raises inside the loop that
        :func:`agentux.cli.run_tui` is already running on. Awaiting the
        coroutine is the supported way to embed the application.
        """
        self.welcome()
        self._pane = _Pane(self)
        app: Application = Application(
            layout=self._build_layout(),
            key_bindings=self._build_bindings(),
            style=STYLE,
            full_screen=True,
            mouse_support=True,
            refresh_interval=None,
        )
        self._app = app
        try:
            await app.run_async()
        finally:
            self._should_exit = True

    def run(self) -> None:
        """Synchronous entry point, for tests and scripts."""
        asyncio.run(self.run_async())

    def _build_layout(self) -> Layout:
        """The screen, as four framed regions.

        Borders are drawn as text rather than as container edges because
        prompt_toolkit has no border primitive: a `Window` renders only its
        control's fragments. So each region supplies its own top and bottom
        rule, sized to the width it was actually given, and the left and right
        edges of the input frame are separate one-column windows.

        The three fixed regions cost a row each for their rules, which is the
        price of the frame. The alternative -- bare text on the background --
        is what made this look unfinished: there was nothing to separate the
        header from the transcript from the prompt.
        """
        return Layout(
            HSplit(
                [
                    # Header, closed with a rule.
                    Window(
                        FormattedTextControl(self.header_fragments),
                        height=Dimension(min=2, max=2),
                        wrap_lines=False,
                    ),
                    Window(
                        FormattedTextControl(self.header_rule),
                        height=1,
                        wrap_lines=False,
                    ),
                    # The scrolling transcript, unbounded.
                    self._pane.attach(
                        Window(
                            FormattedTextControl(self._pane.fragments),
                            wrap_lines=True,
                            always_hide_cursor=True,
                        )
                    ),
                    # Status, then the rule that separates it from the prompt.
                    Window(
                        FormattedTextControl(self.footer_fragments),
                        height=Dimension(min=1, max=1),
                        wrap_lines=False,
                    ),
                    Window(
                        FormattedTextControl(self.footer_rule),
                        height=1,
                        wrap_lines=False,
                    ),
                    # The input, boxed and focused-highlighted. A `VSplit` puts
                    # the two edges beside the buffer rather than under it.
                    # Nesting an `HSplit` here instead would make the container
                    # itself the current control, and `Application.reset` then
                    # calls `is_focusable()` on an `HSplit`, which has no such
                    # method -- the app dies before it draws.
                    Window(
                        FormattedTextControl(self.input_top),
                        height=1,
                        wrap_lines=False,
                    ),
                    VSplit(
                        [
                            Window(
                                FormattedTextControl(self.input_left),
                                width=4,
                                wrap_lines=False,
                            ),
                            Window(
                                BufferControl(
                                    self.input_buffer, lexer=TuiLexer()
                                ),
                                height=Dimension(min=1, max=1),
                                wrap_lines=False,
                            ),
                            Window(
                                FormattedTextControl(self.input_right),
                                width=2,
                                wrap_lines=False,
                            ),
                        ]
                    ),
                    Window(
                        FormattedTextControl(self.input_bottom),
                        height=1,
                        wrap_lines=False,
                    ),
                ]
            )
        )

    def _build_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter", eager=True)
        def _submit(event: Any) -> None:
            buffer = event.current_buffer
            text = buffer.text
            buffer.reset()
            self.on_submit(text)

        @bindings.add("c-c")
        def _interrupt(event: Any) -> None:
            if self._busy:
                # Do not end the run: the user may want to retype after it.
                event.current_buffer.text = ""
                return
            event.app.exit()

        @bindings.add("c-d")
        def _eof(event: Any) -> None:
            if not event.current_buffer.text:
                event.app.exit()

        @bindings.add("c-l")
        def _clear(event: Any) -> None:
            self.transcript.clear()
            self.welcome()
            event.app.invalidate()

        # Scrolling keybindings
        @bindings.add("pageup")
        @bindings.add("s-pageup")
        def _page_up(event: Any) -> None:
            self._pane.scroll_page_up()
            self._redraw()

        @bindings.add("pagedown")
        @bindings.add("s-pagedown")
        def _page_down(event: Any) -> None:
            self._pane.scroll_page_down()
            self._redraw()

        @bindings.add("s-up")
        @bindings.add("c-up")
        def _scroll_up(event: Any) -> None:
            self._pane.scroll_up(lines=4)
            self._redraw()

        @bindings.add("s-down")
        @bindings.add("c-down")
        def _scroll_down(event: Any) -> None:
            self._pane.scroll_down(lines=4)
            self._redraw()

        @bindings.add("<scroll-up>")
        def _wheel_up(event: Any) -> None:
            self._pane.scroll_up(lines=3)
            self._redraw()

        @bindings.add("<scroll-down>")
        def _wheel_down(event: Any) -> None:
            self._pane.scroll_down(lines=3)
            self._redraw()

        @bindings.add("c-home")
        def _scroll_top(event: Any) -> None:
            self._pane.scroll_to_top()
            self._redraw()

        @bindings.add("c-end")
        def _scroll_bottom(event: Any) -> None:
            self._pane.scroll_to_end()
            self._redraw()

        @bindings.add("up", filter=_not_busy(self))
        def _history_prev(event: Any) -> None:
            if not self.history:
                self._pane.scroll_up(lines=2)
                self._redraw()
                return
            if event.current_buffer.text == "" and self._history_index == 0:
                self._pane.scroll_up(lines=2)
                self._redraw()
                return
            self._history_index = max(0, self._history_index - 1)
            event.current_buffer.text = self.history[self._history_index]
            event.current_buffer.cursor_position = len(event.current_buffer.text)

        @bindings.add("down", filter=_not_busy(self))
        def _history_next(event: Any) -> None:
            if not self.history:
                self._pane.scroll_down(lines=2)
                self._redraw()
                return
            if event.current_buffer.text == "" and self._history_index >= len(self.history):
                self._pane.scroll_down(lines=2)
                self._redraw()
                return
            self._history_index = min(len(self.history), self._history_index + 1)
            if self._history_index >= len(self.history):
                event.current_buffer.text = ""
            else:
                event.current_buffer.text = self.history[self._history_index]
            event.current_buffer.cursor_position = len(event.current_buffer.text)

        return bindings

    # -- callbacks used by the runner ----------------------------------
    def begin_answer(self) -> None:
        self._streamed = False
        self.transcript.begin_stream()
        self._pane.scroll_to_end()
        self._redraw()

    def feed_answer(self, piece: str) -> None:
        if piece:
            self._streamed = True
        self.transcript.feed_stream(piece)
        self._pane.scroll_to_end()
        self._redraw()

    def note_tool(self, name: str = "") -> None:
        self.status.note(name)
        self._redraw()

    def end_answer(self) -> None:
        self.transcript._close_stream()
        self._pane.scroll_to_end()
        self._redraw()

    @property
    def answer_streamed(self) -> bool:
        """Whether any text actually reached the screen.

        Not whether a callback was installed: a provider that replies in one
        non-streaming response sends no deltas at all, and treating "a callback
        exists" as "the answer was shown" loses the answer.
        """
        return self._streamed

    def add_answer(self, answer: str) -> None:
        """Append a whole answer that was never streamed."""
        self.transcript.add_assistant(answer)
        self._pane.scroll_to_end()
        self._redraw()

    def add_tool(self, name: str, arguments: str) -> None:
        """One tool call, as it appears in the transcript."""
        self.transcript.add_tool(name, arguments)
        self._pane.scroll_to_end()
        self._redraw()

    def add_text(self, text: str, style: str = "") -> None:
        self.transcript.add_text(text, style)
        self._pane.scroll_to_end()
        self._redraw()

    def add_ansi(self, ansi: str) -> None:
        self.transcript.add_ansi(ansi)
        self._pane.scroll_to_end()
        self._redraw()


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace and truncate on a word boundary."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut) + "…"


def _ellipsis(text: str, limit: int = 44) -> str:
    """Shorten from the left, keeping the end that identifies a path.

    A workspace path is recognised by its tail -- ``.../agentux`` tells you
    where you are, where ``/Users/macb...`` does not.
    """
    if len(text) <= limit:
        return text
    return "…" + text[-(limit - 1):]


def _shorten_path(path: str) -> str:
    """``/Users/macbook/agentux`` -> ``~/agentux``."""
    home = os.path.expanduser("~")
    if home and home != "/" and path.startswith(home):
        return "~" + path[len(home):]
    return path


def _not_busy(tui: "Tui") -> bool:
    return not tui._busy


def _style_name(style: Optional[str]) -> str:
    """Map a rich style string onto one of our class names.

    Rich styles are words like ``"bold red"`` or ``"dim"``, not our class
    names, so this has to translate both a colour word and an intensity word.
    Anything unrecognised falls through unstyled rather than guessed at.
    """
    if not style:
        return ""
    words = set(style.lower().replace(" ", "").split(","))

    if "red" in style.lower():
        return "class:error"
    if "yellow" in style.lower() or "orange" in style.lower():
        return "class:warn"
    if "green" in style.lower():
        return "class:ok"
    if "bold" in words:
        return "class:bold"
    if "dim" in words:
        return "class:label"
    return ""


class _Pane:
    """The scrolling body, with a scrollbar and sticky-bottom behaviour.

    Sticky bottom matters: while an answer streams, new lines arrive faster
    than the user reads. Following the tail is right almost always, but not
    while they have scrolled up to read something -- so an explicit scroll away
    from the bottom is respected until they come back.
    """

    def __init__(self, tui: Tui) -> None:
        self.tui = tui
        self.stick = True
        self.scroll_offset: Optional[int] = None
        self._window: Optional[Window] = None

    def fragments(self) -> List[Tuple[str, str]]:
        width = self._width()
        rendered = self.tui.transcript.renderable(width, self.tui)
        if not rendered:
            return [("class:footer", "  Type a task, or /help for commands.")]
        return rendered

    def _width(self) -> int:
        if self._window is not None:
            try:
                info = self._window.render_info
                if info is not None and info.window_width > 20:
                    return info.window_width
            except Exception:
                pass
        try:
            return max(20, shutil.get_terminal_size().columns - 4)
        except Exception:
            return 80

    def attach(self, window: "Window") -> "Window":
        """Remember the window, so its rendered width is knowable.

        Returns it so this can be called inline where the window is being
        built: ``Layout(HSplit([pane.attach(Window(...)), ...]))``.
        """
        self._window = window
        # prompt_toolkit's Window re-derives its scroll from the cursor
        # position on every render. A FormattedTextControl has no cursor, so
        # the scroll silently resets to 0 and the newest lines vanish off
        # the bottom. ``get_vertical_scroll`` overrides that derivation and
        # is what actually keeps us pinned to the tail.
        window.get_vertical_scroll = lambda win: self._desired_scroll(win)
        return window

    def _desired_scroll(self, window: "Window") -> int:
        """The scroll offset that puts the last rendered line at the bottom, or current manual offset."""
        try:
            info = window.render_info
            if info is None:
                return self.scroll_offset if self.scroll_offset is not None else 0
            height = info.window_height
            content_lines = getattr(info, "ui_content", None)
            if content_lines is None:
                return self.scroll_offset if self.scroll_offset is not None else 0
            total = content_lines.line_count
            max_scroll = max(0, total - height)

            if self.stick or self.scroll_offset is None:
                self.scroll_offset = max_scroll
                return max_scroll

            clamped = max(0, min(max_scroll, self.scroll_offset))
            if clamped >= max_scroll:
                self.stick = True
                self.scroll_offset = max_scroll
            return clamped
        except Exception:
            return self.scroll_offset if self.scroll_offset is not None else 0

    def scroll_up(self, lines: int = 4) -> None:
        if self._window is None:
            return
        try:
            info = self._window.render_info
            if info is not None:
                height = info.window_height
                content_lines = getattr(info, "ui_content", None)
                total = content_lines.line_count if content_lines else 100
                max_scroll = max(0, total - height)
            else:
                max_scroll = 1000

            current = max_scroll if (self.stick or self.scroll_offset is None) else self.scroll_offset
            self.stick = False
            self.scroll_offset = max(0, current - lines)
            self._window.vertical_scroll = self.scroll_offset
            self._window.invalidate()
        except Exception:
            pass

    def scroll_down(self, lines: int = 4) -> None:
        if self._window is None:
            return
        try:
            info = self._window.render_info
            if info is not None:
                height = info.window_height
                content_lines = getattr(info, "ui_content", None)
                total = content_lines.line_count if content_lines else 100
                max_scroll = max(0, total - height)
            else:
                max_scroll = 1000

            current = max_scroll if (self.stick or self.scroll_offset is None) else self.scroll_offset
            new_scroll = current + lines
            if new_scroll >= max_scroll:
                self.stick = True
                self.scroll_offset = max_scroll
                self._window.vertical_scroll = max_scroll
            else:
                self.scroll_offset = new_scroll
                self._window.vertical_scroll = new_scroll
            self._window.invalidate()
        except Exception:
            pass

    def scroll_page_up(self) -> None:
        height = 10
        if self._window and self._window.render_info:
            height = max(4, self._window.render_info.window_height - 2)
        self.scroll_up(height)

    def scroll_page_down(self) -> None:
        height = 10
        if self._window and self._window.render_info:
            height = max(4, self._window.render_info.window_height - 2)
        self.scroll_down(height)

    def scroll_to_top(self) -> None:
        self.stick = False
        self.scroll_offset = 0
        if self._window:
            self._window.vertical_scroll = 0
            self._window.invalidate()

    def scroll_to_end(self) -> None:
        self.stick = True
        self.scroll_offset = None
        if self._window is None:
            return
        try:
            self._window.vertical_scroll = 10**9
            self._window.invalidate()
        except Exception:
            pass


__all__ = ["Tui", "Transcript", "PROMPT_MARK"]