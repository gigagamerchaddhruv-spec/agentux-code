"""Terminal presentation.

Everything that decides how Agentux *looks* lives here rather than in the CLI,
so the same rendering can be reused by a test or another front end.

Three ideas carry the interface:

* **Gradients, sparingly.** A gradient on the wordmark and on nothing else.
  Colour used as decoration everywhere is noise, and it disappears entirely
  when the output is not a terminal. The wordmark's gradient runs down the
  rows and its extruded side is a second, darker colour -- the two together
  are what make the letters read as solid rather than as flat blocks.
* **One live region.** Tool activity scrolls permanently; the answer streams
  into a single live region that re-renders markdown as it arrives. Two
  competing live regions fight each other and flicker.
* **Numbers are labelled.** Tokens come from the provider when it reports them
  and from an estimate when it does not, and the display says which.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple, Union

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .theme import (
    ACCENT_DIM,
    BORDER,
    BORDER_SUBTLE,
    CYAN,
    EMERALD,
    PINK,
    PURPLE,
    TEXT_LIGHT,
    TEXT_MUTED,
    WORDMARK_DEPTH,
    WORDMARK_END,
    WORDMARK_SHADOW,
    WORDMARK_START,
    YELLOW,
)
from .usage import (
    DEFAULT_CONTEXT_WINDOW,
    Usage,
    context_bar,
    conversation_tokens,
    format_duration,
    format_tokens,
)

#: The accent ramp. Emerald -> cyan, from the brand palette.
ACCENT_START = WORDMARK_START
ACCENT_END = WORDMARK_END

#: The colour of the extruded side of each letter. Dim enough to read as depth
#: rather than as a second letter, bright enough to be visible at all.
ACCENT_DEPTH = WORDMARK_DEPTH

#: Letterforms, sharp-cornered, each exactly ``LETTER_WIDTH`` columns by
#: ``LETTER_HEIGHT`` rows. Declared as a grid rather than written out as one
#: banner because a hand-typed banner silently goes ragged -- and ragged rows
#: shear, which is what once made this read as "AOSNTUK" instead of AGENTUX.
#:
#: ``G`` carries a crossbar on its fourth row. Without it ``G`` is a closed
#: ring and reads as ``O``. ``U`` closes across both legs on its last row, so
#: the bowl is actually connected.
LETTER_HEIGHT = 6
LETTER_WIDTH = 8

_LETTERS: Dict[str, List[str]] = {
    "A": [
        "  ████  ",
        " ██  ██ ",
        "████████",
        "██    ██",
        "██    ██",
        "██    ██",
    ],
    "G": [
        " ██████ ",
        "██    ██",
        "██      ",
        "██ ████ ",
        "██    ██",
        " ██████ ",
    ],
    "E": [
        "████████",
        "██      ",
        "███████ ",
        "██      ",
        "██      ",
        "████████",
    ],
    "N": [
        "██    ██",
        "███   ██",
        "████  ██",
        "██ ██ ██",
        "██  ████",
        "██   ███",
    ],
    "T": [
        "████████",
        "  ████  ",
        "  ████  ",
        "  ████  ",
        "  ████  ",
        "  ████  ",
    ],
    "U": [
        "██    ██",
        "██    ██",
        "██    ██",
        "██    ██",
        "██    ██",
        " ██████ ",
    ],
    "X": [
        "██    ██",
        "██    ██",
        " ██  ██ ",
        "  ████  ",
        " ██  ██ ",
        "██    ██",
    ],
}

#: How far the extruded side is offset from the face, in (columns, rows).
#: One column and one row. Anything wider reaches across the gap between
#: neighbouring strokes and fills it in, which fuses the letters together --
#: the depth then reads as fill rather than as an edge.
DEPTH_X = 1
DEPTH_Y = 1


def _wordmark_layers() -> Tuple[List[str], List[str]]:
    """The wordmark as two aligned layers: the face and the extruded side.

    Returned separately rather than composited, because the two want different
    colours. Compositing first would mean re-deriving which cells are shadow,
    and the depth colour is most of what makes the letters read as solid.
    """
    face = [
        [" "] * (len(_LETTERS) * LETTER_WIDTH + (len(_LETTERS) - 1))
        for _ in range(LETTER_HEIGHT)
    ]
    for index, char in enumerate(_WORDMARK_TEXT):
        glyph = _LETTERS[char]
        offset = index * (LETTER_WIDTH + 1)
        for row in range(LETTER_HEIGHT):
            for col, pixel in enumerate(glyph[row]):
                face[row][offset + col] = pixel

    height, width = LETTER_HEIGHT, len(face[0])
    rows, cols = height + DEPTH_Y, width + DEPTH_X

    # The silhouette shifted down-right by (DEPTH_Y, DEPTH_X). Anything drawn
    # from this is "what the letters look like one step further back".
    shifted = [[" "] * cols for _ in range(rows)]
    for row in range(height):
        for col in range(width):
            shifted[row + DEPTH_Y][col + DEPTH_X] = face[row][col]

    def ink(grid: List[List[str]], row: int, col: int) -> bool:
        return 0 <= row < len(grid) and 0 <= col < len(grid[row]) and grid[row][col] != " "

    # Only the *rim* of the shifted silhouette is drawn. Two things are
    # subtracted, and both are load-bearing:
    #
    #   1. The face itself. Painting depth over the face smears the letters.
    #   2. The interior of the shifted silhouette -- the cells a letter's
    #      counter encloses, such as the triangle inside "A". Those are inside
    #      the solid, not on its edge, so filling them turns the counter into
    #      a filled wedge and the letter stops reading.
    #
    # What survives is the one-cell edge along the lower-right of each stroke,
    # which is the only part of an extruded face you would actually see.
    depth = [[" "] * cols for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            if not ink(shifted, row, col) or ink(face, row, col):
                continue
            enclosed = all(
                ink(shifted, row + dy, col + dx)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            if not enclosed:
                depth[row][col] = shifted[row][col]

    return (
        ["".join(row) for row in face],
        ["".join(row) for row in depth],
    )


#: The word, spelled once. `_LETTERS` is the only place the shapes live.
_WORDMARK_TEXT = "AGENTUX"


def interpolate(start: Tuple[int, int, int], end: Tuple[int, int, int], t: float) -> str:
    """A colour between two RGB points, as a rich hex string."""
    t = max(0.0, min(1.0, t))
    return "#{:02x}{:02x}{:02x}".format(
        round(start[0] + (end[0] - start[0]) * t),
        round(start[1] + (end[1] - start[1]) * t),
        round(start[2] + (end[2] - start[2]) * t),
    )


def gradient_text(text: str, start=ACCENT_START, end=ACCENT_END) -> Text:
    """Paint a string with a horizontal colour ramp.

    The ramp runs across the whole string, not per line, so a multi-line
    wordmark reads as one gradient rather than several restarting ones.
    """
    total = len(text)
    if total == 0:
        return Text("")

    lines = text.split("\n")
    styled = Text()
    seen = 0
    for line_index, line in enumerate(lines):
        if line_index:
            styled.append("\n")
        for char in line:
            styled.append(char, style=interpolate(start, end, seen / max(1, total - 1)))
            seen += 1
        seen += 1  # the newline counts toward the ramp
    return styled


def gradient_text_vertical(
    text: str, start=ACCENT_START, end=ACCENT_END
) -> Text:
    """Paint a string with a vertical colour ramp, one colour per line.

    The wordmark needs this rather than a horizontal ramp. Its glyphs are solid
    block characters, so ramping across each line colours adjacent columns of
    the same letter differently -- the letter reads as vertical stripes instead
    of one shape. Ramping down the lines instead gives each row a single colour,
    which keeps every letter solid and still reads as one gradient overall.
    """
    lines = text.split("\n")
    if not lines:
        return Text("")

    styled = Text()
    last = max(1, len(lines) - 1)
    for index, line in enumerate(lines):
        if index:
            styled.append("\n")
        if not line:
            continue
        # Half a step in: the first row fully at `start` and the last fully at
        # `end` both look washed out next to their neighbours.
        styled.append(line, style=interpolate(start, end, (index + 0.5) / (last + 1)))
    return styled


def wordmark() -> Text:
    """The Agentux wordmark: a gradient face with an extruded dark side.

    The two layers are composited onto one grid before any colour is applied,
    because a cell can hold only one glyph. Drawing the depth row and then the
    face row as separate strings would interleave them wrongly -- the rows are
    different lengths, so the face would land in the wrong columns and smear
    sideways across the banner.

    Each row is padded to the full grid width and never stripped: the trailing
    blanks are what keep the letters from shearing against each other.
    """
    face, depth = _wordmark_layers()
    width = max(len(row) for row in depth)

    # Composite: start from the depth, then stamp the face over it.
    grid = [list(row.ljust(width)) for row in depth]
    for index, row in enumerate(face):
        target = grid[index]
        for col, char in enumerate(row):
            if char != " ":
                target[col] = char
    # Which cells are face and which are depth, after compositing. The depth
    # layer is taller than the face (that is the extrusion), so the extra rows
    # have no face to mark and are depth throughout.
    face_mask = [
        [
            index < len(face) and col < len(face[index]) and face[index][col] != " "
            for col in range(width)
        ]
        for index in range(len(grid))
    ]

    last = max(1, len(grid) - 1)
    styled = Text()
    for index in range(len(grid)):
        if index:
            styled.append("\n")
        shade = interpolate(ACCENT_START, ACCENT_END, (index + 0.5) / (last + 1))
        side = interpolate(ACCENT_DEPTH, WORDMARK_SHADOW, index / last)
        for col in range(width):
            char = grid[index][col]
            if char == " ":
                styled.append(" ")
            elif face_mask[index][col]:
                styled.append(char, style=shade)
            else:
                styled.append(char, style=side)

    return styled


#: The face layer only, for tests and for anything that wants plain text.
WORDMARK = "\n".join(_wordmark_layers()[0])


def context_bar_text(fraction: float, width: int = 16) -> Text:
    """The context bar as a styled ``Text``.

    ``usage.context_bar`` returns markup, which is only safe to print with
    markup parsing on. Callers here build a ``Text`` with ``append``, and
    ``Text.append`` treats its argument literally -- so the tags would appear
    on screen as ``[green]...[/green]``. Converting here keeps the style.
    """
    return Text.from_markup(context_bar(fraction, width))


# ----------------------------------------------------------------------
# Rendering into a buffer instead of onto the terminal
# ----------------------------------------------------------------------
# The full-screen TUI owns the terminal, so rich must not print to stdout.
# These render to an in-memory console and return the ANSI text, which the TUI
# hands to prompt_toolkit as styled text. Rich stays the only renderer; it just
# writes somewhere that is not the screen.

def render_markdown_ansi(text: str, width: int = 80) -> str:
    """Render markdown to ANSI text without touching stdout."""
    return _render_ansi(Markdown(text) if text.strip() else Text(""), width)


def render_text_ansi(text: str, width: int = 80) -> str:
    """Render plain text to ANSI, with no markup interpretation."""
    return _render_ansi(Text(text), width)


def render_tool_ansi(text: str, width: int = 100) -> str:
    """Render a verbose tool result as an indented, dimmed block."""
    body = Text()
    for index, line in enumerate(text.rstrip("\n").split("\n")):
        if index:
            body.append("\n")
        body.append("    " + line, style="dim")
    return _render_ansi(body, width)


def _render_ansi(renderable: RenderableType, width: int) -> str:
    from io import StringIO

    console = Console(
        file=StringIO(),
        width=max(20, width),
        force_terminal=True,
        color_system="truecolor",
        legacy_windows=False,
        highlight=False,
        soft_wrap=False,
    )
    console.print(renderable)
    # `print` terminates its last line. Callers embed this in a larger
    # document, so the extra newline shows up as a stray blank row there.
    raw = console.file.getvalue().rstrip("\n")  # type: ignore[union-attr]
    return "\n".join(line.rstrip() for line in raw.split("\n"))


def wordmark_to_ansi(rows: Optional[int] = None) -> str:
    """The wordmark as ANSI text, optionally only its first N rows.

    Truncating has to keep the spans. The gradient lives in them, so slicing
    the plain string and re-rendering would give a flat grey block instead of
    the ramp.
    """
    # Split the Text itself, not its plain form: the spans carry the gradient,
    # and re-joining plain strings would flatten every row to one colour.
    lines = wordmark().split("\n", allow_blank=True)
    # The rendered Text keeps its trailing newline, which would show up as a
    # stray blank line inside the transcript.
    while lines and not lines[-1].plain.strip():
        lines.pop()
    if rows is not None:
        lines = lines[:rows]

    trimmed = Text()
    for index, line in enumerate(lines):
        if index:
            trimmed.append("\n")
        trimmed.append_text(line)
    return _render_ansi(trimmed, 200)


# ----------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------
class StreamRenderer:
    """Renders an answer into one live region, markdown and all.

    Markdown cannot be appended to incrementally -- a half-typed code fence is
    not the same document as a closed one -- so the accumulated text is
    re-rendered on each delta. For answer-sized text that is cheap, and it buys
    correctly formatted output the instant it is complete.
    """

    def __init__(self, console: Console, markdown: bool = True, transient: bool = False) -> None:
        self.console = console
        self.markdown = markdown
        self.buffer: List[str] = []
        self._live: Optional[Live] = None
        self.started = time.monotonic()
        self.elapsed: float = 0.0

    def _renderable(self) -> RenderableType:
        text = "".join(self.buffer)
        if not text:
            return Text("")
        if self.markdown:
            return Markdown(text)
        return Text(text)

    def start(self) -> None:
        if self._live is not None:
            return
        self.started = time.monotonic()
        self._live = Live(
            self._renderable(),
            console=self.console,
            refresh_per_second=12,
            transient=False,
            vertical_overflow="visible",
        )
        self._live.start()

    def feed(self, piece: str) -> None:
        """Append a delta and repaint."""
        self.buffer.append(piece)
        if self._live is not None:
            self._live.update(self._renderable())

    def stop(self) -> str:
        """Finish the region, leaving the rendered answer on screen."""
        self.elapsed = time.monotonic() - self.started
        if self._live is not None:
            self._live.update(self._renderable())
            self._live.stop()
            self._live = None
        return "".join(self.buffer)


# ----------------------------------------------------------------------
# Session chrome
# ----------------------------------------------------------------------
class SessionView:
    """Renders the header, tool activity, status, and footer for one session."""

    def __init__(self, console: Console, agent: Any, mcp_manager: Any = None) -> None:
        self.console = console
        self.agent = agent
        self.mcp_manager = mcp_manager
        self.started = time.monotonic()
        self.usage = Usage()
        self._spinner_cycle = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    # -- header ---------------------------------------------------------
    def banner(self) -> None:
        agent = self.agent
        mcp_tools = len(self.mcp_manager.tools) if self.mcp_manager else 0
        built_in = len(agent.registry) - mcp_tools
        servers = len(self.mcp_manager.servers) if self.mcp_manager else 0

        tools_line = f"{built_in} built-in"
        if servers:
            tools_line += f"  ·  {mcp_tools} from {servers} MCP server{'s' if servers != 1 else ''}"

        stats = self.stats_line()

        body = Group(
            wordmark(),
            Text(""),
            Text.from_markup(f"[dim]v{agentux_version()}[/dim]"),
            Text(""),
            _rule(),
            _field("model", agent.provider.get_model_name()),
            _field("workspace", agent.config.working_dir),
            _field("tools", tools_line),
            _field("context", stats),
            Text(""),
            _rule(),
            Text.from_markup(
                "[dim]/help[/dim] [dim]/model[/dim] [dim]/set[/dim] "
                "[dim]/tools[/dim] [dim]/mcp[/dim] [dim]/context[/dim] "
                "[dim]/reset[/dim] [dim]/exit[/dim]"
            ),
        )

        self.console.print()
        self.console.print(
            Panel(body, border_style=ACCENT_START_HEX, padding=(1, 2), title=None)
        )
        self.console.print()

    # -- activity -------------------------------------------------------
    def narration(self, text: str) -> None:
        self.console.print(f"  [dim italic]{_shorten(text, 160)}[/dim italic]")

    def tool_call(self, text: str) -> None:
        name = text.split("(", 1)[0]
        arguments = text[len(name):]
        self.console.print(
            Text("  ⏺ ", style=ACCENT_START_HEX)
            + Text(name, style=f"bold {ACCENT_START_HEX}")
            + Text(_shorten(arguments, 100), style="dim")
        )

    def tool_error(self, text: str) -> None:
        self.console.print(f"  [red]✗[/red] [red]{_shorten(text, 300)}[/red]")

    def tool_output(self, text: str, verbose: bool) -> None:
        if not verbose:
            return
        self.console.print(
            Panel(_shorten(text, 6000), title="result", border_style="dim", padding=(0, 1))
        )

    def reflection(self, text: str) -> None:
        self.console.print(f"  [yellow]⟳[/yellow] [dim yellow]{_shorten(text, 120)}[/dim yellow]")

    # -- status ---------------------------------------------------------
    def status(self) -> str:
        """The line shown while the agent is working.

        Deliberately one line: a status that reflows the terminal is a status
        the user has to wait for. Everything here is short enough to stay put.
        """
        frame = self._spinner_cycle[int(time.monotonic() * 10) % len(self._spinner_cycle)]
        elapsed = time.monotonic() - self.started
        tools = len(self.agent.step_log)
        parts = [
            f"[{ACCENT_START_HEX}]{frame}[/{ACCENT_START_HEX}] [bold]working[/bold]",
            f"[dim]{_format(elapsed)}[/dim]",
            f"[dim]{tools} tool{'s' if tools != 1 else ''}[/dim]",
        ]
        tokens = self.usage.total_tokens
        if tokens:
            # A leading ~ marks a figure the provider did not report.
            source = "" if self.usage.reported else "~"
            parts.append(f"[dim]{source}{format_tokens(tokens)} tok[/dim]")
        return "  " + "  [dim]·[/dim]  ".join(parts)

    def stats_line(self) -> Text:
        """Context occupancy, for the banner and ``/context``."""
        return self.context_stats()

    def context_stats(self) -> Text:
        """The occupancy bar and its numbers, as a ``Text``.

        A ``Text`` and not a markup string: every caller appends this into an
        existing ``Text``, and ``Text.append`` does not parse markup, so a
        returned ``"[green]...[/green]"`` would be shown to the user verbatim.
        """
        history = getattr(self.agent, "history", [])
        system = history[0].content if history and history[0].role == "system" else ""
        tools = self.agent.registry.definitions()
        used = conversation_tokens(history, system or "", tools)
        window = DEFAULT_CONTEXT_WINDOW
        fraction = min(1.0, used / window) if window else 0.0

        text = Text()
        text.append_text(context_bar_text(fraction, 16))
        text.append(f"  {fraction * 100:.0f}%", style="dim")
        text.append(f"  {format_tokens(used)} / {format_tokens(window)}", style="dim")
        return text

    # -- footer ---------------------------------------------------------
    def summary(self, steps: int) -> None:
        elapsed = time.monotonic() - self.started
        pieces = [
            f"{len(self.agent.step_log)} tool call{'s' if len(self.agent.step_log) != 1 else ''}",
            f"{steps} step{'s' if steps != 1 else ''}",
            _format(elapsed),
        ]
        if self.usage.total_tokens:
            pieces.append(f"{'~' if not self.usage.reported else ''}"
                          f"{format_tokens(self.usage.total_tokens)} tokens")
        failures = sum(1 for entry in self.agent.step_log if entry[2])
        if failures:
            pieces.append(f"{failures} failed")

        self.console.print()
        self.console.print(
            Text("  " + " · ".join(pieces), style="dim")
        )
        self.console.print(_rule())
        self.started = time.monotonic()


# ----------------------------------------------------------------------
ACCENT_START_HEX = interpolate(ACCENT_START, ACCENT_END, 0.15)
ACCENT_END_HEX = interpolate(ACCENT_START, ACCENT_END, 0.85)


def _rule() -> Text:
    return Text("─" * 60, style="dim")


def _field(label: str, value: Union[str, Text]) -> Text:
    """One ``label   value`` row of the banner.

    A ``Text`` value keeps its own styles. The context row passes one so the
    bar can stay green/yellow/red instead of being flattened to cyan, and so
    its markup tags are never shown literally.
    """
    text = Text()
    text.append(f"{label:<11}", style="dim")
    if isinstance(value, Text):
        text.append_text(value.copy())
    else:
        text.append(value, style="cyan")
    return text


def _shorten(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _format(seconds: float) -> str:
    return format_duration(seconds)


def agentux_version() -> str:
    from . import __version__

    return __version__


__all__ = [
    "StreamRenderer",
    "SessionView",
    "gradient_text",
    "wordmark",
    "interpolate",
    "ACCENT_START",
    "ACCENT_END",
]
