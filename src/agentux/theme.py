"""The Agentux palette, taken from the brand at kidux.surge.sh.

One module owns every colour the interface uses, so the terminal and the rich
renderers cannot drift apart. Both :mod:`agentux.ui` and :mod:`agentux.tui`
read from here.

The brand is a dark, retro-hardware palette: near-black grounds, emerald and
cyan carrying the interface, yellow for warnings, purple and pink for accents
and failures. The values below are the brand's own, except the borders, which
the brand defines as translucent white. A terminal has no alpha channel, so
those are composited over ``--bg-pure`` here -- 14% white over ``#030508`` is
``#26282b``, and 8% is ``#17191c``. Deriving them beats guessing a grey: the
ratio between border and ground is the brand's, not ours.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple

# -- grounds ---------------------------------------------------------------
#: The darkest ground - full black for Opencode-like appearance
BG_PURE = "#000000"
#: The raised panel ground, one step up from the page.
BG_SURFACE = "#000000"

# -- accents ---------------------------------------------------------------
#: The primary. The prompt, the brand mark, anything that means "you".
EMERALD = "#00ff88"
#: The secondary. Tool activity and other machine-side events.
CYAN = "#00f0ff"
#: Warnings: recoverable, but the user should look.
YELLOW = "#ffd60a"
#: Secondary accent, for values that need to stand out without alarming.
PURPLE = "#bf5af2"
#: Failures. Named for the brand token even though it is a red.
PINK = "#ff3b30"

# -- text ------------------------------------------------------------------
#: Body copy.
TEXT_LIGHT = "#f4f7fc"
#: Supporting copy: labels, footers, secondary lines.
TEXT_MUTED = "#94a0b3"
#: The quietest text that should still be legible.
TEXT_DIM = "#546073"

# -- borders ---------------------------------------------------------------
#: ``rgba(255, 255, 255, 0.14)`` over :data:`BG_PURE`. Panel and frame edges.
BORDER = "#26282b"
#: ``rgba(255, 255, 255, 0.08)`` over :data:`BG_PURE`. Rules and dividers
#: that should be felt rather than read.
BORDER_SUBTLE = "#17191c"

# -- gradients -------------------------------------------------------------
#: The wordmark ramps emerald -> cyan down the rows.
#:
#: Vertical, not horizontal. The glyphs are solid block characters, so ramping
#: across each row colours neighbouring columns of the same letter differently
#: and the letter reads as vertical stripes instead of one shape.
WORDMARK_START: Tuple[int, int, int] = (0, 255, 136)
WORDMARK_END: Tuple[int, int, int] = (0, 240, 255)
#: The extruded side of each letter. A deep green, so the depth belongs to the
#: same object as the face rather than reading as a separate grey slab.
WORDMARK_DEPTH: Tuple[int, int, int] = (12, 74, 56)
#: What the extrusion fades toward at the bottom: the page ground.
WORDMARK_SHADOW: Tuple[int, int, int] = (3, 5, 8)


#: Operation modes for Agentux.
class Mode(Enum):
    """Operation modes for Agentux."""
    CHAT = "chat"      # Normal conversational mode
    BUILD = "build"    # Focused on building/creating
    PLAN = "plan"      # Focused on planning/designing


def _mix(first: Tuple[int, int, int], second: Tuple[int, int, int], t: float) -> str:
    """A hex colour between two RGB points."""
    t = max(0.0, min(1.0, t))
    return "#{:02x}{:02x}{:02x}".format(
        round(first[0] + (second[0] - first[0]) * t),
        round(first[1] + (second[1] - first[1]) * t),
        round(first[2] + (second[2] - first[2]) * t),
    )


#: A dimmed emerald, for chrome that must recede. Interpolating between the
#: accent and the ground keeps the hue, so dimmed text still reads as *ours*.
ACCENT_DIM = _mix(WORDMARK_START, (3, 5, 8), 0.45)

#: Accent gradient start for interactive elements (used in commands.py)
ACCENT_START_HEX = "#{:02x}{:02x}{:02x}".format(*WORDMARK_START)
#: Accent gradient end for interactive elements (used in commands.py)
ACCENT_END_HEX = "#{:02x}{:02x}{:02x}".format(*WORDMARK_END)

#: The prompt_toolkit class names the TUI styles fragments against, and the
#: hex each one resolves to. Kept beside the palette so adding an accent means
#: editing one list.
STYLE_CLASSES: Dict[str, str] = {
    "frame": BORDER,
    "frame-focus": EMERALD,
    "accent": EMERALD,
    "accent-dim": ACCENT_DIM,
    "brand": EMERALD,
    "label": TEXT_MUTED,
    "value": TEXT_LIGHT,
    "tool": CYAN,
    "tool-arg": TEXT_MUTED,
    "error": PINK,
    "warn": YELLOW,
    "ok": EMERALD,
    "prompt": EMERALD,
    "body": TEXT_LIGHT,
    "footer": TEXT_DIM,
    "rule": BORDER_SUBTLE,
    "scrollbar": BORDER_SUBTLE,
}


__all__ = [
    "BG_PURE",
    "BG_SURFACE",
    "EMERALD",
    "CYAN",
    "YELLOW",
    "PURPLE",
    "PINK",
    "TEXT_LIGHT",
    "TEXT_MUTED",
    "TEXT_DIM",
    "BORDER",
    "BORDER_SUBTLE",
    "ACCENT_DIM",
    "ACCENT_START_HEX",
    "ACCENT_END_HEX",
    "WORDMARK_START",
    "WORDMARK_END",
    "WORDMARK_DEPTH",
    "WORDMARK_SHADOW",
    "STYLE_CLASSES",
    "Mode"
]


__all__ = [
    "BG_PURE",
    "BG_SURFACE",
    "EMERALD",
    "CYAN",
    "YELLOW",
    "PURPLE",
    "PINK",
    "TEXT_LIGHT",
    "TEXT_MUTED",
    "TEXT_DIM",
    "BORDER",
    "BORDER_SUBTLE",
    "ACCENT_DIM",
    "ACCENT_START_HEX",
    "ACCENT_END_HEX",
    "WORDMARK_START",
    "WORDMARK_END",
    "WORDMARK_DEPTH",
    "WORDMARK_SHADOW",
    "STYLE_CLASSES",
    "Mode"
]