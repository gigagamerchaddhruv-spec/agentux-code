"""Token and context accounting.

Two different numbers matter and they are not the same:

* **Usage** is what the provider reports it was billed for: prompt tokens plus
  completion tokens, summed across every request in a run.
* **Context** is how full the next request will be -- system prompt, tool
  schemas, and the whole conversation.

Providers report usage inconsistently, and local servers often report nothing,
so every figure here is labelled with where it came from. A number presented as
exact when it was guessed is worse than no number.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .models import Message, ToolDefinition

#: Rough characters-per-token ratio for English text and code. Used only when
#: the provider gives us nothing; good to within ~10% for prose, worse for
#: code, which is why estimates are never labelled as reported.
CHARS_PER_TOKEN = 4

#: Assumed context window when the provider does not advertise one. Deliberately
#: conservative so the percentage is not reassuring by default.
DEFAULT_CONTEXT_WINDOW = 128_000

#: Above this fraction of the window, the conversation is genuinely at risk.
CONTEXT_WARN_FRACTION = 0.75


def estimate_tokens(text: Optional[str]) -> int:
    """Approximate the token count of a string."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def usage_from_payload(raw: Any) -> Optional[Dict[str, int]]:
    """Read a usage object out of a provider response, if there is one.

    Handles both shapes in the wild: OpenAI's ``usage`` object, and Ollama's
    native ``prompt_eval_count`` / ``eval_count`` fields.
    """
    if not isinstance(raw, dict):
        return None

    usage = raw.get("usage")
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, int) or isinstance(completion, int):
            total = usage.get("total_tokens")
            return {
                "prompt_tokens": prompt if isinstance(prompt, int) else 0,
                "completion_tokens": completion if isinstance(completion, int) else 0,
                "total_tokens": total if isinstance(total, int) else 0,
            }

    prompt = raw.get("prompt_eval_count")
    completion = raw.get("eval_count")
    if isinstance(prompt, int) or isinstance(completion, int):
        return {
            "prompt_tokens": prompt if isinstance(prompt, int) else 0,
            "completion_tokens": completion if isinstance(completion, int) else 0,
            "total_tokens": (prompt if isinstance(prompt, int) else 0)
            + (completion if isinstance(completion, int) else 0),
        }

    return None


class Usage:
    """Running token totals for a session or a single run."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.requests = 0
        #: True when at least one provider response carried real numbers.
        self.reported = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, usage: Optional[Dict[str, int]], fallback_prompt: int = 0,
               fallback_completion: int = 0) -> None:
        """Add one request's usage, falling back to estimates when absent."""
        self.requests += 1
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.reported = True
        else:
            self.prompt_tokens += fallback_prompt
            self.completion_tokens += fallback_completion

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def summary(self) -> str:
        """A one-line, honest description of where these numbers came from."""
        source = "reported" if self.reported else "estimated"
        return (
            f"{self.total_tokens:,} tokens "
            f"({self.prompt_tokens:,} in / {self.completion_tokens:,} out, {source})"
        )

    def __repr__(self) -> str:
        return f"Usage(prompt={self.prompt_tokens}, completion={self.completion_tokens})"


def conversation_tokens(
    history: List[Message],
    system_prompt: str = "",
    tools: Optional[List[ToolDefinition]] = None,
) -> int:
    """Estimate the tokens the next request will carry.

    Covers the three things that actually get sent: the system prompt, the tool
    schemas, and every message. The tool schemas are easy to forget and are
    often a large fraction of the total.
    """
    total = estimate_tokens(system_prompt)

    for tool in tools or []:
        total += estimate_tokens(tool.name)
        total += estimate_tokens(tool.description)
        total += estimate_tokens(json.dumps(tool.parameters, default=str))

    for message in history:
        total += estimate_tokens(message.content)
        for call in message.tool_calls or []:
            total += estimate_tokens(call.name)
            total += estimate_tokens(json.dumps(call.arguments, default=str))

    return total


def format_tokens(count: int) -> str:
    """Compact token counts: 812, 12.4k, 1.2M."""
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.1f}M"


def format_duration(seconds: float) -> str:
    """Short, human durations: 240ms, 3.1s, 2m04s."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{remainder:04.1f}s".replace("m0", "m")
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m"


def context_bar(fraction: float, width: int = 20) -> str:
    """A fixed-width bar for context usage, coloured by how close it is.

    Returns a *markup string*. It must reach the screen through something that
    parses markup -- ``console.print(...)`` or ``Text.from_markup(...)`` -- never
    through ``Text.append``, which takes its argument as literal text and would
    show the ``[green]`` tags verbatim. :func:`agentux.ui.context_bar_text` is
    the safe way to get this into an existing ``Text``.
    """
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    bar = "█" * filled + "░" * (width - filled)
    if fraction >= CONTEXT_WARN_FRACTION:
        return f"[red]{bar}[/red]"
    if fraction >= 0.5:
        return f"[yellow]{bar}[/yellow]"
    return f"[green]{bar}[/green]"


__all__ = [
    "Usage",
    "usage_from_payload",
    "estimate_tokens",
    "conversation_tokens",
    "format_tokens",
    "format_duration",
    "context_bar",
    "CHARS_PER_TOKEN",
    "DEFAULT_CONTEXT_WINDOW",
]
