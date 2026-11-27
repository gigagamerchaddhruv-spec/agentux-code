"""Context management for long-running sessions.

A tool result is a liability. Reading a 400 KB file, grepping a vendored tree,
or running a test suite that prints ten thousand lines puts text into the
conversation that the model will never need again -- but it still has to be
re-sent on every subsequent request, and eventually the request does not fit.

So old tool output gets compressed, in place. Two rules make this safe:

* **Structure is never touched.** Every assistant message with ``tool_calls``
  keeps its following ``tool`` messages, in order, with the same
  ``tool_call_id``. OpenAI-compatible APIs reject a conversation whose calls
  are not answered in order, so dropping or reordering a message breaks the
  request -- this module only ever rewrites the *content* of messages.

* **Recent work stays verbatim.** The current request and the last few steps
  are what the model is actively reasoning about. Compressing those is how an
  agent convinces itself it already read something it didn't.

The compression itself is elision, not truncation: the head and tail of a long
output are what carry the signal (the command, the first error, the summary
line), so both are kept and the gap is marked with its size.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .models import Message

#: Tool results at or below this many characters are left alone. Below this,
#: elision costs more context than it saves.
DEFAULT_MAX_TOOL_CHARS = 8_000

#: How much of an oversized result to keep at each end.
DEFAULT_HEAD_CHARS = 2_500
DEFAULT_TAIL_CHARS = 2_500

#: Steps at the end of the conversation that are never compressed.
DEFAULT_KEEP_RECENT = 4


def elide(text: str, max_chars: int, head: int, tail: int) -> Tuple[str, bool]:
    """Shorten ``text`` to ``max_chars``, keeping its head and tail.

    Returns ``(text, changed)``. The elision marker names how much was dropped
    so the model knows it is looking at a summary rather than the whole thing,
    and can re-run a narrower query if it needs the middle.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False

    # Degenerate budgets would drop the head and tail we just promised to keep.
    budget = max(max_chars, head + tail + 120)
    head = min(head, budget // 2)
    tail = min(tail, budget - head - 120)

    dropped = len(text) - head - tail
    marker = f"\n\n... [{dropped} characters omitted from the middle] ...\n\n"
    return text[:head] + marker + text[-tail:], True


class ContextManager:
    """Compresses stale tool output in place, preserving conversation shape."""

    def __init__(
        self,
        max_tool_chars: int = DEFAULT_MAX_TOOL_CHARS,
        head_chars: int = DEFAULT_HEAD_CHARS,
        tail_chars: int = DEFAULT_TAIL_CHARS,
        keep_recent: int = DEFAULT_KEEP_RECENT,
    ) -> None:
        self.max_tool_chars = max_tool_chars
        self.head_chars = head_chars
        self.tail_chars = tail_chars
        self.keep_recent = max(0, keep_recent)
        self.compressed = 0
        self.saved_chars = 0

    # ------------------------------------------------------------------
    def _protect_from(self, history: List[Message]) -> int:
        """Index of the first message eligible for compression."""
        if not self.keep_recent:
            return 0

        # Walk back over whole steps: a step is an assistant message with tool
        # calls plus the tool messages answering it. Splitting one would leave
        # a tool result outside the protected window while its call is inside.
        index = len(history)
        seen_steps = 0
        while index > 0 and seen_steps < self.keep_recent:
            message = history[index - 1]
            if message.role == "user":
                break  # Never compress across the current request.
            if message.role == "tool":
                while index > 0 and history[index - 1].role == "tool":
                    index -= 1
            if index == 0:
                break
            index -= 1  # step over the assistant message
            seen_steps += 1
        return index

    def compress(self, history: List[Message]) -> List[Message]:
        """Compress stale tool results. Mutates and returns ``history``."""
        limit = self._protect_from(history)
        if limit <= 0:
            return history

        for message in history[:limit]:
            if message.role != "tool" or not message.content:
                continue
            if len(message.content) <= self.max_tool_chars:
                continue
            # Idempotent: a marker is already there, so leave it alone and
            # avoid nesting "[N characters omitted]" inside itself.
            if "characters omitted from the middle" in message.content:
                continue

            original_length = len(message.content)
            new_text, changed = elide(
                message.content, self.max_tool_chars, self.head_chars, self.tail_chars
            )
            if changed:
                message.content = new_text
                self.compressed += 1
                self.saved_chars += original_length - len(new_text)
        return history

    def stats(self) -> Dict[str, Any]:
        return {
            "compressed": self.compressed,
            "keep_recent": self.keep_recent,
            "max_tool_chars": self.max_tool_chars,
        }

    def reset_stats(self) -> None:
        self.compressed = 0
        self.saved_chars = 0


__all__ = ["ContextManager", "elide", "DEFAULT_MAX_TOOL_CHARS"]
