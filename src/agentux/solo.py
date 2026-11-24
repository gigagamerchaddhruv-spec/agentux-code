"""Agentux Solo: the prompt and preset that make a small model an agent.

Solo is qwen2.5:0.5b plus the knowledge store in :mod:`agentux.memory`. The
model is not expected to know things; it is expected to find out and to
remember. This module holds the one source of truth for that behaviour: the
system prompt, and the preset that points a session at the right model, store
and recall count.

The prompt here is deliberately plain and repetitive. A 0.5B model does not
infer intent from a elegant paragraph; it follows instructions it has been
given explicitly. The Ollama Modelfile is generated from this constant, so
there is no second copy to drift.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .agent import Agent

#: The model Solo is built around, and where it lives.
SOLO_MODEL = "qwen2.5:0.5b"
SOLO_BASE_URL = "http://localhost:11434/v1"

#: How many memories to inject automatically each turn.
SOLO_INJECT = 3

SOLO_SYSTEM_PROMPT = """\
You are Agentux Solo, a small model that runs inside agentux. You are not \
expected to know everything. You are expected to find out, and to remember.

You have a persistent memory. Things you learn are kept, and you get better \
the more you use them.

## When you do not know, or are not sure

Say so, then find out. Do not guess and do not invent.

- Search your memory with memory_search before answering something you might \
already know.
- Look at the project with the search and filesystem tools before answering \
from assumption.
- Search the web with web_search and read the page with fetch_web_page when \
the answer is not in the project and you are not sure.

## When you learn something

Write it down with memory_save, one fact per call. Save anything non-obvious: \
a gotcha, a fact about this project, a convention, a command that works, a \
URL, a correction the user gave you. Be specific enough that the fact is still \
useful months from now.

Use scope "project" unless the lesson is genuinely general, in which case use \
scope "global". Most things belong to the project they were learned in.

## How to answer

- Lead with the answer.
- Keep it short. The user is reading a terminal.
- If you checked something, say what you checked.
- If you could not find out, say so plainly rather than guessing.
"""


@dataclass
class SoloPreset:
    """Everything a session needs to become Solo."""

    model: str
    base_url: str
    system_prompt: str
    memory_path: str
    inject: int


#: Question words that make a prompt look like it wants information rather
#: than action.
QUESTION_WORDS = (
    "what", "who", "whom", "whose", "when", "where", "why", "how", "which",
    "is", "are", "do", "does", "did", "can", "could", "should", "would",
    "will", "explain", "tell", "name", "list",
)

#: Short exchanges that are not questions about the world.
_SMALL_TALK = frozenset(
    {
        "how are you", "how's it going", "who are you", "what are you",
        "what is your name", "hello", "hi", "hey", "thanks", "thank you",
        "good morning", "good evening", "what can you do", "help",
    }
)

#: Greetings that go on to add a question mark and a few extra words.
_SMALL_TALK_PREFIXES = (
    "how are you", "how's it going", "who are you", "what are you",
    "what is your name", "what can you do",
)


def looks_like_question(prompt: str) -> bool:
    """Whether a prompt is asking for information the store may not hold.

    Solo uses this to decide when to look something up. It is deliberately a
    heuristic rather than a judgement, and it errs towards searching, because
    a wrong answer costs more than an unnecessary search.

    A question mark or a leading question word is enough on its own -- a 3
    word question like "Who discovered gravity?" is still a factual question
    and still deserves a lookup, while "hi" and "thanks" are not.
    """
    text = " ".join(str(prompt).strip().lower().split())
    if not text or text in _SMALL_TALK:
        return False
    if any(text.startswith(prefix) for prefix in _SMALL_TALK_PREFIXES):
        return False
    if text.endswith("?"):
        return True
    return text.split(None, 1)[0] in QUESTION_WORDS


def solo_preset(working_dir: str, memory_path: Optional[str] = None) -> SoloPreset:
    """Build the Solo preset for a workspace.

    The store defaults to ``~/.agentux/memory.db`` so that memories survive
    across projects and sessions; pass ``memory_path`` to keep one per
    workspace instead, or set ``AGENTUX_SOLO_MEMORY`` to point every session at
    a different store.
    """
    if memory_path is None:
        memory_path = os.getenv("AGENTUX_SOLO_MEMORY") or os.path.join(
            os.path.expanduser("~"), ".agentux", "memory.db"
        )
    return SoloPreset(
        model=SOLO_MODEL,
        base_url=SOLO_BASE_URL,
        system_prompt=SOLO_SYSTEM_PROMPT,
        memory_path=os.path.abspath(os.path.expanduser(memory_path)),
        inject=SOLO_INJECT,
    )


def apply_solo(agent: "Agent", preset: Optional[SoloPreset] = None) -> None:
    """Turn an existing agent into an Agentux Solo session.

    Sets the model and base URL, swaps in the Solo prompt, and gives the agent
    its store plus the two memory tools. The provider reads the model and base
    URL from its config on every request, so no rebuild is needed.
    """
    preset = preset or solo_preset(agent.config.working_dir)

    agent.config.model = preset.model
    agent.config.base_url = preset.base_url
    agent._system_prompt = preset.system_prompt

    from .memory import MemoryStore
    from .tools.memory_tool import MemorySaveTool, MemorySearchTool

    project = agent.config.working_dir or os.getcwd()

    # Reuse a store that is already pointed at the same file, so switching to
    # Solo twice does not leak a connection.
    store = agent.memory
    if store is None or os.path.abspath(store.path) != os.path.abspath(
        preset.memory_path
    ):
        if store is not None:
            store.close()
        store = MemoryStore(path=preset.memory_path)

    agent.attach_memory(store, project=project, inject=preset.inject)
    agent.registry.register(MemorySaveTool(store=store, project=project))
    agent.registry.register(MemorySearchTool(store=store, project=project))


__all__ = [
    "SOLO_MODEL",
    "SOLO_BASE_URL",
    "SOLO_INJECT",
    "SOLO_SYSTEM_PROMPT",
    "SoloPreset",
    "looks_like_question",
    "solo_preset",
    "apply_solo",
]
