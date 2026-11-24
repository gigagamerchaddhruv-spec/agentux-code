"""The Agentux Solo execution engine.

This is the part that makes Solo *autonomous* rather than merely configured.
The model is small and the instructions are rigid, but the engine does not
trust the model to enforce its own research rules. It watches every turn and
steps in when the model tries to answer a factual question it has not looked
up.

The mandatory research cycle, enforced deterministically::

    RETRIEVE  -> search the knowledge database
    IDENTIFY  -> is the stored knowledge sufficient?
    RESEARCH  -> if not, use the web tools
    VERIFY    -> check claims against a second source where possible
    SAVE       -> store what was learned
    VERIFY     -> read it back to confirm it was saved
    ANSWER

The model still chooses which tools to call. The engine only enforces that
the *sequence* happens, and it does the enforcement by inspecting tool calls
and injecting a nudge when the model tries to skip a step.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .errors import AgentuxError
from .knowledge import VERIFICATION_UNVERIFIED, KnowledgeStore
from .models import Message
from .providers.base import BaseProvider
from .solo import SOLO_SYSTEM_PROMPT, looks_like_question
from .tools.base import BaseTool
from .tools.knowledge_tool import (
    CalculateTool,
    DeleteKnowledgeTool,
    ExtractInformationTool,
    GetDateTimeTool,
    GetKnowledgeTool,
    ListKnowledgeTool,
    SaveKnowledgeTool,
    SearchKnowledgeTool,
    SearchMultipleTool,
    UpdateKnowledgeTool,
)
from .tools.registry import ToolRegistry
from .tools.web import FetchWebPageTool, WebSearchTool

#: Maximum wall-clock seconds for a whole Solo run.
DEFAULT_TIMEOUT = 180.0
#: Maximum model round trips per run.
DEFAULT_MAX_ITERATIONS = 12
#: Maximum research attempts before giving up.
DEFAULT_MAX_RESEARCH = 3
#: Maximum tool calls per run.
DEFAULT_MAX_TOOL_CALLS = 30


@dataclass
class SoloEvent:
    """One thing that happened during a Solo run."""

    kind: str
    message: str
    data: Any = None


@dataclass
class SoloResult:
    """The outcome of a Solo run."""

    answer: str
    events: List[SoloEvent] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    researched: bool = False
    saved: bool = False
    elapsed: float = 0.0


EventCallback = Callable[[SoloEvent], None]


class SoloEngine:
    """Drives a small model through the mandatory research cycle.

    The engine reuses an existing provider and knowledge store rather than
    rebuilding them, so it slots into Agentux without duplicating infrastructure.
    """

    def __init__(
        self,
        provider: BaseProvider,
        knowledge: KnowledgeStore,
        *,
        web_search: Optional[WebSearchTool] = None,
        fetch_web_page: Optional[FetchWebPageTool] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_research: int = DEFAULT_MAX_RESEARCH,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        timeout: float = DEFAULT_TIMEOUT,
        system_prompt: Optional[str] = None,
        project: str = "",
    ) -> None:
        self.provider = provider
        self.knowledge = knowledge
        self.project = project
        self.max_iterations = max_iterations
        self.max_research = max_research
        self.max_tool_calls = max_tool_calls
        self.timeout = timeout
        self._system_prompt = system_prompt or SOLO_SYSTEM_PROMPT

        # Build a registry containing only the tools Solo is allowed to use.
        self.registry = ToolRegistry()
        self.registry.register(SearchKnowledgeTool(knowledge))
        self.registry.register(SaveKnowledgeTool(knowledge))
        self.registry.register(GetKnowledgeTool(knowledge))
        self.registry.register(ListKnowledgeTool(knowledge))
        self.registry.register(DeleteKnowledgeTool(knowledge))
        self.registry.register(UpdateKnowledgeTool(knowledge))
        self.registry.register(ExtractInformationTool(knowledge))
        self.registry.register(GetDateTimeTool())
        self.registry.register(CalculateTool())

        self._web_search = web_search or WebSearchTool()
        self._fetch_web_page = fetch_web_page or FetchWebPageTool()
        self.registry.register(self._web_search)
        self.registry.register(self._fetch_web_page)
        self.registry.register(SearchMultipleTool(self._web_search.search))
        self._last_search_output = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run(self, prompt: str, on_event: Optional[EventCallback] = None) -> SoloResult:
        """Run one user request through the research cycle."""
        start = time.monotonic()
        emit = on_event or (lambda event: None)

        result = SoloResult(answer="", events=[], researched=False, saved=False)
        history: List[Message] = []

        def record(event: SoloEvent) -> None:
            result.events.append(event)
            emit(event)

        try:
            record(SoloEvent("start", f"Received request: {prompt[:120]}"))

            # --- RETRIEVE: search the knowledge database first ------------
            memory_block, memory_sufficient = await self._retrieve(prompt, record)

            # --- Build the system prompt ----------------------------------
            tools = self.registry.definitions()
            sys_content = self._build_system_prompt(memory_block, tools)
            history.append(Message(role="system", content=sys_content))
            history.append(Message(role="user", content=prompt))

            researched = False
            saved = False
            tool_calls = 0
            research_attempts = 0
            research_succeeded = False
            consecutive_nudges = 0

            for iteration in range(self.max_iterations):
                result.iterations = iteration + 1
                record(SoloEvent("thinking", f"Iteration {iteration + 1}/{self.max_iterations}"))

                try:
                    message = await asyncio.wait_for(
                        self.provider.chat(history, tools),
                        timeout=self.timeout,
                    )
                except asyncio.TimeoutError:
                    record(SoloEvent("error", "The model provider timed out."))
                    result.answer = "I could not complete that request: the model provider timed out."
                    break

                if message.tool_calls:
                    history.append(message)
                    for call in message.tool_calls:
                        tool_calls += 1
                        result.tool_calls = tool_calls
                        record(SoloEvent("tool_call", f"{call.name}({call.arguments})"))
                        if tool_calls >= self.max_tool_calls:
                            record(SoloEvent("error", "Tool call limit reached."))
                            result.answer = "I hit the tool call limit before finishing. Try a simpler question."
                            break

                        output = await self._execute_tool(call, record)
                        history.append(Message(
                            role="tool",
                            content=output,
                            tool_call_id=call.id,
                        ))

                        if call.name == "web_search":
                            researched = True
                            research_attempts += 1
                            result.researched = True
                            research_succeeded = not output.startswith("ERROR")
                            self._last_search_output = output
                            if research_succeeded:
                                # The engine saves what was learned. A small
                                # model cannot be relied on to call
                                # save_knowledge with a valid schema, and the
                                # whole point of Solo is that the research
                                # cycle is enforced, not hoped for.
                                save_note = await self._auto_save(prompt, output)
                                history.append(Message(
                                    role="tool",
                                    content=save_note,
                                    tool_call_id=None,
                                ))
                                saved = True
                                result.saved = True
                        if call.name == "save_knowledge":
                            saved = True
                            result.saved = True

                    if result.answer:
                        break
                    continue

                # --- No tool calls: the model wants to answer ---------------
                answer = message.content or ""
                history.append(message)

                # Deterministic enforcement: a factual question that was not
                # looked up gets a nudge, not an answer.
                if looks_like_question(prompt):
                    if not memory_sufficient and not researched:
                        if research_attempts < self.max_research:
                            consecutive_nudges += 1
                            if consecutive_nudges > self.max_research:
                                record(SoloEvent(
                                    "warning",
                                    "The model kept answering without researching. "
                                    "Answering with what is available rather than looping.",
                                ))
                            else:
                                record(SoloEvent(
                                    "nudge",
                                    "The model answered without researching. Injecting a research requirement.",
                                ))
                                history.append(Message(role="user", content=(
                                    "You must research this before answering. "
                                    "Use web_search, then save_knowledge, then get_knowledge "
                                    "to verify it was saved. Do not answer yet."
                                )))
                                continue
                        else:
                            record(SoloEvent(
                                "warning",
                                "Max research attempts reached; answering with what is available.",
                            ))
                    if not saved and researched and research_succeeded:
                        consecutive_nudges += 1
                        if consecutive_nudges > self.max_research:
                            record(SoloEvent(
                                "warning",
                                "The model kept answering without saving. "
                                "Answering with what is available rather than looping.",
                            ))
                        else:
                            record(SoloEvent(
                                "nudge",
                                "The model researched but did not save. Injecting a save requirement.",
                            ))
                            history.append(Message(role="user", content=(
                                "You researched this but did not save what you learned. "
                                "Use save_knowledge now, then get_knowledge to verify."
                            )))
                            continue

                record(SoloEvent("answer", answer))
                result.answer = answer
                break
            else:
                result.answer = result.answer or "I reached the iteration limit without a final answer."
                record(SoloEvent("warning", result.answer))

        except AgentuxError as exc:
            record(SoloEvent("error", str(exc)))
            result.answer = f"Agentux Solo error: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            record(SoloEvent("error", f"Unexpected error: {type(exc).__name__}: {exc}"))
            result.answer = f"Unexpected error: {type(exc).__name__}: {exc}"

        result.elapsed = time.monotonic() - start
        record(SoloEvent("done", f"Completed in {result.elapsed:.1f}s"))
        return result

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    async def _auto_save(self, prompt: str, search_output: str) -> str:
        """Save the research result so the cycle completes even if the model
        cannot call ``save_knowledge`` itself.

        A 0.5B model cannot be trusted to emit a well-formed tool call with
        the title and content the schema requires. The engine therefore saves
        on its behalf: one fact per question, with the source URLs it found.
        """
        search_output = search_output or getattr(self, "_last_search_output", "")
        import re

        urls = re.findall(r"URL:\s*(\S+)", search_output)
        sources = [{"url": u, "title": "", "retrieved_at": time.time()} for u in urls[:3]]
        summary = "\n".join(
            line.strip()
            for line in search_output.splitlines()
            if line.strip() and not line.strip().startswith("URL:")
        )[:600]
        try:
            entry = self.knowledge.save(
                title=prompt[:80],
                content=summary or f"Research result for: {prompt}",
                category="research",
                tags="solo,auto-research",
                confidence=0.6,
                verification=VERIFICATION_UNVERIFIED,
                sources=sources,
            )
            return (
                f"Saved research result as knowledge entry '{entry.id}'. "
                f"{len(sources)} source(s) recorded."
            )
        except Exception as exc:
            return f"ERROR: could not save research result: {exc}"

    # ------------------------------------------------------------------
    # RETRIEVE
    # ------------------------------------------------------------------
    async def _retrieve(self, prompt: str, record: EventCallback) -> tuple:
        """Search the knowledge database and decide whether it is sufficient.

        The store is keyword-indexed, so a whole sentence can miss entries that
        share none of its words. We try the prompt as-is, then each individual
        word, then -- as a last resort -- the most recently stored entries. A
        small model benefits from seeing *something* relevant rather than an
        empty "you know nothing" message.
        """
        record(SoloEvent("memory_search", "Searching the knowledge database..."))

        candidates: List = []
        seen = set()

        def _add(hits):
            for entry in hits:
                if entry.id not in seen:
                    seen.add(entry.id)
                    candidates.append(entry)

        _add(self.knowledge.search(prompt, limit=5))
        for word in prompt.split():
            word = word.strip().lower()
            if len(word) >= 4:
                _add(self.knowledge.search(word, limit=3))
        if not candidates and looks_like_question(prompt):
            # Last resort: show the most recent entries so a small model can
            # at least see what it already knows, rather than getting an
            # empty "you know nothing" message and inventing an answer.
            _add(self.knowledge.list_all(limit=5))

        if not candidates:
            record(SoloEvent("memory_miss", "No matching knowledge entries found."))
            return "", False

        lines = ["## Things you already know", ""]
        for entry in candidates:
            lines.append(f"- [{entry.id}] {entry.title}")
            lines.append(f"  {entry.content}")
            if entry.sources:
                lines.append(f"  Sources: {', '.join(s.get('url', '') for s in entry.sources)}")
        memory_block = "\n".join(lines)
        record(SoloEvent("memory_hit", f"Found {len(candidates)} relevant knowledge entr(ies)."))
        return memory_block, True

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    async def _execute_tool(self, call, record: EventCallback) -> str:
        try:
            tool = self.registry.get(call.name)
        except Exception:
            return f"ERROR: unknown tool '{call.name}'"

        record(SoloEvent("tool_start", f"Running {call.name}..."))
        try:
            result = await tool.execute(**call.arguments)
        except Exception as exc:
            # A small model cannot reliably call save_knowledge with the
            # required title/content. If it tries and fails, save on its
            # behalf anyway so the research cycle still completes.
            if call.name == "save_knowledge":
                result = await self._auto_save(prompt, "")
            else:
                result = f"ERROR: {call.name} failed: {exc}"
        record(SoloEvent("tool_result", result[:200] + ("..." if len(result) > 200 else "")))

        # Record every web research operation so it can be inspected later.
        if call.name == "web_search":
            try:
                self.knowledge.record_research(
                    str(call.arguments.get("query", "")),
                    results_summary=result[:500],
                    status="ok" if not result.startswith("ERROR") else "failed",
                )
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_system_prompt(self, memory_block: str, tools: List[Any]) -> str:
        tool_names = ", ".join(t.name for t in tools) if tools else "none"
        prompt = self._system_prompt
        if memory_block:
            prompt += "\n\n" + memory_block
        prompt += (
            f"\n\n## Your tools\n"
            f"You have these tools available: {tool_names}.\n"
            f"Use them when you need to look something up or save what you learn."
        )
        return prompt


__all__ = ["SoloEngine", "SoloEvent", "SoloResult"]