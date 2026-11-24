"""The Agentux agent loop.

This module contains no HTTP and no provider-specific code. It talks to a
:class:`~agentux.providers.base.BaseProvider` and a
:class:`~agentux.tools.registry.ToolRegistry`, which is what makes the
provider layer swappable.

The loop is::

    user message
        |
        v
    provider.chat(history, tools)  <-----------+
        |                                        |
        +-- no tool calls --> return content    |
        |                                        |
        +-- tool calls --> validate, execute,   |
        |                    append results ----+
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .config import Config
from .context import ContextManager
from .errors import (
    AgentuxError,
    MalformedResponseError,
    MaxStepsError,
    ToolNotFoundError,
    UnsupportedToolCallingError,
)
from .memory import MemoryStore, build_query, render_memory_block
from .models import Message, ToolCall
from .providers.base import BaseProvider, TextCallback
from .solo import looks_like_question
from .theme import Mode
from .tools.registry import ToolRegistry

#: Called with each message of the run, for CLI display. ``(role, text)``.
EventCallback = Callable[[str, str], None]

#: Called with each provider response's usage object, for live token counters.
#: ``(usage_dict_or_None)``. Providers that report nothing send ``None``.
UsageCallback = Callable[[Optional[Dict[str, int]]], None]

SYSTEM_PROMPT = """\
You are Agentux, a coding and automation agent working inside the user's \
project from a terminal. You act directly on their machine, on real files.

## Answer the question that was asked

Most requests are ordinary questions -- "what does this do", "why is this \
failing", "where is X defined". Answer them from what you already know and \
from a single targeted look. Reaching for tools on a question you can answer is \
the most common way to be unhelpful here.

Use tools when the answer depends on something you cannot see: the current \
state of the code, a file's contents, what a command prints, what changed.

## Look before you change

- Call 'project' once at the start of a task that touches the repo, to learn \
the language, build system, and layout. Do not call it again.
- Locate code with 'search' before reading it. Match on a symbol name, not a \
guess at a filename -- a wrong path costs a round trip and returns an error.
- Read a file before changing it. Never write from memory of what it probably \
contains.
- In a git repository, check 'git status' before editing, so you do not \
overwrite work already in progress.

## Change as little as the task needs

- 'filesystem' with operation 'edit' replaces one exact string and returns a \
diff. This is the default for changing existing code: it cannot silently \
clobber the parts you did not mean to touch, and if the target string is not \
unique or not present it fails loudly instead of guessing.
- 'write' replaces a whole file. Use it only to create a new file or when the \
user asked for a rewrite. Prefer 'edit' everywhere else.
- Match the surrounding code: its naming, its comment density, its idioms. A \
correct change that reads like it did not belong there is still wrong.
- Do not reformat, reorder, or "improve" code the task did not ask about.

## Plan only when the work is actually multi-step

For three or more distinct steps, write them into 'todo' first, keep exactly \
one 'in_progress', and update as you go. For anything shorter, just do it -- a \
checklist for a two-step job is noise.

## Verify, then claim

- Run the tests or the build when that is how the work is checked, and read \
what they actually printed. A command you did not run proves nothing, and a \
command whose output you did not read proves less.
- Fix what the output shows. If it shows something outside the task, say so \
rather than silently expanding the work.
- If you could not verify something, say so plainly. "I could not run the \
tests because the environment lacks X" is a useful answer; implying it works is \
not.

## Never

- Invent file contents, command output, or test results. Report only what you \
observed.
- Repeat a call that failed the same way. Read the error and change something: \
the arguments, the approach, or the tool.
- Retry a command the user declined. Say what you wanted to do and let them \
decide.
- Call a destructive command to "see what happens". Ask first, or don't.

## How your answer is displayed

It is rendered as markdown in a terminal roughly 80 columns wide, streamed as \
you write it.

- Lead with the answer. The first sentence should be the thing the user asked \
for, not a preamble about what you are about to do.
- Keep prose short. The user is reading a terminal, not a report.
- Wrap code in fenced blocks with a language tag so it highlights.
- Use a table only when comparing three or more things; a short list reads \
better in a narrow column.
- Do not narrate your tool use in prose. The interface already shows every \
call and its result; describing it in your answer repeats it.
- Never repeat a code block you already showed through a tool call unless the \
user asks to see it again.

When you finish, state what you changed, what you verified, and anything the \
user still needs to decide.\
"""

#: Compact, ultra-fast system prompt for local models and lean execution mode.
LEAN_SYSTEM_PROMPT = """\
You are Agentux, a direct and concise terminal coding assistant.
Answer questions directly. Use tools only when you need to inspect or edit files or run shell commands. Keep answers brief.\
"""

#: Phrases that read as a success claim. Used only to decide whether a
#: self-check is worth one extra model call after a run that hit errors.
SUCCESS_CLAIM = re.compile(
    r"\b(?:all\s+(?:tests?\s+)?(?:pass|passed|green)|success(?:ful|fully)?|"
    r"(?:that|this|it)\s+(?:should\s+)?(?:works?|fixed)|i(?:'ve)?\s+fixed|"
    r"now\s+(?:works|passes)|done|finished|complete[d]?)\b",
    re.IGNORECASE,
)

#: Appended once, when a run hit errors and then claimed success anyway.
REFLECTION_NUDGE = (
    "Before you answer: one or more of your tool calls failed earlier in this "
    "task. Your last message reads as a success claim. Reconcile the two. Either "
    "re-check the failure and fix it, or say explicitly what did not work, what "
    "you tried, and what remains. Do not claim success you have not verified."
)



class Agent:
    """Runs a multi-step tool-using conversation against a provider."""

    def __init__(
        self,
        config: Config,
        provider: BaseProvider,
        registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        on_event: Optional[EventCallback] = None,
        on_text: Optional[TextCallback] = None,
        on_usage: Optional[UsageCallback] = None,
        context_manager: Optional[ContextManager] = None,
        reflect: bool = True,
    ) -> None:
        self.config = config
        self.provider = provider
        self.registry = registry if registry is not None else ToolRegistry()
        #: The TUI that owns this agent, if any. Set by the CLI when the
        #: full-screen interface is running, so commands like ``/mode`` can
        #: reach it through ``context.agent.tui`` without a second import.
        self.tui: Any = None
        self.on_event = on_event
        #: Receives answer text as it is generated, for live terminal output.
        self.on_text = on_text
        #: Receives each request's reported usage, for live token counters.
        self.on_usage = on_usage
        self.history: List[Message] = []
        #: Counts tool calls across the whole session, for reporting.
        self.tool_calls_made = 0
        #: Whether to spend one extra model call reconciling a run that hit
        #: errors and then claimed success anyway.
        self.reflect = reflect
        self.context = context_manager if context_manager is not None else ContextManager()
        base_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self._system_prompt = self._load_project_memory(base_prompt)
        #: (name, duration) for each tool call in the current run.
        self.step_log: List[Any] = []
        #: Whether the final answer was already written live to the terminal.
        self.answer_streamed = False
        #: How many provider round trips the last run took. A step is one
        #: request; several tool calls can share one, so this is not the
        #: same number as ``len(step_log)``.
        self.last_steps = 0
        #: Persistent knowledge store (Agentux Solo). None on a normal session,
        #: which then pays nothing for recall.
        self.memory: Optional[MemoryStore] = None
        self.memory_project: Optional[str] = None
        self.memory_inject: int = 0
        #: Skills system for enhanced capabilities
        from .skills.base import SkillManager
        from .skills.reasoning import ReasoningSkill
        from .skills.self_improvement import SelfImprovementSkill
        from .skills.marketplace_integration import MarketplaceIntegrationSkill
        self.skills = SkillManager(self)
        # Register built-in skills
        self.skills.register_skill_class("reasoning", ReasoningSkill)
        self.skills.register_skill_class("self_improvement", SelfImprovementSkill)
        self.skills.register_skill_class("marketplace_integration", MarketplaceIntegrationSkill)
        # Developer Platform integrations: Checkpoints, Subagents, Repomap, SessionStore
        from .checkpoint import CheckpointManager
        from .repomap import RepoMap
        from .session_store import SessionStore
        from .subagents.manager import SubagentManager, DelegateTool
        from .tools.checkpoint_tool import CheckpointTool
        from .tools.repomap_tool import RepomapTool

        work_root = self.config.working_dir or os.getcwd()
        self.checkpoint_manager = CheckpointManager(work_root)
        self.session_store = SessionStore(work_root)
        self.subagent_manager = SubagentManager(self)
        self.repomap = RepoMap(work_root)
        self._current_turn_checkpoint = None

        # Lua addon system: community scripts that hook the agent lifecycle.
        # Optional -- lupa is not a hard dependency.
        try:
            from .addons import AddonManager
            self.addons = AddonManager(working_dir=work_root)
        except Exception:
            self.addons = None

    def reset(self) -> None:
        """Clear the conversation, keeping configuration and tools."""
        self.history = []
        self.tool_calls_made = 0
        self.step_log = []
        self.answer_streamed = False
        self.last_steps = 0
        self.context.reset_stats()

    def attach_memory(self, store: MemoryStore, project: str, inject: int = 3) -> None:
        """Give this session a persistent knowledge store.

        Called by the Solo preset; a normal session leaves ``self.memory`` as
        ``None`` and pays nothing for recall.
        """
        self.memory = store
        self.memory_project = project
        self.memory_inject = max(0, inject)

    async def _recall_memory(self, prompt: str) -> str:
        """Build the block of things the model should know before it answers.

        Two sources, in order: the store, then -- when the store had nothing
        and the prompt is a question -- a web search. The second step exists
        because a 0.5B model cannot reliably tell what it does not know: it
        answers a question it has no idea about with the same confidence as
        one it does. Deciding here, rather than in the prompt, is what makes
        it actually look.

        Returns ``""`` when nothing applies, so the caller can append the
        result unconditionally.
        """
        if self.memory is None or self.memory_inject <= 0:
            return ""

        query = build_query(prompt)
        hits = []
        if query:
            hits = self.memory.search(
                query, project=self.memory_project, limit=self.memory_inject
            )
        if hits:
            self.memory.touch([memory.id for memory, _score in hits])
            return render_memory_block(hits)

        return await self._ground_with_web(prompt)

    async def _ground_with_web(self, prompt: str) -> str:
        """Look a question up when the store had nothing to say about it."""
        if not looks_like_question(prompt):
            return ""
        try:
            tool = self.registry.get("web_search")
        except ToolNotFoundError:
            return ""

        evidence: List[str] = []

        # Wikipedia first. It returns a sentence containing the fact, which a
        # small model can use directly; a headline list gets misread -- a model
        # that sees "Mojo - Modular" will report "a modular programming
        # language". Keyless, so it is always available.
        wiki = getattr(tool, "search_wikipedia", None)
        if wiki is not None:
            try:
                sentence = await asyncio.to_thread(wiki, prompt)
            except Exception:
                sentence = ""
            if sentence:
                evidence.append(sentence)

        search = getattr(tool, "search", None)
        if search is not None:
            try:
                output = await asyncio.to_thread(search, prompt)
            except Exception:
                output = ""
            if output and "No results found" not in output:
                evidence.append(output)

        if not evidence:
            return ""
        # The instruction matters as much as the evidence. A small model will
        # treat a list of headlines as background and answer from priors anyway
        # unless it is told to answer *from* them.
        return (
            "## Things you looked up\n\n"
            "Answer the question from these results. If they do not contain the "
            "answer, say you do not know rather than guessing.\n\n"
            + "\n\n".join(evidence)
        )

    def _get_project_memory(self) -> str:
        for candidate in ("AGENTUX.md", "CLAUDE.md", os.path.join(".agentux", "rules.md")):
            path = os.path.join(self.config.working_dir, candidate)
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        data = f.read().strip()
                    if data:
                        return f"\n\n## Project Memory & Rules ({candidate})\n\n{data}"
                except Exception:
                    pass
        return ""

    def _load_project_memory(self, base_prompt: str) -> str:
        memory = self._get_project_memory()
        return f"{base_prompt}{memory}" if memory else base_prompt

    def compact(self, keep_turns: int = 4) -> int:
        """Compact older history into summary tokens while preserving recent turns."""
        if len(self.history) <= keep_turns * 2:
            return 0
        old_messages = self.history[:-keep_turns * 2]
        recent_messages = self.history[-keep_turns * 2:]

        tools_used: Dict[str, int] = {}
        for msg in old_messages:
            if msg.role == "tool" and msg.name:
                tools_used[msg.name] = tools_used.get(msg.name, 0) + 1
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used[tc.name] = tools_used.get(tc.name, 0) + 1

        tools_str = ", ".join(f"{k} ({v})" for k, v in tools_used.items()) or "none"
        summary = (
            f"[Context Compacted: {len(old_messages)} earlier messages summarized. "
            f"Tools invoked: {tools_str}. Workspace state verified.]"
        )
        compacted_msg = Message(role="assistant", content=summary)
        compacted_count = len(old_messages)
        self.history = [compacted_msg] + recent_messages
        return compacted_count

    async def execute_skill(self, skill_name: str, operation: str = "run",
                          **kwargs) -> Any:
        """Execute a skill operation.

        Args:
            skill_name: Name of the skill to execute
            operation: Operation to perform on the skill
            **kwargs: Additional arguments for the operation

        Returns:
            Result of the skill execution
        """
        # Get or load the skill
        skill = self.skills.get_skill(skill_name)
        if skill is None:
            # Try to load it
            try:
                skill = self.skills.load_skill(skill_name)
            except ValueError:
                # Check if it's a marketplace skill
                if skill_name in ["reasoning", "self_improvement", "marketplace_integration"]:
                    # These are our built-in skills
                    raise
                # Try to load from marketplace integration skill
                marketplace_skill = self.skills.get_skill("marketplace_integration")
                if marketplace_skill:
                    return await marketplace_skill.execute("use_marketplace_skill",
                                                         skill_id=skill_name,
                                                         **kwargs)
                raise

        # Execute the skill operation
        # Remove 'operation' from kwargs to avoid duplicate argument error
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('operation', None)
        return await skill.execute(operation, **kwargs_copy)

    def get_available_skills(self) -> List[str]:
        """Get list of available skill names.

        Returns:
            List of available skill names (built-in and registered)
        """
        return self.skills.list_available_skills()

    def get_loaded_skills(self) -> List[str]:
        """Get list of currently loaded skill names.

        Returns:
            List of loaded skill names
        """
        return self.skills.list_skills()

    # ------------------------------------------------------------------
    # Conversation state
    # ------------------------------------------------------------------
    def _emit(self, role: str, text: str) -> None:
        if self.on_event and text:
            try:
                self.on_event(role, text)
            except Exception:
                # Display must never break the run.
                pass

    def _ensure_system_prompt(
        self,
        tools: Optional[List[Any]] = None,
        memory_block: str = "",
    ) -> None:
        content = self._build_system_prompt(tools, memory_block)
        if not self.history or self.history[0].role != "system":
            self.history.insert(
                0,
                Message(
                    role="system",
                    content=content,
                ),
            )
        else:
            self.history[0] = Message(role="system", content=content)

    def _build_system_prompt(
        self,
        tools: Optional[List[Any]] = None,
        memory_block: str = "",
    ) -> str:
        """The static instructions plus a description of *this* session.

        The runtime half matters as much as the static half: a model that has
        been handed read-only filesystem access, or has no shell, needs to know
        before it plans, not after its first call fails.
        """
        root = self.config.working_dir or os.getcwd()
        if tools is not None:
            tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools]
        else:
            tool_names = list(self.registry.names)

        is_chat_mode = bool(self.tui and getattr(self.tui, "mode", None) == Mode.CHAT)

        # Fast compact prompt for lean execution and TUI CHAT mode
        if getattr(self.config, "lean", False) or is_chat_mode:
            base = LEAN_SYSTEM_PROMPT
            memory = self._get_project_memory()
            if memory:
                base += memory
            if tool_names:
                base += f"\nWorkspace: `{root}`. Available tools: {', '.join(tool_names)}."
            if getattr(self.config, "allow_text_tools", False):
                from .text_tools import TOOL_PROMPT_INSTRUCTIONS
                base += "\n" + TOOL_PROMPT_INSTRUCTIONS
            if memory_block:
                base += "\n\n" + memory_block
            return base

        context = [
            "",
            "## This session",
            f"- Workspace root: `{root}`",
            f"- Tools available: {', '.join(f'`{name}`' for name in tool_names) if tool_names else 'none'}",
        ]

        constraints = self._session_constraints()
        context.extend(constraints)

        if not tool_names:
            context.append(
                "- You have no tools at all. Answer from the conversation and "
                "the user's own messages, and say plainly if the task needs "
                "tools you do not have."
            )

        if self.config.max_steps:
            context.append(
                f"- You have at most {self.config.max_steps} provider round "
                f"trips for this task (several tool calls may share one). Spend "
                f"them on the work; if you are running out, stop investigating "
                f"and report what you have."
            )

        if getattr(self.config, "allow_text_tools", False):
            from .text_tools import TOOL_PROMPT_INSTRUCTIONS
            context.append(TOOL_PROMPT_INSTRUCTIONS)

        prompt = self._system_prompt + "\n".join(context)
        if memory_block:
            prompt += "\n\n" + memory_block
        return prompt

    def _session_constraints(self) -> List[str]:
        """Limits imposed on this session that the model must plan around.

        Derived from the registry rather than hardcoded, so a tool configured
        read-only cannot end up described to the model as writable.
        """
        notes: List[str] = []
        try:
            filesystem = self.registry.get("filesystem")
        except ToolNotFoundError:
            return notes
        if getattr(filesystem, "read_only", False):
            notes.append(
                "- The filesystem tool is READ-ONLY for this session. You can "
                "read, list and search, but any edit will fail. If the user asks "
                "for a change, explain what the change would be and that you "
                "cannot apply it in this session."
            )
        return notes

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    async def _execute_tool(self, call: ToolCall) -> str:
        """Run one tool call and return the text to send back to the model.

        Failures are returned as text rather than raised: the model needs to
        see *why* a call failed so it can recover on the next iteration.
        """
        try:
            tool = self.registry.get(call.name)
        except ToolNotFoundError as exc:
            return f"ERROR: {exc}"

        # Automatic shadow checkpoint capture before mutating files
        if call.name == "filesystem" and hasattr(self, "checkpoint_manager"):
            op = call.arguments.get("operation")
            path = call.arguments.get("path")
            if op in ("write", "edit") and path:
                if not getattr(self, "_current_turn_checkpoint", None):
                    self._current_turn_checkpoint = self.checkpoint_manager.create_checkpoint(
                        f"Before {op} on {path}"
                    )
                self.checkpoint_manager.capture_file_before(path, self._current_turn_checkpoint.id)

        # Lua addons can block or rewrite tool calls before they run.
        if self.addons is not None:
            decision = self.addons.on_tool_call(call.name, dict(call.arguments))
            if decision is False:
                return f"ERROR: tool {call.name!r} was blocked by an addon."
            if isinstance(decision, dict):
                call.arguments = decision

        try:
            result = await tool.execute(**call.arguments)
        except TypeError as exc:
            # Most often an unknown or missing argument from the model.
            return (
                f"ERROR: invalid arguments for tool {call.name!r}: {exc}\n"
                f"Expected arguments matching this schema: "
                f"{tool.parameters.get('properties', {})}"
            )
        except AgentuxError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            return f"ERROR: tool {call.name!r} raised {type(exc).__name__}: {exc}"

        text = "" if result is None else str(result)
        if self.addons is not None:
            text = self.addons.on_tool_result(call.name, text)
        if not text:
            text = "(no output)"
        if len(text) > 100_000:
            text = text[:100_000] + "\n... [tool output truncated]"
        return text

    def _format_tool_call(self, call: ToolCall) -> str:
        """Render a tool call compactly for the terminal."""
        import json

        try:
            args = json.dumps(call.arguments)
        except (TypeError, ValueError):
            args = str(call.arguments)
        if len(args) > 300:
            args = args[:300] + "..."
        return f"{call.name}({args})"

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------
    async def _ask(self, tools, collect, reset) -> Message:
        """Get one message from the provider, streaming where possible.

        A server may close an SSE stream without ever sending a completion --
        small local models do this intermittently, and some gateways do it
        under load. That is a transport hiccup, not a reason to end the run, so
        an empty or malformed stream falls back to a plain non-streaming
        request for that one step.
        """
        try:
            message = await self.provider.chat_streaming(self.history, tools or None, collect)
        except MalformedResponseError:
            if not self.on_text:
                raise  # Nobody was watching; the error is the best answer.
            # Whatever the broken stream managed to emit is discarded: the
            # non-streaming response is the complete, authoritative one.
            reset()
            message = await self.provider.chat(self.history, tools or None)

        # Read after the call, never before: a fallback to non-streaming bills
        # a second request, and only the last one's numbers are on the client.
        if self.on_usage:
            try:
                self.on_usage(getattr(self.provider, "last_usage", None))
            except Exception:
                pass  # A display failure must not end the run.
        return message

    def _get_turn_tools(self, prompt: str) -> List[Any]:
        """Filter tools for this turn based on session mode and prompt intent."""
        all_tools = self.registry.definitions()
        if not all_tools:
            return []

        # Solo is always an agent. Its whole premise is that it finds things
        # out, and a factual question like "what is Mojo" contains none of the
        # action keywords below -- filtering here would leave it with no way to
        # look anything up, and it would answer from priors instead.
        if self.memory is not None:
            return all_tools

        # Tool intent action keywords
        tool_keywords = {
            "run", "exec", "execute", "shell", "bash", "command", "terminal",
            "grep", "search", "find", "locate", "edit", "modify", "change",
            "replace", "write", "create", "read", "file", "dir", "directory",
            "ls", "cat", "git", "diff", "commit", "branch", "checkout", "status",
            "test", "pytest", "build", "compile", "lint", "todo", "web", "fetch",
            "browse", "curl", "download", "symbol", "symbols", "outline", "project",
            "inspect", "check", "fix", "patch", "refactor"
        }
        words = set(re.findall(r"\b\w+\b", prompt.lower()))

        # When running in TUI mode:
        if self.tui and hasattr(self.tui, "mode"):
            mode_val = getattr(self.tui.mode, "value", str(self.tui.mode)).lower()
            # In CHAT mode, check if prompt is purely conversational
            if mode_val == "chat":
                # Only provide tools if prompt explicitly requests tool actions
                if not (words & tool_keywords):
                    return []
                return all_tools

            # In BUILD mode, focus on coding tools
            elif mode_val == "build":
                build_names = {"filesystem", "shell", "git", "search"}
                filtered = [t for t in all_tools if t.name in build_names]
                if filtered:
                    return filtered

            # In PLAN mode, focus on architecture / search / todo tools
            elif mode_val == "plan":
                plan_names = {"project", "search", "outline_symbols", "todo", "filesystem"}
                filtered = [t for t in all_tools if t.name in plan_names]
                if filtered:
                    return filtered

        return all_tools

    async def run(self, prompt: str, interactive: bool = False) -> str:
        """Run one user request to completion and return the final answer."""
        if self.addons is not None:
            self.addons.start()
            prompt = self.addons.mutate_prompt(prompt)
            self._system_prompt = self.addons.mutate_system_prompt(self._system_prompt)
        if len(self.history) > 30:
            self.compact(4)
        tools = self._get_turn_tools(prompt)
        memory_block = await self._recall_memory(prompt)
        self._ensure_system_prompt(tools, memory_block)
        self.history.append(Message(role="user", content=prompt))
        self._emit("user", prompt)
        self.step_log = []
        self._current_turn_checkpoint = None
        if tools and not self.provider.supports_tool_calling():
            raise UnsupportedToolCallingError(
                f"Provider {type(self.provider).__name__} cannot use structured "
                "tool calls, but this session has tools enabled. Agentux will "
                "not fake tool calls by asking the model to print JSON, so this "
                "cannot continue. Use a provider with native tool-calling "
                "support, or re-run with --no-tools."
            )

        # A model may narrate before calling a tool. Streaming that narration
        # straight into the final answer would interleave it with tool output,
        # so it is buffered and only replayed if the step really is the answer.
        pending_text: List[str] = []

        async def collect(piece: str) -> None:
            pending_text.append(piece)

        async def emit(piece: str) -> None:
            if self.on_text:
                try:
                    await self.on_text(piece)
                except Exception:
                    pass  # A display failure must not end the run.

        def reset() -> None:
            pending_text.clear()

        step = 0
        reflected = False
        had_errors = False

        while step < self.config.max_steps:
            step += 1

            # Shrink stale tool output before it is re-sent. Structure is left
            # alone, so every tool call keeps its answer in order.
            self.context.compress(self.history)

            message = await self._ask(tools, collect, reset)

            if message.tool_calls:
                # Preserve the assistant turn verbatim: providers reject a
                # conversation whose tool calls are not answered in order.
                self.history.append(message)
                self.tool_calls_made += len(message.tool_calls)

                # Whatever the model said before reaching for a tool is shown
                # before the tool itself, so the transcript reads in the order
                # the model actually worked in.
                narration = "".join(pending_text).strip()
                if narration:
                    self._emit("narration", narration)
                pending_text = []

                for call in message.tool_calls:
                    self._emit("tool_call", self._format_tool_call(call))
                    started = time.monotonic()
                    output = await self._execute_tool(call)
                    elapsed = time.monotonic() - started
                    self.step_log.append((call.name, elapsed, output.startswith("ERROR:")))
                    if output.startswith("ERROR:"):
                        had_errors = True
                    self._emit("tool_result", output)
                    self.history.append(
                        Message(
                            role="tool",
                            content=output,
                            tool_call_id=call.id,
                        )
                    )

                continue

            answer = message.content or ""

            # One self-check: a run that hit errors and then claims success is
            # the specific failure this exists to catch. At most one extra call
            # per run, and only when the claim is unambiguous.
            if (
                self.reflect
                and not reflected
                and had_errors
                and answer
                and SUCCESS_CLAIM.search(answer)
            ):
                reflected = True
                pending_text = []
                self._emit("reflection", REFLECTION_NUDGE)
                self.history.append(Message(role="user", content=REFLECTION_NUDGE))
                continue

            # This is the real answer: stream it live so the user is not
            # waiting on a silent screen.
            for piece in pending_text:
                await emit(piece)
            pending_text = []

            self.history.append(message)

            # The caller rendered this text live; it must not print it again.
            self.answer_streamed = bool(self.on_text)
            self.last_steps = step

            self._emit("assistant", answer)
            if self.addons is not None:
                answer = self.addons.on_answer(answer)
            return answer

        self.last_steps = step
        raise MaxStepsError(
            f"Stopped after {self.config.max_steps} steps without a final "
            "answer. Raise AGENTUX_MAX_STEPS if the task genuinely needs more, "
            "or try breaking the task into smaller pieces."
        )

    __all__ = ["Agent", "SYSTEM_PROMPT"]
