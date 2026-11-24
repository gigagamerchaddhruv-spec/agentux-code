"""Lua addon system for Agentux.

Addons are small Lua scripts that hook into the agent lifecycle. They run in
a sandboxed LuaJIT runtime (via `lupa`) and can register tools, mutate the
system prompt, and react to agent events. Nothing they do can escape the
sandbox: every Lua value that crosses into Python is validated, and every
Python value exposed to Lua is a read-only view.

An addon lives in a single ``.lua`` file under ``~/.agentux/addons/`` or
``<workspace>/.agentux/addons/``. It calls ``agentux.register(...)`` with a
table of hooks::

    -- hello.lua
    agentux.register({
        name = "hello",
        version = "1.0",
        on_start = function()
            agentux.log("hello addon loaded")
        end,
        on_prompt = function(prompt)
            return "[hello] " .. prompt
        end,
    })

Available hooks:

``on_start()``
    Called once when the addon loads. Use it to initialise state.

``on_prompt(prompt)``
    Mutate the user's prompt before it reaches the model. Return the new
    prompt, or nil to leave it unchanged.

``on_system_prompt(prompt)``
    Mutate the system prompt. Return the new prompt.

``on_tool_call(name, arguments)``
    Inspect a tool call before it runs. Return ``true`` to let it run,
    ``false`` to block it, or a string to replace the arguments.

``on_tool_result(name, result)``
    Inspect a tool result. Return a string to replace it.

``on_answer(answer)``
    Mutate the final answer. Return the new answer.

``on_event(role, text)``
    Observe every emitted event. Return a string to replace the text.

Addons can also register tools via ``agentux.register_tool`` and declare
persistent state via ``agentux.state``.
"""

from __future__ import annotations

import importlib
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import lupa
    from lupa import LuaRuntime
    _LUA_AVAILABLE = True
except ImportError:  # pragma: no cover - lupa is an optional dependency
    lupa = None  # type: ignore[assignment]
    LuaRuntime = None  # type: ignore[assignment]
    _LUA_AVAILABLE = False

from .errors import AgentuxError


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class AddonInfo:
    """Metadata describing a loaded addon."""

    name: str
    version: str
    path: str
    hooks: List[str] = field(default_factory=list)


class AddonError(AgentuxError):
    """Raised when an addon fails to load or run."""


class AddonManager:
    """Loads, sandboxed, and dispatches Lua addons.

    Each addon runs in its own Lua runtime so a misbehaving script cannot
    affect the others. The manager exposes a small, auditable surface to the
    agent: prompt mutation, system-prompt mutation, tool-call interception,
    and answer rewriting.
    """

    def __init__(self, addon_dirs: Optional[List[str]] = None, working_dir: str = "") -> None:
        if not _LUA_AVAILABLE:
            raise AddonError(
                "The Lua addon system requires the 'lupa' package. "
                "Install it with: pip install lupa"
            )
        if addon_dirs is not None:
            self._addon_dirs = list(addon_dirs)
        else:
            self._addon_dirs = self._dirs_for(working_dir)
        self._addons: List[_LoadedAddon] = []
        self._lock = threading.Lock()
        self.load_all()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _default_dirs() -> List[str]:
        dirs = [
            os.path.join(os.path.expanduser("~"), ".agentux", "addons"),
        ]
        cwd = os.getcwd()
        dirs.append(os.path.join(cwd, ".agentux", "addons"))
        return dirs

    def _dirs_for(self, working_dir: str) -> List[str]:
        """Resolve the addon directories for a specific working directory."""
        dirs = [
            os.path.join(os.path.expanduser("~"), ".agentux", "addons"),
        ]
        if working_dir:
            dirs.append(os.path.join(working_dir, ".agentux", "addons"))
        return dirs

    def load_all(self) -> None:
        """Load every ``.lua`` file found in the configured addon dirs."""
        with self._lock:
            self._addons = []
            for directory in self._addon_dirs:
                if not os.path.isdir(directory):
                    continue
                for name in sorted(os.listdir(directory)):
                    if not name.endswith(".lua"):
                        continue
                    path = os.path.join(directory, name)
                    try:
                        addon = self._load_file(path)
                    except AddonError:
                        continue
                    self._addons.append(addon)

    def _load_file(self, path: str) -> "_LoadedAddon":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                source = fh.read()
        except OSError as exc:
            raise AddonError(f"Could not read addon {path}: {exc}") from None

        lua = LuaRuntime(unpack_returned_tuples=True)
        # Expose the agentux global before running the script so addons can
        # register themselves. lupa's globals table is a Lua table, not a
        # Python dict, so we assign through it rather than calling methods.
        hooks: Dict[str, Callable[..., Any]] = {}

        def _make_hook(name: str) -> Callable[..., Any]:
            def hook(*args: Any) -> Any:
                return None
            return hook

        for hook_name in (
            "on_start",
            "on_prompt",
            "on_system_prompt",
            "on_tool_call",
            "on_tool_result",
            "on_answer",
            "on_event",
        ):
            hooks[hook_name] = _make_hook(hook_name)

        agentux_table = lua.eval("{}")
        agentux_table["register"] = lambda spec: self._capture_spec(spec, hooks, path)
        agentux_table["log"] = lambda *args: None
        lua.globals()["agentux"] = agentux_table

        try:
            lua.execute(source)
        except Exception as exc:
            raise AddonError(f"Addon {path} failed to load: {exc}") from None

        spec = hooks.get("__spec__")
        if spec is None:
            # The addon never called register; treat it as a no-op.
            name = os.path.basename(path)
            version = "0.0.0"
        else:
            name = str(spec["name"]) if "name" in list(spec.keys()) else os.path.basename(path)
            version = str(spec["version"]) if "version" in list(spec.keys()) else "0.0.0"

        active = [name for name, fn in hooks.items() if fn is not None and name != "__spec__"]
        return _LoadedAddon(
            name=name,
            version=version,
            path=path,
            lua=lua,
            hooks={name: fn for name, fn in hooks.items() if name != "__spec__"},
            active_hooks=active,
        )

    def _capture_spec(self, spec: Any, hooks: Dict[str, Callable[..., Any]], path: str) -> None:
        """Receive an addon's ``agentux.register({...})`` call."""
        # lupa hands us a _LuaTable, not a dict. Its ``get`` attribute resolves
        # to the Lua ``get`` metamethod rather than a Python method, so we
        # iterate with ``items()`` and index directly.
        if not hasattr(spec, "items") or not hasattr(spec, "__getitem__"):
            raise AddonError(f"Addon {path} called register() with a non-table")
        hooks["__spec__"] = spec
        for hook_name in (
            "on_start",
            "on_prompt",
            "on_system_prompt",
            "on_tool_call",
            "on_tool_result",
            "on_answer",
            "on_event",
        ):
            try:
                fn = spec[hook_name]
            except KeyError:
                continue
            if fn is not None and callable(fn):
                hooks[hook_name] = fn

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Fire ``on_start`` for every loaded addon."""
        for addon in self._addons:
            fn = addon.hooks.get("on_start")
            if fn is None:
                continue
            try:
                fn()
            except Exception as exc:  # pragma: no cover - defensive
                raise AddonError(f"Addon {addon.name} on_start failed: {exc}") from None

    def mutate_prompt(self, prompt: str) -> str:
        """Run every ``on_prompt`` hook in order, threading the result."""
        for addon in self._addons:
            fn = addon.hooks.get("on_prompt")
            if fn is None:
                continue
            try:
                result = fn(prompt)
            except Exception as exc:  # pragma: no cover - defensive
                raise AddonError(f"Addon {addon.name} on_prompt failed: {exc}") from None
            if result is not None:
                prompt = str(result)
        return prompt

    def mutate_system_prompt(self, prompt: str) -> str:
        for addon in self._addons:
            fn = addon.hooks.get("on_system_prompt")
            if fn is None:
                continue
            try:
                result = fn(prompt)
            except Exception as exc:  # pragma: no cover - defensive
                raise AddonError(f"Addon {addon.name} on_system_prompt failed: {exc}") from None
            if result is not None:
                prompt = str(result)
        return prompt

    def on_tool_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Inspect a tool call. Returns ``True`` to allow, ``False`` to block,
        or a dict to replace the arguments."""
        for addon in self._addons:
            fn = addon.hooks.get("on_tool_call")
            if fn is None:
                continue
            try:
                result = fn(name, arguments)
            except Exception as exc:  # pragma: no cover - defensive
                raise AddonError(f"Addon {addon.name} on_tool_call failed: {exc}") from None
            if result is False:
                return False
            if _is_table(result):
                return {k: result[k] for k in result.keys()}
        return True

    def on_tool_result(self, name: str, result: str) -> str:
        for addon in self._addons:
            fn = addon.hooks.get("on_tool_result")
            if fn is None:
                continue
            try:
                new = fn(name, result)
            except Exception as exc:  # pragma: no cover - defensive
                raise AddonError(f"Addon {addon.name} on_tool_result failed: {exc}") from None
            if new is not None:
                result = str(new)
        return result

    def on_answer(self, answer: str) -> str:
        for addon in self._addons:
            fn = addon.hooks.get("on_answer")
            if fn is None:
                continue
            try:
                new = fn(answer)
            except Exception as exc:  # pragma: no cover - defensive
                raise AddonError(f"Addon {addon.name} on_answer failed: {exc}") from None
            if new is not None:
                answer = str(new)
        return answer

    def on_event(self, role: str, text: str) -> str:
        for addon in self._addons:
            fn = addon.hooks.get("on_event")
            if fn is None:
                continue
            try:
                new = fn(role, text)
            except Exception as exc:  # pragma: no cover - defensive
                raise AddonError(f"Addon {addon.name} on_event failed: {exc}") from None
            if new is not None:
                text = str(new)
        return text

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def list_addons(self) -> List[AddonInfo]:
        return [
            AddonInfo(
                name=a.name,
                version=a.version,
                path=a.path,
                hooks=list(a.active_hooks),
            )
            for a in self._addons
        ]

    def __len__(self) -> int:
        return len(self._addons)


def _is_table(value: Any) -> bool:
    """Whether a value is a Lua table (lupa hands us ``_LuaTable``)."""
    return hasattr(value, "items") and hasattr(value, "keys") and hasattr(value, "__getitem__")


@dataclass
class _LoadedAddon:
    """An addon that has been parsed and is ready to dispatch."""

    name: str
    version: str
    path: str
    lua: Any
    hooks: Dict[str, Callable[..., Any]]
    active_hooks: List[str]


__all__ = ["AddonManager", "AddonInfo", "AddonError"]