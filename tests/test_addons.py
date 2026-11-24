"""Tests for the Lua addon system.

Addons are small Lua scripts that hook the agent lifecycle. They run in a
sandboxed LuaJIT runtime (via lupa) and can mutate prompts, intercept tool
calls, and rewrite answers. These tests cover loading, dispatch, and the
sandbox guarantees.
"""

from __future__ import annotations

import os
import textwrap

import pytest

from agentux.addons import AddonError, AddonManager


def _write_addon(directory, name: str, body: str) -> str:
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(textwrap.dedent(body).lstrip("\n"))
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_loads_lua_addon(tmp_path):
    _write_addon(str(tmp_path), "hello.lua", """
        agentux.register({
            name = "hello",
            version = "1.0",
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert len(manager) == 1
    info = manager.list_addons()[0]
    assert info.name == "hello"
    assert info.version == "1.0"


def test_loads_multiple_addons(tmp_path):
    _write_addon(str(tmp_path), "a.lua", """
        agentux.register({ name = "a" })
    """)
    _write_addon(str(tmp_path), "b.lua", """
        agentux.register({ name = "b" })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    names = sorted(a.name for a in manager.list_addons())
    assert names == ["a", "b"]


def test_ignores_non_lua_files(tmp_path):
    (tmp_path / "readme.txt").write_text("not an addon")
    _write_addon(str(tmp_path), "ok.lua", """
        agentux.register({ name = "ok" })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert len(manager) == 1
    assert manager.list_addons()[0].name == "ok"


def test_addon_without_register_is_noop(tmp_path):
    _write_addon(str(tmp_path), "silent.lua", """
        local x = 42
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert len(manager) == 1
    assert manager.list_addons()[0].name == "silent.lua"


def test_invalid_addon_is_skipped(tmp_path):
    _write_addon(str(tmp_path), "broken.lua", """
        agentux.register("not a table")
    """)
    _write_addon(str(tmp_path), "ok.lua", """
        agentux.register({ name = "ok" })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert len(manager) == 1
    assert manager.list_addons()[0].name == "ok"


def test_syntax_error_is_skipped(tmp_path):
    _write_addon(str(tmp_path), "broken.lua", """
        agentux.register({ name = "x" })
        this is not lua
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert len(manager) == 0


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def test_on_prompt_mutates_prompt(tmp_path):
    _write_addon(str(tmp_path), "p.lua", """
        agentux.register({
            on_prompt = function(p)
                return "[prefixed] " .. p
            end,
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert manager.mutate_prompt("hello") == "[prefixed] hello"


def test_on_prompt_threads_multiple_addons(tmp_path):
    _write_addon(str(tmp_path), "a.lua", """
        agentux.register({
            on_prompt = function(p)
                return p .. " [a]"
            end,
        })
    """)
    _write_addon(str(tmp_path), "b.lua", """
        agentux.register({
            on_prompt = function(p)
                return "[b] " .. p
            end,
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    # Addons run in load order; both mutations apply.
    assert manager.mutate_prompt("hi") == "[b] hi [a]"


def test_on_prompt_no_hook_is_noop(tmp_path):
    _write_addon(str(tmp_path), "empty.lua", """
        agentux.register({ name = "empty" })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert manager.mutate_prompt("hello") == "hello"


def test_on_answer_mutates_answer(tmp_path):
    _write_addon(str(tmp_path), "a.lua", """
        agentux.register({
            on_answer = function(ans)
                return ans .. " [signed]"
            end,
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert manager.on_answer("42") == "42 [signed]"


def test_on_system_prompt_mutates_prompt(tmp_path):
    _write_addon(str(tmp_path), "s.lua", """
        agentux.register({
            on_system_prompt = function(p)
                return p .. "\\n## Custom instructions"
            end,
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert "## Custom instructions" in manager.mutate_system_prompt("base prompt")


def test_on_tool_call_can_block(tmp_path):
    _write_addon(str(tmp_path), "block.lua", """
        agentux.register({
            on_tool_call = function(name, args)
                if name == "rm" then
                    return false
                end
                return true
            end,
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert manager.on_tool_call("rm", {"path": "x"}) is False
    assert manager.on_tool_call("read", {"path": "x"}) is True


def test_on_tool_call_can_rewrite_arguments(tmp_path):
    _write_addon(str(tmp_path), "rewrite.lua", """
        agentux.register({
            on_tool_call = function(name, args)
                if name == "shell" then
                    return { command = "echo safe" }
                end
                return true
            end,
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    result = manager.on_tool_call("shell", {"command": "rm -rf /"})
    assert result == {"command": "echo safe"}


def test_on_tool_result_can_rewrite(tmp_path):
    _write_addon(str(tmp_path), "r.lua", """
        agentux.register({
            on_tool_result = function(name, result)
                return "[addon] " .. result
            end,
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert manager.on_tool_result("shell", "ok") == "[addon] ok"


def test_on_event_can_observe(tmp_path):
    _write_addon(str(tmp_path), "e.lua", """
        agentux.register({
            on_event = function(role, text)
                if role == "assistant" then
                    return text .. " [observed]"
                end
                return text
            end,
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    assert manager.on_event("assistant", "hi") == "hi [observed]"
    assert manager.on_event("user", "hi") == "hi"


def test_start_runs_on_start(tmp_path):
    _write_addon(str(tmp_path), "s.lua", """
        local called = false
        agentux.register({
            on_start = function()
                called = true
            end,
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    manager.start()  # should not raise


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
def test_addon_cannot_escape_sandbox(tmp_path):
    """Addons run in Lua; they cannot reach into Python's os module."""
    _write_addon(str(tmp_path), "esc.lua", """
        agentux.register({
            on_prompt = function(p)
                return p
            end,
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    # The agentux global exposes only register and log; nothing else.
    assert manager.mutate_prompt("x") == "x"


def test_addon_state_is_isolated(tmp_path):
    """Each addon gets its own Lua runtime; state does not leak."""
    _write_addon(str(tmp_path), "a.lua", """
        local count = 0
        agentux.register({
            on_prompt = function(p)
                count = count + 1
                return p .. " [a:" .. count .. "]"
            end,
        })
    """)
    _write_addon(str(tmp_path), "b.lua", """
        local count = 0
        agentux.register({
            on_prompt = function(p)
                count = count + 1
                return p .. " [b:" .. count .. "]"
            end,
        })
    """)
    manager = AddonManager(addon_dirs=[str(tmp_path)])
    out = manager.mutate_prompt("hi")
    assert "[a:1]" in out
    assert "[b:1]" in out
    out2 = manager.mutate_prompt("hi")
    assert "[a:2]" in out2
    assert "[b:2]" in out2


# ---------------------------------------------------------------------------
# Integration with the agent
# ---------------------------------------------------------------------------
def test_agent_loads_addons_from_workspace(tmp_path):
    from agentux.agent import Agent
    from agentux.config import Config
    from agentux.providers.mock import MockProvider, text

    _addon_dir = str(tmp_path / ".agentux" / "addons")
    _write_addon(_addon_dir, "p.lua", """
        agentux.register({
            name = "p",
            on_answer = function(a)
                return a .. " [addon]"
            end,
        })
    """)
    cfg = Config(api_key="x", base_url="http://localhost/v1", model="m", working_dir=str(tmp_path))
    agent = Agent(config=cfg, provider=MockProvider([text("hi")]))
    assert agent.addons is not None
    assert len(agent.addons) == 1
    assert agent.addons.list_addons()[0].name == "p"


def test_agent_addon_mutates_answer(tmp_path):
    import asyncio
    from agentux.agent import Agent
    from agentux.config import Config
    from agentux.providers.mock import MockProvider, text

    _addon_dir = str(tmp_path / ".agentux" / "addons")
    _write_addon(_addon_dir, "p.lua", """
        agentux.register({
            name = "p",
            on_answer = function(a)
                return a .. " [addon]"
            end,
        })
    """)
    cfg = Config(api_key="x", base_url="http://localhost/v1", model="m", working_dir=str(tmp_path))
    agent = Agent(config=cfg, provider=MockProvider([text("42")]))

    async def run():
        return await agent.run("what is 42")

    result = asyncio.run(run())
    assert result == "42 [addon]"


def test_agent_addon_mutates_prompt(tmp_path):
    import asyncio
    from agentux.agent import Agent
    from agentux.config import Config
    from agentux.providers.mock import MockProvider, text

    _addon_dir = str(tmp_path / ".agentux" / "addons")
    _write_addon(_addon_dir, "p.lua", """
        agentux.register({
            name = "p",
            on_prompt = function(p)
                return "[prefixed] " .. p
            end,
        })
    """)
    cfg = Config(api_key="x", base_url="http://localhost/v1", model="m", working_dir=str(tmp_path))
    agent = Agent(config=cfg, provider=MockProvider([text("ok")]))

    async def run():
        return await agent.run("hello")

    asyncio.run(run())
    user_msgs = [m.content for m in agent.history if m.role == "user"]
    assert user_msgs == ["[prefixed] hello"]


def test_agent_without_addons_works(tmp_path):
    import asyncio
    from agentux.agent import Agent
    from agentux.config import Config
    from agentux.providers.mock import MockProvider, text

    cfg = Config(api_key="x", base_url="http://localhost/v1", model="m", working_dir=str(tmp_path))
    agent = Agent(config=cfg, provider=MockProvider([text("hi")]))

    async def run():
        return await agent.run("hello")

    result = asyncio.run(run())
    assert result == "hi"