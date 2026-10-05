"""The agent loop: tool dispatch, history, and failure recovery."""

from pathlib import Path

import pytest

from agentux.agent import Agent
from agentux.config import Config
from agentux.errors import MaxStepsError, UnsupportedToolCallingError
from agentux.models import Message, ToolCall
from agentux.providers.base import BaseProvider
from agentux.providers.mock import MockProvider, text, tool_call
from agentux.tools.filesystem import FilesystemTool
from agentux.tools.registry import ToolRegistry


class NoToolProvider(BaseProvider):
    """A provider that cannot do structured tool calling."""

    async def chat(self, messages, tools=None):
        return Message(role="assistant", content="plain text only")

    def supports_tool_calling(self):
        return False


def make_agent(provider, config, registry=None):
    return Agent(config, provider, registry if registry is not None else ToolRegistry())


async def test_returns_final_text_without_tools(config):
    provider = MockProvider([text("The answer is 42.")])
    agent = make_agent(provider, config)

    answer = await agent.run("What is the answer?")

    assert answer == "The answer is 42."


async def test_history_records_system_user_assistant(config):
    provider = MockProvider([text("Hi there")])
    agent = make_agent(provider, config)

    await agent.run("Hello")

    assert [m.role for m in agent.history] == ["system", "user", "assistant"]
    assert agent.history[1].content == "Hello"


async def test_system_prompt_includes_workspace_and_tools(config):
    provider = MockProvider([text("ok")])
    agent = Agent(config, provider, ToolRegistry([FilesystemTool(root=config.working_dir)]))

    await agent.run("hello")

    system = agent.history[0].content
    assert config.working_dir in system
    assert "filesystem" in system


async def test_custom_system_prompt_is_used(config):
    provider = MockProvider([text("ok")])
    agent = make_agent(provider, config, None)
    agent._system_prompt = "BE BRIEF."

    await agent.run("hello")

    assert agent.history[0].content.startswith("BE BRIEF.")


async def test_executes_a_tool_then_returns_the_answer(config, tmp_path):
    """The core loop: model asks, tool runs, result goes back, model answers."""
    target = tmp_path / "note.txt"
    target.write_text("hello from disk")

    provider = MockProvider(
        [
            tool_call("call_1", "filesystem", operation="read", path="note.txt"),
            text("The file says: hello from disk"),
        ]
    )
    agent = make_agent(provider, config, ToolRegistry([FilesystemTool(root=str(tmp_path))]))

    answer = await agent.run("What is in note.txt?")

    assert answer == "The file says: hello from disk"
    assert [m.role for m in agent.history] == [
        "system", "user", "assistant", "tool", "assistant",
    ]
    assert agent.history[3].content == "hello from disk"
    assert agent.history[3].tool_call_id == "call_1"
    assert target.read_text() == "hello from disk"


async def test_assistant_tool_call_turn_is_preserved(config, tmp_path):
    """Providers reject histories whose tool calls are not answered in order."""
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")

    provider = MockProvider(
        [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="c1", name="filesystem", arguments={"operation": "read", "path": "a.txt"}),
                    ToolCall(id="c2", name="filesystem", arguments={"operation": "read", "path": "b.txt"}),
                ],
            ),
            text("Both read."),
        ]
    )
    agent = make_agent(provider, config, ToolRegistry([FilesystemTool(root=str(tmp_path))]))

    await agent.run("read both")

    roles = [m.role for m in agent.history]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert agent.history[3].tool_call_id == "c1"
    assert agent.history[4].tool_call_id == "c2"


async def test_multiple_tool_calls_in_one_turn_all_execute(config, tmp_path):
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")

    provider = MockProvider(
        [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="c1", name="filesystem", arguments={"operation": "read", "path": "a.txt"}),
                    ToolCall(id="c2", name="filesystem", arguments={"operation": "read", "path": "b.txt"}),
                ],
            ),
            text("done"),
        ]
    )
    agent = make_agent(provider, config, ToolRegistry([FilesystemTool(root=str(tmp_path))]))

    await agent.run("read both")

    tool_messages = [m for m in agent.history if m.role == "tool"]
    assert [m.content for m in tool_messages] == ["A", "B"]
    assert agent.tool_calls_made == 2


async def test_multiple_iterations(config, tmp_path):
    """Two separate tool round-trips before the final answer."""
    (tmp_path / "one.txt").write_text("1")
    (tmp_path / "two.txt").write_text("2")

    provider = MockProvider(
        [
            tool_call("c1", "filesystem", operation="read", path="one.txt"),
            tool_call("c2", "filesystem", operation="read", path="two.txt"),
            text("Read both files."),
        ]
    )
    agent = make_agent(provider, config, ToolRegistry([FilesystemTool(root=str(tmp_path))]))

    answer = await agent.run("read one.txt then two.txt")

    assert answer == "Read both files."
    assert len([m for m in agent.history if m.role == "tool"]) == 2
    assert provider.calls.__len__() == 3


async def test_tool_failure_is_returned_to_the_model_for_recovery(config, tmp_path):
    """A failing tool must not abort the run; the model gets the error."""
    provider = MockProvider(
        [
            tool_call("c1", "filesystem", operation="read", path="missing.txt"),
            text("That file does not exist."),
        ]
    )
    agent = make_agent(provider, config, ToolRegistry([FilesystemTool(root=str(tmp_path))]))

    answer = await agent.run("read missing.txt")

    assert answer == "That file does not exist."
    assert "ERROR" in agent.history[3].content
    assert "does not exist" in agent.history[3].content


async def test_unknown_tool_is_reported_to_the_model(config):
    provider = MockProvider(
        [tool_call("c1", "nonexistent_tool", foo="bar"), text("I cannot do that.")]
    )
    agent = make_agent(provider, config, ToolRegistry([FilesystemTool(root="/tmp")]))

    answer = await agent.run("do something impossible")

    assert answer == "I cannot do that."
    assert "No tool named" in agent.history[3].content


async def test_invalid_arguments_are_reported_to_the_model(config):
    provider = MockProvider(
        [tool_call("c1", "filesystem", operation="read", nonexistent_arg="x"), text("Bad args.")]
    )
    agent = make_agent(provider, config, ToolRegistry([FilesystemTool(root="/tmp")]))

    await agent.run("bad call")

    assert "ERROR" in agent.history[3].content


async def test_tool_definitions_are_passed_to_the_provider(config, tmp_path):
    provider = MockProvider([text("ok")])
    agent = make_agent(provider, config, ToolRegistry([FilesystemTool(root=str(tmp_path))]))

    await agent.run("hello")

    tools = provider.tools_seen[0]
    assert tools is not None
    assert [t.name for t in tools] == ["filesystem"]
    assert tools[0].parameters["type"] == "object"


async def test_no_tools_registered_sends_no_definitions(config):
    provider = MockProvider([text("ok")])
    agent = make_agent(provider, config, ToolRegistry())

    await agent.run("hello")

    assert provider.tools_seen[0] is None


async def test_max_steps_prevents_an_infinite_loop(config):
    """A model that keeps calling tools must not loop forever."""
    looping = [
        tool_call(f"c{i}", "filesystem", operation="ls", path=".") for i in range(50)
    ]
    config.max_steps = 5
    agent = make_agent(MockProvider(looping), config, ToolRegistry([FilesystemTool(root=str(config.working_dir))]))

    with pytest.raises(MaxStepsError) as exc:
        await agent.run("loop forever")

    assert "5 steps" in str(exc.value)


async def test_refuses_tools_when_provider_cannot_call_them(config):
    agent = make_agent(NoToolProvider(config), config, ToolRegistry([FilesystemTool(root="/tmp")]))

    with pytest.raises(UnsupportedToolCallingError) as exc:
        await agent.run("do something")

    assert "will not fake tool calls" in str(exc.value)


async def test_history_is_preserved_across_turns(config):
    provider = MockProvider([text("first"), text("second")])
    agent = make_agent(provider, config)

    await agent.run("one")
    await agent.run("two")

    assert [m.content for m in agent.history if m.role == "user"] == ["one", "two"]
    assert sum(1 for m in agent.history if m.role == "system") == 1


async def test_reset_clears_history_but_keeps_tools(config, tmp_path):
    provider = MockProvider([text("a"), text("b")])
    registry = ToolRegistry([FilesystemTool(root=str(tmp_path))])
    agent = make_agent(provider, config, registry)

    await agent.run("one")
    agent.reset()
    await agent.run("two")

    assert len(agent.history) == 3
    assert len(registry) == 1


async def test_events_are_reported(config, tmp_path):
    (tmp_path / "f.txt").write_text("data")
    events = []
    provider = MockProvider(
        [tool_call("c1", "filesystem", operation="read", path="f.txt"), text("done")]
    )
    agent = Agent(
        config,
        provider,
        ToolRegistry([FilesystemTool(root=str(tmp_path))]),
        on_event=lambda role, text: events.append((role, text)),
    )

    await agent.run("read f.txt")

    kinds = [kind for kind, _ in events]
    assert "user" in kinds and "tool_call" in kinds and "assistant" in kinds


async def test_a_broken_event_callback_does_not_break_the_run(config):
    (Path(config.working_dir) / "f.txt").write_text("data")

    def explode(role, text):
        raise RuntimeError("display is on fire")

    provider = MockProvider(
        [tool_call("c1", "filesystem", operation="read", path="f.txt"), text("done")]
    )
    agent = Agent(
        config,
        provider,
        ToolRegistry([FilesystemTool(root=config.working_dir)]),
        on_event=explode,
    )

    assert await agent.run("read f.txt") == "done"


async def test_empty_tool_output_is_reported(config):
    class SilentTool(FilesystemTool):
        async def execute(self, **kwargs):
            return ""

    provider = MockProvider([tool_call("c1", "filesystem", operation="ls", path="."), text("ok")])
    agent = make_agent(provider, config, ToolRegistry([SilentTool(root=str(config.working_dir))]))

    await agent.run("go")

    assert agent.history[3].content == "(no output)"
