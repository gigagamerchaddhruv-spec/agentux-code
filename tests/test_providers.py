"""Provider layer: request shaping, response parsing, and error mapping.

httpx is mocked throughout; nothing here touches the network.
"""

import json

import httpx
import pytest

from agentux.config import Config
from agentux.errors import (
    AuthenticationError,
    MalformedResponseError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
    UnsupportedToolCallingError,
)
from agentux.models import Message, ToolCall, ToolDefinition
from agentux.providers.openai_compatible import OpenAICompatibleProvider


@pytest.fixture
def config():
    return Config(api_key="test-key", base_url="http://api.test/v1", model="test-model")


@pytest.fixture
def provider(config):
    return OpenAICompatibleProvider(config)


def _response(payload, status_code=200):
    """Build a fake httpx.Response."""
    response = httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", "http://api.test/v1/chat/completions"),
    )
    return response


def _patch_post(monkeypatch, response):
    async def fake_post(self, url, **kwargs):
        fake_post.last_call = {"url": url, **kwargs}
        return response

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return fake_post


async def test_returns_assistant_content(provider, monkeypatch):
    _patch_post(monkeypatch, _response({"choices": [{"message": {"content": "Hi"}}]}))

    message = await provider.chat([Message(role="user", content="Hello")])

    assert message.role == "assistant"
    assert message.content == "Hi"
    assert message.tool_calls is None


async def test_sends_model_and_authorization(provider, monkeypatch):
    fake = _patch_post(monkeypatch, _response({"choices": [{"message": {"content": "ok"}}]}))

    await provider.chat([Message(role="user", content="Hello")])

    assert fake.last_call["url"] == "http://api.test/v1/chat/completions"
    assert fake.last_call["headers"]["Authorization"] == "Bearer test-key"
    assert fake.last_call["json"]["model"] == "test-model"


async def test_parses_structured_tool_calls(provider, monkeypatch):
    """The model must be able to request tools through the native mechanism."""
    _patch_post(
        monkeypatch,
        _response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "shell",
                                        "arguments": '{"command": "ls"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        ),
    )

    message = await provider.chat([Message(role="user", content="List files")])

    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].name == "shell"
    assert message.tool_calls[0].arguments == {"command": "ls"}
    assert message.tool_calls[0].id == "call_1"


async def test_parses_multiple_tool_calls(provider, monkeypatch):
    _patch_post(
        monkeypatch,
        _response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "read", "arguments": '{"path": "a"}'},
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "read", "arguments": '{"path": "b"}'},
                                },
                            ]
                        }
                    }
                ]
            }
        ),
    )

    message = await provider.chat([Message(role="user", content="go")])

    assert [c.name for c in message.tool_calls] == ["read", "read"]
    assert [c.id for c in message.tool_calls] == ["call_1", "call_2"]


async def test_tool_definitions_are_sent_with_openai_schema(provider, monkeypatch):
    fake = _patch_post(monkeypatch, _response({"choices": [{"message": {"content": "ok"}}]}))
    tools = [
        ToolDefinition(
            name="filesystem",
            description="Files",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    ]

    await provider.chat([Message(role="user", content="x")], tools)

    payload = fake.last_call["json"]
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["function"]["name"] == "filesystem"
    assert payload["tool_choice"] == "auto"


async def test_omits_tools_key_when_no_tools(provider, monkeypatch):
    fake = _patch_post(monkeypatch, _response({"choices": [{"message": {"content": "ok"}}]}))

    await provider.chat([Message(role="user", content="x")], [])

    assert "tools" not in fake.last_call["json"]


async def test_tool_result_message_round_trips(provider, monkeypatch):
    """A tool result must be sent back with its tool_call_id intact."""
    fake = _patch_post(monkeypatch, _response({"choices": [{"message": {"content": "ok"}}]}))

    await provider.chat(
        [
            Message(role="user", content="run ls"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="call_1", name="shell", arguments={"command": "ls"})],
            ),
            Message(role="tool", content="file.txt", tool_call_id="call_1"),
        ]
    )

    sent = fake.last_call["json"]["messages"]
    assert sent[1]["tool_calls"][0]["id"] == "call_1"
    assert json.loads(sent[1]["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}
    assert sent[2]["role"] == "tool"
    assert sent[2]["tool_call_id"] == "call_1"


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, AuthenticationError),
        (403, AuthenticationError),
        (404, ModelNotFoundError),
        (429, RateLimitError),
        (500, ProviderError),
        (400, ProviderError),
        (418, ProviderError),
    ],
)
async def test_http_errors_map_to_useful_exceptions(provider, monkeypatch, status, expected):
    _patch_post(monkeypatch, _response({"error": "nope"}, status_code=status))

    with pytest.raises(expected):
        await provider.chat([Message(role="user", content="x")])


async def test_auth_error_mentions_the_env_var(provider, monkeypatch):
    _patch_post(monkeypatch, _response({"error": "bad key"}, status_code=401))

    with pytest.raises(AuthenticationError) as exc:
        await provider.chat([Message(role="user", content="x")])

    assert "AGENTUX_API_KEY" in str(exc.value)


async def test_network_failure_is_reported(provider, monkeypatch):
    async def boom(self, url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", boom)

    with pytest.raises(NetworkError):
        await provider.chat([Message(role="user", content="x")])


async def test_timeout_is_reported(provider, monkeypatch):
    async def boom(self, url, **kwargs):
        raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(httpx.AsyncClient, "post", boom)

    with pytest.raises(ProviderError) as exc:
        await provider.chat([Message(role="user", content="x")])

    assert "timeout" in str(exc.value).lower()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": "not an object"}]},
        [1, 2, 3],
    ],
)
async def test_malformed_responses_raise(provider, monkeypatch, payload):
    _patch_post(monkeypatch, _response(payload))

    with pytest.raises(MalformedResponseError):
        await provider.chat([Message(role="user", content="x")])


async def test_unparseable_tool_arguments_raise(provider, monkeypatch):
    _patch_post(
        monkeypatch,
        _response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "shell", "arguments": "{not json"},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
    )

    with pytest.raises(MalformedResponseError):
        await provider.chat([Message(role="user", content="x")])


async def test_empty_tool_arguments_become_empty_dict(provider, monkeypatch):
    _patch_post(
        monkeypatch,
        _response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "shell", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
    )

    message = await provider.chat([Message(role="user", content="x")])
    assert message.tool_calls[0].arguments == {}


async def test_tool_call_without_function_name_raises(provider, monkeypatch):
    _patch_post(
        monkeypatch,
        _response(
            {
                "choices": [
                    {"message": {"tool_calls": [{"id": "c1", "type": "function"}]}}
                ]
            }
        ),
    )

    with pytest.raises(MalformedResponseError):
        await provider.chat([Message(role="user", content="x")])


async def test_non_json_body_raises(provider, monkeypatch):
    response = httpx.Response(
        status_code=200,
        text="<html>login page</html>",
        request=httpx.Request("POST", "http://api.test/v1/chat/completions"),
    )
    _patch_post(monkeypatch, response)

    with pytest.raises(MalformedResponseError):
        await provider.chat([Message(role="user", content="x")])


def test_supports_tool_calling(provider):
    assert provider.supports_tool_calling() is True


def test_get_model_name(provider):
    assert provider.get_model_name() == "test-model"


# ------------------------------------------------------- Ollama compatibility
def _ollama_provider():
    return OpenAICompatibleProvider(
        Config(api_key="ollama", base_url="http://localhost:11434/v1", model="qwen2.5-coder:7b")
    )


async def test_ollama_tool_calls_are_parsed(monkeypatch):
    """Ollama returns standard OpenAI-shaped tool calls; they must work."""
    _patch_post(
        monkeypatch,
        _response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_2jhbt3ex",
                                    "index": 0,
                                    "type": "function",
                                    "function": {
                                        "name": "filesystem",
                                        "arguments": '{"action":"overview"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
    )

    message = await _ollama_provider().chat(
        [Message(role="user", content="look")],
        [ToolDefinition(name="filesystem", description="f", parameters={"type": "object"})],
    )

    assert message.tool_calls[0].name == "filesystem"
    assert message.tool_calls[0].arguments == {"action": "overview"}
    assert message.content is None, "empty content beside tool calls becomes None"


async def test_emulated_tool_call_is_refused_not_executed(monkeypatch):
    """A model printing JSON as text must be rejected, never parsed and run."""
    _patch_post(
        monkeypatch,
        _response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"name": "filesystem", "arguments": {"operation": "rm"}}',
                        }
                    }
                ]
            }
        ),
    )

    with pytest.raises(UnsupportedToolCallingError) as exc:
        await _ollama_provider().chat(
            [Message(role="user", content="delete things")],
            [ToolDefinition(name="filesystem", description="f", parameters={"type": "object"})],
        )

    assert "plain text" in str(exc.value)


async def test_plain_prose_is_not_mistaken_for_a_tool_call(monkeypatch):
    """Normal answers must never trip the emulation check."""
    _patch_post(
        monkeypatch,
        _response({"choices": [{"message": {"content": "I can read files for you."}}]}),
    )

    message = await _ollama_provider().chat(
        [Message(role="user", content="hi")],
        [ToolDefinition(name="filesystem", description="f", parameters={"type": "object"})],
    )

    assert message.content == "I can read files for you."


async def test_json_for_an_unknown_tool_is_left_alone(monkeypatch):
    """Only shapes naming a real tool count as a botched tool call."""
    _patch_post(
        monkeypatch,
        _response(
            {"choices": [{"message": {"content": '{"name": "rm_rf_slash", "arguments": {}}'}}]}
        ),
    )

    message = await _ollama_provider().chat(
        [Message(role="user", content="hi")],
        [ToolDefinition(name="filesystem", description="f", parameters={"type": "object"})],
    )

    assert message.content.startswith('{"name"')


async def test_ollama_no_tool_support_error_is_actionable(monkeypatch):
    _patch_post(
        monkeypatch,
        _response(
            {
                "error": {
                    "message": "registry.ollama.ai/library/deepseek-coder:33b does not support tools",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
        ),
    )

    with pytest.raises(UnsupportedToolCallingError) as exc:
        await _ollama_provider().chat([Message(role="user", content="x")])

    message = str(exc.value)
    assert "deepseek-coder:33b" in message
    assert "--no-tools" in message


async def test_ollama_404_suggests_pull(monkeypatch):
    _patch_post(
        monkeypatch,
        _response({"error": {"message": "model 'nope' not found", "type": "not_found_error"}},
                  status_code=404),
    )

    with pytest.raises(ModelNotFoundError) as exc:
        await _ollama_provider().chat([Message(role="user", content="x")])

    assert "ollama pull" in str(exc.value)
    assert "--list-models" in str(exc.value)


async def test_thinking_model_reasoning_field_is_ignored(monkeypatch):
    """Ollama thinking models add a 'reasoning' key; it must not corrupt the answer."""
    _patch_post(
        monkeypatch,
        _response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The answer is 4.",
                            "reasoning": "Let me think about this...",
                        }
                    }
                ]
            }
        ),
    )

    message = await _ollama_provider().chat([Message(role="user", content="2+2?")])

    assert message.content == "The answer is 4."
    assert "reasoning" not in message.content


async def test_list_models_reads_the_openai_shapes(provider, monkeypatch):
    async def fake_get(self, url, **kwargs):
        fake_get.url = url
        return _response({"object": "list", "data": [{"id": "granite4.1:3b"}, {"id": "llama3.1:8b"}]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    names = await provider.list_models()

    assert names == ["granite4.1:3b", "llama3.1:8b"]
    assert fake_get.url == "http://api.test/v1/models"


async def test_list_models_merges_ollama_capabilities(monkeypatch):
    """Ollama's native /api/tags is what reports tool support."""
    provider = _ollama_provider()
    calls = []

    async def fake_get(self, url, **kwargs):
        calls.append(url)
        if url.endswith("/v1/models"):
            return _response({"data": [{"id": "granite4.1:3b"}, {"id": "smollm:135m"}]})
        return _response(
            {
                "models": [
                    {"name": "granite4.1:3b", "capabilities": ["completion", "tools"]},
                    {"name": "smollm:135m", "capabilities": ["completion"]},
                ]
            }
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    models = await provider.list_models_detailed()
    by_name = {m["name"]: m for m in models}

    assert by_name["granite4.1:3b"]["capabilities"] == ["completion", "tools"]
    assert by_name["smollm:135m"]["capabilities"] == ["completion"]
    assert "http://localhost:11434/api/tags" in calls


async def test_list_models_degrades_when_capabilities_are_unavailable(monkeypatch):
    """A local server that is not Ollama must still list models."""
    provider = OpenAICompatibleProvider(
        Config(api_key="k", base_url="http://localhost:1234/v1", model="m")
    )

    async def fake_get(self, url, **kwargs):
        if url.endswith("/v1/models"):
            return _response({"data": [{"id": "some-model"}]})
        return _response({"error": "not found"}, status_code=404)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    models = await provider.list_models_detailed()

    assert models == [{"name": "some-model"}]


async def test_list_models_reports_an_unreachable_server(provider, monkeypatch):
    async def boom(self, url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "get", boom)

    with pytest.raises(NetworkError) as exc:
        await provider.list_models()

    assert "ollama serve" in str(exc.value)


def test_looks_like_emulated_tool_call_shapes():
    from agentux.providers.openai_compatible import looks_like_emulated_tool_call

    tools = [ToolDefinition(name="filesystem", description="f", parameters={})]

    assert looks_like_emulated_tool_call('{"name": "filesystem", "arguments": {}}', tools)
    assert looks_like_emulated_tool_call(
        '{"name": "filesystem", "arguments": "{\\"path\\": \\"a\\"}"}', tools
    )
    assert not looks_like_emulated_tool_call("just prose", tools)
    assert not looks_like_emulated_tool_call('{"name": "other", "arguments": {}}', tools)
    assert not looks_like_emulated_tool_call("no tools offered", None)
