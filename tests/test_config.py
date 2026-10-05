"""Configuration loading and validation."""

import pytest

from agentux.config import DEFAULT_BASE_URL, Config, load_config
from agentux.errors import ConfigurationError


def test_missing_api_key_raises_with_guidance(monkeypatch):
    monkeypatch.delenv("AGENTUX_API_KEY", raising=False)
    with pytest.raises(ConfigurationError) as exc:
        load_config()
    assert "AGENTUX_API_KEY is not set" in str(exc.value)


def test_key_is_optional_when_not_required(monkeypatch):
    monkeypatch.delenv("AGENTUX_API_KEY", raising=False)
    config = load_config(require_api_key=False)
    assert config.api_key is None


def test_loads_from_environment(monkeypatch):
    monkeypatch.setenv("AGENTUX_API_KEY", "test-key")
    monkeypatch.setenv("AGENTUX_MODEL", "test-model")
    monkeypatch.setenv("AGENTUX_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("AGENTUX_TEMPERATURE", raising=False)
    monkeypatch.delenv("AGENTUX_MAX_STEPS", raising=False)

    config = load_config()

    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.base_url == "https://example.test/v1"


def test_default_base_url(monkeypatch):
    monkeypatch.setenv("AGENTUX_API_KEY", "k")
    monkeypatch.delenv("AGENTUX_BASE_URL", raising=False)
    assert load_config().base_url == DEFAULT_BASE_URL


def test_whitespace_only_key_is_rejected(monkeypatch):
    monkeypatch.setenv("AGENTUX_API_KEY", "   ")
    with pytest.raises(ConfigurationError):
        load_config()


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("AGENTUX_MAX_STEPS", "not-an-int", "must be an integer"),
        ("AGENTUX_MAX_TOKENS", "many", "must be an integer"),
        ("AGENTUX_TEMPERATURE", "warm", "must be a number"),
        ("AGENTUX_TIMEOUT", "slow", "must be a number"),
    ],
)
def test_invalid_numeric_values(monkeypatch, name, value, message):
    monkeypatch.setenv("AGENTUX_API_KEY", "k")
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError) as exc:
        load_config()
    assert name in str(exc.value)
    assert message in str(exc.value)


def test_blank_numeric_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("AGENTUX_API_KEY", "k")
    monkeypatch.setenv("AGENTUX_MAX_STEPS", "  ")
    assert load_config().max_steps == 25


def test_repr_never_leaks_the_api_key():
    config = Config(api_key="super-secret-key")
    assert "super-secret-key" not in repr(config)
    assert "super-secret-key" not in str(config)
    assert "***" in repr(config)


def test_chat_url_joins_correctly():
    config = Config(api_key="k", base_url="https://x.test/v1/")
    assert config.chat_url == "https://x.test/v1/chat/completions"
    assert Config(api_key="k", base_url="https://x.test/v1").chat_url.endswith(
        "/chat/completions"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": ""},
        {"base_url": "ftp://x.test"},
        {"model": ""},
        {"max_steps": 0},
        {"max_tokens": 0},
        {"timeout": 0},
    ],
)
def test_validate_rejects_bad_settings(kwargs):
    with pytest.raises(ConfigurationError):
        Config(api_key="k", **kwargs).validate()


def test_validate_accepts_good_settings():
    assert Config(api_key="k").validate() is not None


# --------------------------------------------------------------- local servers
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:1234/v1",
        "http://0.0.0.0:8000/v1",
        "http://[::1]:11434/v1",
    ],
)
def test_local_urls_are_detected(url):
    from agentux.config import is_local_url

    assert is_local_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "http://ollama.internal:11434/v1",
        "https://example.com/v1",
        "",
    ],
)
def test_remote_urls_are_not_local(url):
    from agentux.config import is_local_url

    assert is_local_url(url) is False


def test_no_api_key_needed_for_a_local_server(monkeypatch):
    """Ollama has no auth; demanding a key there is busywork."""
    monkeypatch.delenv("AGENTUX_API_KEY", raising=False)
    monkeypatch.setenv("AGENTUX_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("AGENTUX_MODEL", "granite4.1:3b")

    config = load_config()

    assert config.api_key, "a placeholder is filled in for local servers"
    assert config.model == "granite4.1:3b"


def test_explicit_key_still_wins_for_local(monkeypatch):
    monkeypatch.setenv("AGENTUX_API_KEY", "my-key")
    monkeypatch.setenv("AGENTUX_BASE_URL", "http://localhost:11434/v1")
    assert load_config().api_key == "my-key"


def test_remote_server_still_requires_a_key(monkeypatch):
    monkeypatch.delenv("AGENTUX_API_KEY", raising=False)
    monkeypatch.setenv("AGENTUX_BASE_URL", "https://api.openai.com/v1")
    with pytest.raises(ConfigurationError):
        load_config()


def test_config_exposes_endpoints_and_locality():
    config = Config(api_key="k", base_url="http://localhost:11434/v1")
    assert config.is_local is True
    assert config.models_url == "http://localhost:11434/v1/models"
    assert config.chat_url == "http://localhost:11434/v1/chat/completions"

    remote = Config(api_key="k", base_url="https://api.openai.com/v1")
    assert remote.is_local is False
