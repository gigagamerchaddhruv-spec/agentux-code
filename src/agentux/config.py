"""Configuration loaded from environment variables.

Agentux is provider-agnostic: it speaks the OpenAI-compatible HTTP API, and
every provider that exposes that shape (OpenAI, Ollama, LiteLLM, vLLM, Qwen,
LM Studio, ...) works by changing ``AGENTUX_BASE_URL`` and ``AGENTUX_MODEL``.
No credentials are ever read from a file or hardcoded in the source.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .errors import AgentuxError as ConfigurationError

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_STEPS = 25
DEFAULT_TIMEOUT = 120

#: Hosts that serve models locally and therefore need no credential.
#: Ollama, LM Studio, vLLM and friends listen on loopback without auth.
LOCAL_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]",
    "host.docker.internal",
}

#: Keeps ``repr(config)`` from ever leaking the key into logs or tracebacks.
REDACTED = "***"


def is_local_url(base_url: str) -> bool:
    """Whether ``base_url`` points at a server on this machine.

    Local servers (Ollama, LM Studio, a local vLLM) do not authenticate, so
    requiring a key for them would be busywork.
    """
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return False
    return host in LOCAL_HOSTS


@dataclass
class Config:
    """Runtime settings for one Agentux session."""

    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_steps: int = DEFAULT_MAX_STEPS
    timeout: float = DEFAULT_TIMEOUT
    #: Working directory the agent is confined to and runs commands in.
    working_dir: str = field(default_factory=os.getcwd)
    #: Whether to allow models to invoke tools via structured text/markdown (useful for local models).
    allow_text_tools: bool = False
    #: Use a lean/compact system prompt for fast prefill on local models.
    lean: bool = False

    def __repr__(self) -> str:
        key = REDACTED if self.api_key else None
        return (
            f"Config(api_key={key!r}, base_url={self.base_url!r}, "
            f"model={self.model!r}, temperature={self.temperature!r}, "
            f"max_tokens={self.max_tokens!r}, max_steps={self.max_steps!r}, "
            f"timeout={self.timeout!r}, working_dir={self.working_dir!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def validate(self) -> "Config":
        """Fail fast on settings that cannot possibly work."""
        if not self.base_url:
            raise ConfigurationError(
                "AGENTUX_BASE_URL is empty. Set it to your provider's "
                f"OpenAI-compatible endpoint (default: {DEFAULT_BASE_URL})."
            )
        if not self.base_url.startswith(("http://", "https://")):
            raise ConfigurationError(
                f"AGENTUX_BASE_URL must start with http:// or https://, got {self.base_url!r}."
            )
        if not self.model:
            raise ConfigurationError("AGENTUX_MODEL is empty. Set the model name to use.")
        if self.max_steps < 1:
            raise ConfigurationError("AGENTUX_MAX_STEPS must be at least 1.")
        if self.max_tokens < 1:
            raise ConfigurationError("AGENTUX_MAX_TOKENS must be at least 1.")
        if self.timeout <= 0:
            raise ConfigurationError("AGENTUX_TIMEOUT must be greater than 0.")
        return self

    @property
    def chat_url(self) -> str:
        """The chat-completions endpoint for this provider."""
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        """The models-listing endpoint for this provider."""
        return f"{self.base_url.rstrip('/')}/models"

    @property
    def is_local(self) -> bool:
        """Whether this points at a server running on this machine."""
        return is_local_url(self.base_url)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigurationError(
            f"{name} must be a number, got {raw!r}. Fix it by setting "
            f"{name} to a numeric value, or unset it to use {default}."
        )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(
            f"{name} must be an integer, got {raw!r}. Fix it by setting "
            f"{name} to a whole number, or unset it to use {default}."
        )


def settings_path() -> str:
    """Where ``/set save`` persists non-secret preferences."""
    override = os.getenv("AGENTUX_SETTINGS_FILE")
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".config", "agentux", "settings.json")


def load_settings() -> Dict[str, Any]:
    """Read persisted settings. Missing or broken files are not fatal."""
    path = settings_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(values: Dict[str, Any]) -> str:
    """Write settings to disk, creating the directory if needed.

    Secrets are refused: a plaintext API key on disk is a liability, and the
    environment variable or the flag is the right place for one.
    """
    if "api_key" in values:
        raise ConfigurationError(
            "Agentux will not save an API key to disk. Set AGENTUX_API_KEY in "
            "your environment, or pass --api-key, or use '/set api-key <key>' "
            "to set it for this session only."
        )

    path = settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def load_config(require_api_key: bool = True) -> Config:
    """Build a :class:`Config` from settings, then the environment.

    Precedence, highest first: ``AGENTUX_*`` environment variables, then the
    persisted settings file, then the built-in defaults. That ordering means
    an exported variable still wins over anything saved earlier, so an
    environment override never becomes sticky.

    An API key is required for remote providers but not for local ones:
    Ollama and friends listen on loopback without authentication, so
    ``AGENTUX_API_KEY`` is filled in with a placeholder when the base URL is
    local and no key was supplied.
    """
    settings = load_settings()
    api_key = os.getenv("AGENTUX_API_KEY", "").strip() or settings.get("api_key") or None
    base_url = _env_str("AGENTUX_BASE_URL", _setting(settings, "base_url", DEFAULT_BASE_URL))

    if require_api_key and not api_key:
        if is_local_url(base_url):
            # Ollama ignores the header; send a placeholder so the provider
            # layer has something to work with.
            api_key = "ollama"
        else:
            raise ConfigurationError(
                "AGENTUX_API_KEY is not set.\n"
                "Set it to your provider's API key, for example:\n"
                '  export AGENTUX_API_KEY="..."\n'
                "\n"
                "Prefer not to use environment variables? Start Agentux and "
                "run '/set api-key <key>' to set it for the session, then "
                "'/set save' to keep the rest of your preferences.\n"
                "\n"
                "Using a local model? Point Agentux at Ollama and no key is "
                "needed:\n"
                '  export AGENTUX_BASE_URL="http://localhost:11434/v1"\n'
                '  export AGENTUX_MODEL="qwen2.5-coder"'
            )

    config = Config(
        api_key=api_key,
        base_url=base_url,
        model=_env_str("AGENTUX_MODEL", _setting(settings, "model", DEFAULT_MODEL)),
        temperature=_env_float(
            "AGENTUX_TEMPERATURE", _setting(settings, "temperature", DEFAULT_TEMPERATURE)
        ),
        max_tokens=_env_int(
            "AGENTUX_MAX_TOKENS", _setting(settings, "max_tokens", DEFAULT_MAX_TOKENS)
        ),
        max_steps=_env_int(
            "AGENTUX_MAX_STEPS", _setting(settings, "max_steps", DEFAULT_MAX_STEPS)
        ),
        timeout=_env_float("AGENTUX_TIMEOUT", _setting(settings, "timeout", DEFAULT_TIMEOUT)),
        working_dir=_env_str(
            "AGENTUX_CWD", _setting(settings, "working_dir", os.getcwd())
        ),
        allow_text_tools=os.getenv("AGENTUX_TEXT_TOOLS", "").lower() in ("1", "true", "yes")
        or bool(settings.get("allow_text_tools", False)),
        lean=os.getenv("AGENTUX_LEAN", "").lower() in ("1", "true", "yes")
        or bool(settings.get("lean", False)),
    )
    return config.validate()


def _setting(settings: Dict[str, Any], key: str, default: Any) -> Any:
    """A settings value, ignoring anything of the wrong type."""
    value = settings.get(key)
    return value if value is not None and isinstance(value, type(default)) else default
