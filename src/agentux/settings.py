"""Session settings, changed from inside the running agent.

The environment variable is the right place for a *secret* and the wrong place
for a preference. ``AGENTUX_MODEL=qwen3:14b`` typed once per terminal is
noise; retyping it every session is worse. This module lets ``/set`` change
either kind for the live session, and ``/set save`` persist everything that is
not a secret.

Two rules hold throughout:

* **The API key is never displayed.** It is shown as ``set`` or ``not set``,
  never partially, never masked-into-shape, because a masked key still leaks
  its length and its last characters.
* **A secret is never written to disk.** :func:`agentux.config.save_settings`
  refuses it outright; ``/set save`` simply never offers it.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.text import Text

from .config import Config
from .errors import ConfigurationError

#: ``/set <name>`` -> the :class:`Config` attribute it writes. Aliases are
#: resolved in :func:`resolve_key`, so this table only holds canonical names.
SETTABLE: Dict[str, str] = {
    "model": "model",
    "base-url": "base_url",
    "api-key": "api_key",
    "max-steps": "max_steps",
    "timeout": "timeout",
    "temperature": "temperature",
    "max-tokens": "max_tokens",
    "cwd": "working_dir",
}

#: Every spelling a user might reasonably type.
ALIASES: Dict[str, str] = {
    "name": "model",
    "url": "base-url",
    "baseurl": "base-url",
    "endpoint": "base-url",
    "key": "api-key",
    "apikey": "api-key",
    "token": "api-key",
    "steps": "max-steps",
    "maxsteps": "max-steps",
    "dir": "cwd",
    "directory": "cwd",
    "workspace": "cwd",
    "temp": "temperature",
    "tok": "max-tokens",
    "maxtokens": "max-tokens",
    "verbose": "verbose",
    "stream": "stream",
}

#: Session toggles. These are not on :class:`Config`; they are behaviour the
#: CLI owns, so they live in a plain dict the CLI reads after each command.
TOGGLES: Dict[str, str] = {
    "verbose": "Show full tool arguments and results",
    "stream": "Print the answer as it is generated",
}

#: Changing any of these invalidates the HTTP client, so the provider has to be
#: rebuilt or the change silently does nothing.
REBUILDS_PROVIDER = {"model", "base_url", "api_key"}

#: Changing this invalidates every tool's root and cwd.
REBUILDS_TOOLS = {"working_dir"}

#: Changing these changes what the next request looks like, so the conversation
#: is no longer the same conversation.
INVALIDATES_HISTORY = {"model", "base_url", "api_key", "temperature"}


def resolve_key(name: str) -> Optional[str]:
    """Map a user-typed name to a canonical setting, or ``None``."""
    key = name.strip().lower().replace("_", "-")
    if key in SETTABLE or key in TOGGLES:
        return key
    key = ALIASES.get(key, key)
    if key in SETTABLE or key in TOGGLES:
        return key
    # A unique prefix is enough to be unambiguous: "max" -> nothing (two
    # matches), "base" -> "base-url".
    matches = [k for k in list(SETTABLE) + list(TOGGLES) if k.startswith(key)]
    return matches[0] if len(matches) == 1 else None


def describe_settings(config: Config) -> Text:
    """Render the current settings for display."""
    text = Text()
    text.append("current settings\n")

    rows: List[Tuple[str, str, bool]] = [
        ("model", config.model, False),
        ("base-url", config.base_url, False),
        # Presence only. Even the last four characters are a leak.
        ("api-key", "set" if config.api_key else "not set", True),
        ("max-steps", str(config.max_steps), False),
        ("timeout", f"{config.timeout:g}s", False),
        ("temperature", f"{config.temperature:g}", False),
        ("max-tokens", str(config.max_tokens), False),
        ("cwd", config.working_dir, False),
    ]

    for label, value, muted in rows:
        text.append(f"  {label:<12}", style="dim")
        text.append(value, style="dim" if muted else "cyan")
        text.append("\n")

    return text


def parse_value(key: str, value: str, config: Config) -> Any:
    """Convert and range-check one value, or raise with a usable message."""
    if key == "model":
        if not value.strip():
            raise ConfigurationError("The model name cannot be empty.")
        return value.strip()

    if key == "base-url":
        candidate = value.strip().rstrip("/")
        if not candidate.startswith(("http://", "https://")):
            raise ConfigurationError(
                f"The base URL must start with http:// or https://. Got {value!r}."
            )
        return candidate

    if key == "api-key":
        # An explicitly empty value is a way to clear the key, which matters
        # when switching from a remote provider to a local one.
        return value.strip() or None

    if key == "cwd":
        path = os.path.abspath(os.path.expanduser(value.strip()))
        if not os.path.isdir(path):
            raise ConfigurationError(f"{path} is not a directory.")
        return path

    if key in ("max-steps", "max-tokens"):
        try:
            number = int(value)
        except ValueError:
            raise ConfigurationError(f"{key.replace('-', ' ')} must be a whole number, got {value!r}.")
        if number < 1:
            raise ConfigurationError(f"{key.replace('-', ' ')} must be at least 1.")
        return number

    if key == "timeout":
        try:
            number = float(value)
        except ValueError:
            raise ConfigurationError(f"Timeout must be a number of seconds, got {value!r}.")
        if number <= 0:
            raise ConfigurationError("Timeout must be greater than 0.")
        return number

    if key == "temperature":
        try:
            number = float(value)
        except ValueError:
            raise ConfigurationError(f"Temperature must be a number, got {value!r}.")
        if not 0.0 <= number <= 2.0:
            raise ConfigurationError(
                f"Temperature must be between 0 and 2. Got {number}. "
                "0 is deterministic, which is what you usually want for code."
            )
        return number

    raise ConfigurationError(f"{key!r} is not a setting that takes a value.")


def to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("on", "true", "yes", "1", "y"):
        return True
    if lowered in ("off", "false", "no", "0", "n"):
        return False
    raise ConfigurationError(f"Expected on or off, got {value!r}.")


async def apply_setting(
    agent: Any,
    key: str,
    value: str,
    session: Optional[Dict[str, Any]] = None,
    rebuild_tools: Optional[Callable[[], Any]] = None,
) -> str:
    """Apply one setting to the live session and return a human message.

    ``session`` is the CLI's bag of non-Config behaviour toggles. ``rebuild_tools``
    is a callback the CLI supplies so a working-directory change can re-root
    the tools; keeping it a callback avoids importing the CLI from here, which
    would be a cycle.
    """
    config: Config = agent.config
    session = session if session is not None else {}

    if key in TOGGLES:
        enabled = to_bool(value)
        session[key] = enabled
        return f"{TOGGLES[key]}: {'on' if enabled else 'off'} (this session only)."

    attribute = SETTABLE.get(key)
    if attribute is None:
        raise ConfigurationError(f"{key!r} is not a setting.")

    parsed = parse_value(key, value, config)

    if attribute in REBUILDS_TOOLS and parsed == config.working_dir:
        return f"Working directory is already {parsed}."

    previous = getattr(config, attribute)
    setattr(config, attribute, parsed)
    try:
        config.validate()
    except ConfigurationError:
        setattr(config, attribute, previous)  # Never leave a broken config live.
        raise

    messages = []

    if attribute in REBUILDS_PROVIDER:
        from .providers.openai_compatible import OpenAICompatibleProvider

        agent.provider = OpenAICompatibleProvider(config)
        messages.append(f"Now using {config.base_url} with {config.model}.")

    if attribute in REBUILDS_TOOLS and rebuild_tools is not None:
        agent.registry = rebuild_tools()
        messages.append(f"Tools are now rooted at {config.working_dir}.")

    if attribute in INVALIDATES_HISTORY and agent.history:
        # Keeping messages that were exchanged with a different model or
        # temperature is misleading: the context is not what it was.
        agent.reset()
        messages.append("Conversation cleared, because the model or server changed.")

    if attribute == "api_key":
        messages.append(
            "API key set for this session only. It is never written to disk; "
            "export AGENTUX_API_KEY to make it permanent."
            if parsed
            else "API key cleared."
        )
    else:
        messages.append(f"{key} = {parsed}")

    return " ".join(messages)


def savable(config: Config) -> Dict[str, Any]:
    """The settings that may be written to disk. Never includes the key."""
    return {
        "model": config.model,
        "base_url": config.base_url,
        "max_steps": config.max_steps,
        "timeout": config.timeout,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "working_dir": config.working_dir,
    }


__all__ = ["SETTABLE", "TOGGLES", "ALIASES", "apply_setting", "describe_settings",
           "resolve_key", "parse_value", "savable"]
