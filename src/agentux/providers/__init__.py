"""Providers: the model backends Agentux can talk to.

The agent loop depends only on :class:`BaseProvider`. To add a provider,
implement that interface and register it in :func:`create_provider` -- no
change to the agent is required.
"""

from __future__ import annotations


from ..config import Config
from .base import BaseProvider
from .mock import MockProvider
from .openai_compatible import OpenAICompatibleProvider

#: Providers selectable by name via AGENTUX_PROVIDER.
REGISTRY = {
    "openai": OpenAICompatibleProvider,
    "openai-compatible": OpenAICompatibleProvider,
    "ollama": OpenAICompatibleProvider,
    "litellm": OpenAICompatibleProvider,
    "mock": MockProvider,
}


def create_provider(name: str, config: Config) -> BaseProvider:
    """Instantiate a provider by name."""
    try:
        provider_class = REGISTRY[name.lower()]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown provider {name!r}. Known providers: {known}") from None
    return provider_class(config)


__all__ = [
    "BaseProvider",
    "OpenAICompatibleProvider",
    "MockProvider",
    "create_provider",
    "REGISTRY",
]
