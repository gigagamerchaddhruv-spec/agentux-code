"""Agentux - a terminal-native, provider-agnostic AI coding agent.

Agentux is the product; the model provider is infrastructure. The core depends
only on the :class:`~agentux.providers.base.BaseProvider` interface, so it runs
against any server exposing an OpenAI-compatible API (OpenAI, Ollama,
LiteLLM, vLLM, LM Studio, ...) without changing a line of the agent loop.

Quick start::

    export AGENTUX_API_KEY="..."
    export AGENTUX_BASE_URL="https://api.openai.com/v1"
    export AGENTUX_MODEL="gpt-4o-mini"
    agentux "explain this project"
"""

__version__ = "0.1.0"

from .agent import Agent
from .config import Config, load_config
from .errors import AgentuxError

__all__ = [
    "__version__",
    "Agent",
    "Config",
    "load_config",
    "AgentuxError",
]
