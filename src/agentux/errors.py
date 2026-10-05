"""Exception hierarchy for Agentux.

Every error raised on purpose by Agentux derives from :class:`AgentuxError`.
The CLI relies on that to tell an expected, user-facing failure (a missing API
key, a rate limit, a declined command) apart from a genuine bug, so new error
types should subclass ``AgentuxError`` rather than ``Exception``.
"""


class AgentuxError(Exception):
    """Base class for all Agentux exceptions."""


class ConfigurationError(AgentuxError):
    """Raised when configuration is missing or invalid."""


class ProviderError(AgentuxError):
    """Raised when a provider call fails."""


class AuthenticationError(ProviderError):
    """Raised when the provider rejects the API key."""


class ModelNotFoundError(ProviderError):
    """Raised when the requested model does not exist on the provider."""


class RateLimitError(ProviderError):
    """Raised when the provider rate-limits the request."""


class NetworkError(ProviderError):
    """Raised when the provider cannot be reached."""


class MalformedResponseError(ProviderError):
    """Raised when the provider returns a response Agentux cannot parse."""


class UnsupportedToolCallingError(AgentuxError):
    """Raised when tool use is requested from a provider that cannot do it.

    Agentux refuses to emulate structured tool calls by asking a model to print
    JSON as plain text, so this is a hard failure rather than a silent
    downgrade.
    """


class ToolError(AgentuxError):
    """Raised when a tool fails to execute."""


class ToolNotFoundError(ToolError):
    """Raised when the model asks for a tool that is not registered."""


class ToolValidationError(ToolError):
    """Raised when a tool call's arguments do not match the tool's schema."""


class SafetyError(ToolError):
    """Raised when a destructive command is blocked or declined."""


class MaxStepsError(AgentuxError):
    """Raised when the agent loop hits its iteration ceiling."""


class UserAbort(AgentuxError):
    """Raised when the user interrupts an in-flight operation."""


__all__ = [
    "AgentuxError",
    "ConfigurationError",
    "ProviderError",
    "AuthenticationError",
    "ModelNotFoundError",
    "RateLimitError",
    "NetworkError",
    "MalformedResponseError",
    "UnsupportedToolCallingError",
    "ToolError",
    "ToolNotFoundError",
    "ToolValidationError",
    "SafetyError",
    "MaxStepsError",
    "UserAbort",
]
