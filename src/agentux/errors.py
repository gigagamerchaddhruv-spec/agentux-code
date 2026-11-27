"""Agentux-specific error types."""

from __future__ import annotations


class AgentuxError(Exception):
    """Base exception for all Agentux errors."""
    pass


# Alias for backward compatibility with existing tests
ConfigurationError = AgentuxError


class ConfigError(AgentuxError):
    """Configuration-related errors."""
    pass


class MCPConfigError(AgentuxError):
    """MCP configuration errors."""
    pass


class MCPError(AgentuxError):
    """MCP runtime errors."""
    pass


class MCPProtocolError(MCPError):
    """MCP protocol errors."""
    pass


class MarketplaceError(AgentuxError):
    """Marketplace-related errors."""
    pass


class ToolNotFoundError(AgentuxError):
    """Raised when a requested tool is not found."""
    pass


class MalformedResponseError(AgentuxError):
    """Raised when a provider returns a malformed response."""
    pass


class MaxStepsError(AgentuxError):
    """Raised when the agent exceeds the maximum number of steps."""
    pass


class UnsupportedToolCallingError(AgentuxError):
    """Raised when the provider doesn't support tool calling."""
    pass


class AuthenticationError(AgentuxError):
    """Raised when authentication fails with a provider."""
    pass


class ModelNotFoundError(AgentuxError):
    """Raised when a requested model is not available."""
    pass


class NetworkError(AgentuxError):
    """Raised when network communication fails."""
    pass


class ProviderError(AgentuxError):
    """Base exception for provider-related errors."""
    pass


class RateLimitError(AgentuxError):
    """Raised when a provider rate limits the request."""
    pass


class ToolError(AgentuxError):
    """Base exception for tool-related errors."""
    pass


class SafetyError(AgentuxError):
    """Raised when a safety check fails."""
    pass


class UserAbort(AgentuxError):
    """Raised when the user aborts an operation."""
    pass


__all__ = [
    "AgentuxError",
    "ConfigurationError",
    "ConfigError",
    "MCPConfigError",
    "MCPError",
    "MCPProtocolError",
    "MarketplaceError",
    "ToolNotFoundError",
    "MalformedResponseError",
    "MaxStepsError",
    "UnsupportedToolCallingError",
    "AuthenticationError",
    "ModelNotFoundError",
    "NetworkError",
    "ProviderError",
    "RateLimitError",
    "ToolError",
    "SafetyError",
    "UserAbort",
]