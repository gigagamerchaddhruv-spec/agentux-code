"""Provider-neutral data types shared by the agent, providers, and tools.

These deliberately contain no HTTP or provider-specific concepts. A provider
translates its own wire format into these and the agent loop only ever reads
them, which is what keeps the provider layer replaceable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A structured request from the model to run one tool.

    This is only ever populated from a provider's native function-calling
    mechanism -- never from text the model printed.
    """

    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """One turn of the conversation."""

    role: Role
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the OpenAI chat-completions wire shape."""
        payload: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": _dumps(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


class ToolDefinition(BaseModel):
    """A tool advertised to the model.

    ``parameters`` is a JSON Schema object describing the accepted arguments.
    """

    name: str
    description: str
    parameters: Dict[str, Any]


class ToolResult(BaseModel):
    """The outcome of running a tool, ready to be sent back to the model."""

    tool_call_id: str
    output: str
    is_error: bool = False


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value)
