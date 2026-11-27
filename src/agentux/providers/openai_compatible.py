"""OpenAI-compatible provider.

Implements the ``/chat/completions`` contract that OpenAI, Ollama, LiteLLM,
vLLM, LM Studio, Qwen and most other servers expose. Nothing here is
OpenAI-*specific* in behaviour -- only in wire format.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from ..config import is_local_url
from ..errors import (
    AuthenticationError,
    MalformedResponseError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
    UnsupportedToolCallingError,
)
from ..models import Message, ToolCall, ToolDefinition
from ..usage import usage_from_payload
from .base import BaseProvider, TextCallback

#: Server-sent-event prefix used by the streaming endpoint.
_SSE_PREFIX = "data:"
#: Sentinel that terminates an OpenAI-compatible SSE stream.
_SSE_DONE = "[DONE]"


def _tool_schema(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    """Parse the JSON string a provider sends for tool arguments.

    Providers send arguments as a JSON-encoded string; some send an object.
    An empty or malformed string becomes ``{}`` so the tool's own validation
    produces a useful message instead of an opaque JSONDecodeError.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedResponseError(
                f"Provider sent unparseable tool arguments ({raw!r}): {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise MalformedResponseError(
                f"Provider sent tool arguments that were not an object: {raw!r}"
            )
        return parsed
    raise MalformedResponseError(f"Unexpected tool arguments type: {type(raw).__name__}")


def _error_detail(body: str) -> str:
    """Pull the human-readable message out of an error body.

    OpenAI-style servers wrap errors in ``{"error": {"message": ...}}``;
    Ollama uses the same shape, so one parser covers both.
    """
    text = (body or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text[:500]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("detail")
            if message:
                return str(message)[:500]
        elif isinstance(error, str):
            return error[:500]
        if isinstance(data.get("message"), str):
            return data["message"][:500]
        return text[:500]
    return text[:500]


def looks_like_emulated_tool_call(content: str, tools: Optional[List[ToolDefinition]]) -> bool:
    """Detect a model that printed a tool call as text instead of calling it.

    Some local models advertise a ``tools`` capability but, when served through
    an OpenAI-compatible shim, return ``{"name": ..., "arguments": ...}`` as
    ordinary content. Agentux deliberately does **not** execute that: parsing
    arbitrary text as a tool call would be exactly the faking the design
    forbids, and it would let any prose that happens to look like JSON run
    commands. Instead we recognise the shape and report it, so the user can
    pick a model that uses native tool calling.
    """
    if not content or not tools:
        return False
    text = content.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return False
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(parsed, dict):
        return False
    # Accept both the bare and the wrapped shapes seen in the wild.
    candidate = parsed.get("tool_call") if isinstance(parsed.get("tool_call"), dict) else parsed
    name = candidate.get("name") or candidate.get("function")
    if isinstance(name, dict):
        name = name.get("name")
    if not isinstance(name, str):
        return False
    known = {tool.name for tool in tools}
    return name in known and isinstance(candidate.get("arguments"), (dict, str))


def _raise_for_status(status: int, body: str, base_url: str = "") -> None:
    """Translate an HTTP error into a message that says how to fix it."""
    detail = _error_detail(body)
    is_local = is_local_url(base_url)

    if status in (401, 403):
        if is_local:
            raise AuthenticationError(
                f"The server at {base_url} rejected the API key (HTTP {status}).\n"
                "Local servers such as Ollama normally ignore the key, so this "
                "is unusual. Another service may be listening on that port.\n"
                f"Server said: {detail}"
            )
        raise AuthenticationError(
            f"The provider rejected the API key (HTTP {status}).\n"
            "Check AGENTUX_API_KEY. If you changed providers, remember that "
            "each one has its own key.\n"
            f"Provider said: {detail}"
        )

    if status == 404:
        raise ModelNotFoundError(
            f"The model was not found (HTTP 404).\n"
            "Run 'agentux --list-models' to see what this server offers.\n"
            "For Ollama, pull the model first: ollama pull <name>\n"
            f"Server said: {detail}"
        )

    if status == 429:
        raise RateLimitError(
            f"Rate limited by the provider (HTTP 429). Wait a moment and retry. "
            f"Provider said: {detail}"
        )

    if status >= 500:
        raise ProviderError(
            f"The provider failed with HTTP {status}. This is a server-side "
            f"problem; retrying later may help. Provider said: {detail}"
        )

    if status == 400:
        # Ollama says this explicitly when the model lacks the tools capability.
        if "does not support tools" in detail.lower():
            raise UnsupportedToolCallingError(
                f"The model {self_model_from_detail(detail)!r} does not support "
                "tool calling.\n"
                "Some models are completion-only and cannot use tools at all. "
                "Pick one that lists 'tools' in its capabilities:\n"
                "  ollama list          # look for 'tools' in the capabilities\n"
                "or run a plain conversation with --no-tools.\n"
                f"Server said: {detail}"
            )
        raise ProviderError(
            f"The provider rejected the request (HTTP 400). Common causes: a "
            "model that cannot use tools, a context window that is too small "
            "for the conversation, or an unsupported parameter.\n"
            f"Server said: {detail}"
        )

    raise ProviderError(f"Provider returned HTTP {status}: {detail}")


def self_model_from_detail(detail: str) -> str:
    """Pull the model name out of an Ollama tool-support error."""
    marker = "library/"
    if marker in detail:
        return detail.split(marker, 1)[1].split()[0].strip()
    return detail.split()[0] if detail else "unknown"


class OpenAICompatibleProvider(BaseProvider):
    """Talks to any server implementing the OpenAI chat-completions API."""

    def _payload(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]],
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = _tool_schema(tools)
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True
            # Ask for a final usage chunk. Servers that do not support it ignore
            # the field, so this is safe to send everywhere.
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def supports_tool_calling(self) -> bool:
        return True

    def _parse_message(self, raw: Dict[str, Any]) -> Message:
        """Convert a raw assistant message into a :class:`Message`."""
        if not isinstance(raw, dict):
            raise MalformedResponseError(
                f"Expected an object for the assistant message, got {type(raw).__name__}."
            )

        tool_calls: List[ToolCall] = []
        for index, entry in enumerate(raw.get("tool_calls") or []):
            if not isinstance(entry, dict):
                raise MalformedResponseError(
                    f"Tool call #{index} was not an object: {entry!r}"
                )
            function = entry.get("function")
            if not isinstance(function, dict) or "name" not in function:
                raise MalformedResponseError(
                    f"Tool call #{index} is missing a function name: {entry!r}"
                )
            tool_calls.append(
                ToolCall(
                    id=str(entry.get("id") or f"call_{index}"),
                    name=str(function["name"]),
                    arguments=_parse_arguments(function.get("arguments")),
                )
            )

        return Message(
            role="assistant",
            content=raw.get("content") or None,
            tool_calls=tool_calls or None,
        )

    def _first_choice(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise MalformedResponseError(
                f"Provider returned {type(data).__name__}, expected a JSON object."
            )
        choices = data.get("choices")
        if not choices:
            raise MalformedResponseError(
                "Provider returned no choices. The model may have been rejected "
                f"or the response was empty. Full response: {json.dumps(data)[:500]}"
            )
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if message is None:
            raise MalformedResponseError(
                f"Provider choice had no 'message' field: {choices[0]!r}"
            )
        return message

    def _tool_call_from_deltas(self, deltas: Dict[int, Dict[str, Any]]) -> List[ToolCall]:
        """Rebuild whole tool calls from the fragments a stream delivers.

        A streamed tool call arrives in pieces: the first chunk carries ``id``
        and ``name``, later chunks carry more of the ``arguments`` JSON string.
        Each fragment is indexed, so they are reassembled in index order.
        """
        calls: List[ToolCall] = []
        for index in sorted(deltas):
            entry = deltas[index]
            function = entry.get("function") or {}
            name = function.get("name")
            if not name:
                # A call whose name never arrived is unusable; building one from
                # the arguments alone would be guessing at what to execute.
                continue
            calls.append(
                ToolCall(
                    id=str(entry.get("id") or f"call_{index}"),
                    name=str(name),
                    arguments=_parse_arguments(function.get("arguments")),
                )
            )
        return calls

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> Message:
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                response = await client.post(
                    self.config.chat_url,
                    headers=self._headers(),
                    json=self._payload(messages, tools, stream=False),
                )
            except httpx.TimeoutException as exc:
                raise ProviderError(
                    f"The provider did not respond within {self.config.timeout:g}s. "
                    "Raise AGENTUX_TIMEOUT if the model is simply slow, or check "
                    "that AGENTUX_BASE_URL is reachable."
                ) from exc
            except httpx.RequestError as exc:
                raise NetworkError(
                    f"Could not reach the provider at {self.config.base_url}: {exc}\n"
                    "Check your network connection and AGENTUX_BASE_URL."
                ) from exc

            if response.status_code >= 400:
                _raise_for_status(response.status_code, response.text, self.config.base_url)

            try:
                data = response.json()
            except ValueError as exc:
                raise MalformedResponseError(
                    "Provider returned a response that is not valid JSON. This "
                    "usually means the base URL points at a web page or proxy "
                    "rather than an API endpoint."
                ) from exc

        message = self._parse_message(self._first_choice(data))
        self.last_usage = usage_from_payload(data)

        if tools and not message.tool_calls:
            if getattr(self.config, "allow_text_tools", False):
                from ..text_tools import extract_text_tool_calls
                extracted_calls, cleaned = extract_text_tool_calls(message.content or "", tools)
                if extracted_calls:
                    message.tool_calls = extracted_calls
                    message.content = cleaned or None
                    return message

            if looks_like_emulated_tool_call(message.content, tools):
                raise UnsupportedToolCallingError(
                    f"{self.config.model} returned a tool call as plain text instead "
                    "of using native tool calling.\n\n"
                    "Agentux will not run tool calls that were printed as text: "
                    "parsing prose as a command is how an agent runs something it "
                    "was never asked to run. It is treated as an error instead.\n\n"
                    "This model advertises a 'tools' capability but its chat "
                    "template does not emit structured calls. Try a model whose "
                    "template supports tool use (run 'agentux --list-models'), "
                    "update the model with 'ollama pull', or enable --text-tools."
                )

        return message

    async def list_models(self) -> List[str]:
        """List the models this server offers.

        Works against the OpenAI-compatible ``/models`` endpoint. When the
        server is a local Ollama, the native ``/api/tags`` endpoint is also
        queried because it reports each model's capabilities -- which is how a
        user finds out whether a model can call tools at all.
        """
        names: List[str] = []
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                response = await client.get(
                    self.config.models_url, headers=self._headers()
                )
                if response.status_code < 400:
                    data = response.json()
                    for entry in data.get("data") or []:
                        name = entry.get("id")
                        if name:
                            names.append(str(name))
                elif response.status_code == 404:
                    raise ModelNotFoundError(
                        f"{self.config.base_url} does not expose a /models "
                        "endpoint, so Agentux cannot list models. Set "
                        "AGENTUX_MODEL explicitly."
                    )
                else:
                    _raise_for_status(
                        response.status_code, response.text, self.config.base_url
                    )
            except httpx.TimeoutException as exc:
                raise ProviderError(
                    f"Listing models timed out after {self.config.timeout:g}s."
                ) from exc
            except httpx.RequestError as exc:
                raise NetworkError(
                    f"Could not reach the provider at {self.config.base_url}: {exc}\n"
                    "Is Ollama running? Try: ollama serve"
                ) from exc
            except ValueError as exc:
                raise MalformedResponseError(
                    "The model list was not valid JSON; the base URL may point "
                    "at a web page rather than an API endpoint."
                ) from exc

        return names

    async def list_models_detailed(self) -> List[Dict[str, Any]]:
        """Model names plus Ollama capabilities, when the server reports them.

        Returns the same shape as :meth:`list_models` (names only, empty dict
        values) for servers that do not expose capabilities, so callers can use
        one code path.
        """
        names = await self.list_models()
        details: List[Dict[str, Any]] = [{"name": name} for name in names]

        if not self.config.is_local:
            return details

        # Ollama's native endpoint reports capabilities; try it, but do not
        # fail the command if this server is some other local provider.
        ollama_root = self.config.base_url.rstrip("/")
        if ollama_root.endswith("/v1"):
            ollama_root = ollama_root[: -len("/v1")]
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(f"{ollama_root}/api/tags")
                if response.status_code >= 400:
                    return details
                payload = response.json()
        except Exception:
            return details

        capabilities = {}
        for entry in payload.get("models") or []:
            name = entry.get("name") or entry.get("model")
            if name:
                capabilities[name] = entry.get("capabilities") or []

        for item in details:
            for name, caps in capabilities.items():
                if name == item["name"] or item["name"].startswith(name.split(":")[0]):
                    item["capabilities"] = caps
                    break
        return details

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas from a streaming completion.

        Tool calls are not streamed: a partial tool call cannot be executed
        safely. When the model starts calling a tool we stop yielding text and
        let the caller fall back to :meth:`chat`.
        """
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    self.config.chat_url,
                    headers=self._headers(),
                    json=self._payload(messages, tools, stream=True),
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        _raise_for_status(
                            response.status_code,
                            response.text_content,
                            self.config.base_url,
                        )
                    async for line in response.aiter_lines():
                        if not line or not line.startswith(_SSE_PREFIX):
                            continue
                        data = line[len(_SSE_PREFIX):].strip()
                        if not data or data == _SSE_DONE:
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if delta.get("tool_calls"):
                                # Partial tool calls are not executable.
                                return
                            piece = delta.get("content")
                            if piece:
                                yield piece
            except httpx.TimeoutException as exc:
                raise ProviderError(
                    f"The provider did not respond within {self.config.timeout:g}s "
                    "while streaming."
                ) from exc
            except httpx.RequestError as exc:
                raise NetworkError(
                    f"Could not reach the provider at {self.config.base_url}: {exc}"
                ) from exc

    async def chat_streaming(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        on_text: Optional[TextCallback] = None,
    ) -> Message:
        """One streaming request that returns a complete message.

        Text deltas go to ``on_text`` as they arrive; tool-call fragments are
        accumulated and parsed at the end. Both come from the same request, so
        a step that produces a tool call costs exactly what a step that
        produces prose costs.
        """
        pieces: List[str] = []
        # Keyed by the stream's own index field: a model may emit several tool
        # calls in one step, interleaved across chunks.
        deltas: Dict[int, Dict[str, Any]] = {}

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    self.config.chat_url,
                    headers=self._headers(),
                    json=self._payload(messages, tools, stream=True),
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        _raise_for_status(
                            response.status_code,
                            response.text_content,
                            self.config.base_url,
                        )
                    async for line in response.aiter_lines():
                        if not line or not line.startswith(_SSE_PREFIX):
                            continue
                        data = line[len(_SSE_PREFIX):].strip()
                        if not data or data == _SSE_DONE:
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        # A server supporting stream_options sends one final
                        # chunk with usage and no choices.
                        chunk_usage = usage_from_payload(chunk)
                        if chunk_usage:
                            self.last_usage = chunk_usage

                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}

                            for fragment in delta.get("tool_calls") or []:
                                if not isinstance(fragment, dict):
                                    continue
                                index = fragment.get("index")
                                index = index if isinstance(index, int) else len(deltas)
                                slot = deltas.setdefault(
                                    index, {"id": None, "function": {"name": None, "arguments": ""}}
                                )
                                if fragment.get("id"):
                                    slot["id"] = fragment["id"]
                                function = fragment.get("function") or {}
                                if function.get("name"):
                                    slot["function"]["name"] = (
                                        slot["function"]["name"] or ""
                                    ) + str(function["name"])
                                if function.get("arguments"):
                                    slot["function"]["arguments"] += str(
                                        function["arguments"]
                                    )

                            piece = delta.get("content")
                            if piece:
                                pieces.append(piece)
                                if on_text:
                                    await on_text(piece)
            except httpx.TimeoutException as exc:
                raise ProviderError(
                    f"The provider did not respond within {self.config.timeout:g}s "
                    "while streaming."
                ) from exc
            except httpx.RequestError as exc:
                raise NetworkError(
                    f"Could not reach the provider at {self.config.base_url}: {exc}"
                ) from exc

        tool_calls = self._tool_call_from_deltas(deltas)
        content = "".join(pieces) or None

        # When native tool calls are absent, check for text-based tool extraction
        # if allow_text_tools is enabled, else reject emulated calls in strict mode.
        if tools and not tool_calls:
            if getattr(self.config, "allow_text_tools", False):
                from ..text_tools import extract_text_tool_calls
                extracted_calls, cleaned = extract_text_tool_calls(content or "", tools)
                if extracted_calls:
                    return Message(role="assistant", content=cleaned or None, tool_calls=extracted_calls)

            if looks_like_emulated_tool_call(content or "", tools):
                raise UnsupportedToolCallingError(
                    f"{self.config.model} returned a tool call as plain text instead of "
                    "using native tool calling.\n\n"
                    "Agentux will not run tool calls that were printed as text: "
                    "parsing prose as a command is how an agent runs something it was "
                    "never asked to run. It is treated as an error instead.\n\n"
                    "This model advertises a 'tools' capability but its chat template "
                    "does not emit structured calls. Try a model whose template supports "
                    "tool use (run 'agentux --list-models'), update the model with "
                    "'ollama pull', or enable --text-tools."
                )

        if not content and not tool_calls:
            raise MalformedResponseError(
                "The provider streamed a response with neither content nor tool "
                "calls. The model may have returned nothing."
            )

        return Message(role="assistant", content=content, tool_calls=tool_calls or None)


__all__ = ["OpenAICompatibleProvider"]
