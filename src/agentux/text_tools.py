"""Text-based tool calling extraction and system prompt formatting.

Allows smaller or local models (e.g. qwen2.5-coder:3b, llama-3, phi-3) whose
chat templates do not emit structured OpenAI API tool_calls to invoke tools
reliably via clear, structured markdown or XML text blocks.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import ToolCall, ToolDefinition


TOOL_PROMPT_INSTRUCTIONS = """
## Tool Calling Format
You can invoke workspace tools directly. If your environment does not support native API tool calling, you MUST output your tool invocation using this exact, recognizable format:

### tool
*<tool_name>
```json
{
  "<argument_name>": "<argument_value>"
}
```

Or using standard tool call tags:
<tool_call>
{"name": "<tool_name>", "arguments": { ... }}
</tool_call>

When a tool finishes running, the system will return the output in this format:
### tool_result
*<tool_name>
<result>

You can then inspect the output and proceed to the next step or give your final answer.
"""

# Aliases for common shorthand operations
TOOL_ALIASES: Dict[str, Tuple[str, Optional[Dict[str, Any]]]] = {
    "read": ("filesystem", {"operation": "read"}),
    "write": ("filesystem", {"operation": "write"}),
    "edit": ("filesystem", {"operation": "edit"}),
    "ls": ("filesystem", {"operation": "ls"}),
    "mkdir": ("filesystem", {"operation": "mkdir"}),
    "bash": ("shell", None),
    "sh": ("shell", None),
    "run": ("shell", None),
}


def extract_text_tool_calls(
    content: str,
    tools: Optional[List[ToolDefinition]] = None,
) -> Tuple[List[ToolCall], str]:
    """Extract tool calls from text content.

    Returns:
        (tool_calls, cleaned_content)
    """
    if not content:
        return [], ""

    known_tools: Set[str] = {t.name for t in tools} if tools else set()
    tool_calls: List[ToolCall] = []
    cleaned: str = content

    # 1. User/Markdown format:
    # ### tool
    # *<name>
    # ```json
    # { ... }
    # ```
    pattern_md = re.compile(
        r"###\s*tool\s*\n\*\s*([a-zA-Z0-9_]+)\s*\n(?:```(?:json)?\s*)?(\{[\s\S]*?\})(?:\s*```)?",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern_md.finditer(content):
        raw_name = m.group(1).strip()
        raw_json = m.group(2).strip()
        name, default_args = _resolve_tool_name(raw_name, known_tools)
        if name:
            try:
                args = json.loads(raw_json)
                if not isinstance(args, dict):
                    args = {}
                if default_args:
                    args = {**default_args, **args}
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=name,
                        arguments=args,
                    )
                )
                cleaned = cleaned.replace(m.group(0), "")
            except Exception:
                pass

    # 2. XML tag format:
    # <tool_call>
    # {"name": "...", "arguments": { ... }}
    # </tool_call>
    pattern_xml = re.compile(
        r"<tool_call>\s*(?:```(?:json)?\s*)?(\{[\s\S]*?\})(?:\s*```)?\s*</tool_call>",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern_xml.finditer(content):
        raw_json = m.group(1).strip()
        try:
            data = json.loads(raw_json)
            if isinstance(data, dict):
                raw_name = data.get("name") or data.get("function")
                if isinstance(raw_name, dict):
                    raw_name = raw_name.get("name")
                if isinstance(raw_name, str):
                    name, default_args = _resolve_tool_name(raw_name, known_tools)
                    if name:
                        args = data.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        if not isinstance(args, dict):
                            args = {}
                        if default_args:
                            args = {**default_args, **args}
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:12]}",
                                name=name,
                                arguments=args,
                            )
                        )
                        cleaned = cleaned.replace(m.group(0), "")
        except Exception:
            pass

    # 3. Fenced code block or bare JSON:
    # {"name": "filesystem", "arguments": { ... }}
    if not tool_calls:
        text_strip = content.strip()
        # Check if entire content is bare JSON or fenced JSON
        json_candidate = text_strip
        if json_candidate.startswith("```"):
            lines = json_candidate.split("\n")
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                json_candidate = "\n".join(lines[1:-1]).strip()

        if json_candidate.startswith("{") and json_candidate.endswith("}"):
            try:
                parsed = json.loads(json_candidate)
                if isinstance(parsed, dict):
                    candidate = parsed.get("tool_call") if isinstance(parsed.get("tool_call"), dict) else parsed
                    raw_name = candidate.get("name") or candidate.get("function")
                    if isinstance(raw_name, dict):
                        raw_name = raw_name.get("name")
                    if isinstance(raw_name, str):
                        name, default_args = _resolve_tool_name(raw_name, known_tools)
                        if name:
                            args = candidate.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    args = {}
                            if not isinstance(args, dict):
                                args = {}
                            if default_args:
                                args = {**default_args, **args}
                            tool_calls.append(
                                ToolCall(
                                    id=f"call_{uuid.uuid4().hex[:12]}",
                                    name=name,
                                    arguments=args,
                                )
                            )
                            cleaned = ""
            except Exception:
                pass

    return tool_calls, cleaned.strip()


def _resolve_tool_name(
    raw_name: str,
    known_tools: Set[str],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Resolve tool name against known registered tools and shorthand aliases."""
    name_clean = raw_name.strip()
    if name_clean in known_tools:
        return name_clean, None

    # Check lowercase alias
    alias_match = TOOL_ALIASES.get(name_clean.lower())
    if alias_match:
        target, def_args = alias_match
        if target in known_tools:
            return target, def_args

    return None, None


__all__ = ["extract_text_tool_calls", "TOOL_PROMPT_INSTRUCTIONS", "TOOL_ALIASES"]
