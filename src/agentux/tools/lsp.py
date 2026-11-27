"""Language Server Protocol (LSP) integration.

Provides semantic code navigation tools (go to definition, find references,
and get symbol info) by talking to a language server over stdio.

The agent uses this instead of `grep` when it needs to know what a symbol
*means* rather than where its name appears. A grep match can be a comment,
a string, or a shadowed variable; an LSP response is the compiler's own
understanding of the code.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import threading
from typing import Any, Dict, List, Optional

from ..errors import ToolError
from .base import BaseTool


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 over stdio (LSP framing: Content-Length headers)
# ---------------------------------------------------------------------------

class _LSPClient:
    """A minimal synchronous JSON-RPC 2.0 client for an LSP server process.

    The reader runs in a dedicated daemon thread doing blocking ``os.read``
    on the server's stdout pipe; the writer is the caller's thread. Responses
    are delivered back through ``threading.Event`` objects so ``request()``
    can block synchronously. This is the same pattern as the proven sync test
    and avoids the fragility of ``loop.connect_read_pipe`` on CPython 3.14,
    where the pipe transport can silently stop delivering data.

    The async tools wrap calls to this client with ``asyncio.to_thread``.
    """

    def __init__(self, command: List[str], root: str) -> None:
        self.command = command
        self.root = root
        self._proc: Optional[subprocess.Popen] = None
        self._req_id = 0
        self._buf = b""
        self._responses: Dict[int, Tuple[threading.Event, Dict[str, Any]]] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self._closed = False

    def start(self) -> None:
        if self._proc is not None:
            return
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("AGENTUX_")
        }
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env=env,
        )
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True
        )
        self._reader_thread.start()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        try:
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                self._buf += chunk
                # Parse as many complete LSP messages as are available.
                while True:
                    msg, self._buf = self._extract_message(self._buf)
                    if msg is None:
                        break
                    try:
                        resp = json.loads(msg)
                    except Exception:
                        continue
                    if "id" in resp and resp["id"] is not None:
                        entry = self._responses.get(resp["id"])
                        if entry is not None:
                            entry[1]["resp"] = resp
                            entry[0].set()
        except OSError:
            pass

    @staticmethod
    def _extract_message(buf: bytes) -> tuple:
        """Return (message_bytes, remaining_buf) or (None, buf)."""
        header_end = buf.find(b"\r\n\r\n")
        if header_end < 0:
            return None, buf
        header = buf[:header_end].decode("utf-8", "replace")
        length = None
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                try:
                    length = int(line.split(":", 1)[1].strip())
                except ValueError:
                    return None, buf
        if length is None:
            return None, buf[header_end + 4:]
        body_start = header_end + 4
        if len(buf) < body_start + length:
            return None, buf
        return buf[body_start:body_start + length], buf[body_start + length:]

    def _write(self, body: bytes) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self._proc.stdin.write(header + body)
        self._proc.stdin.flush()

    def _send(self, method: str, params: Any, expect_response: bool = True) -> Any:
        self.start()
        request_id = None
        if expect_response:
            self._req_id += 1
            request_id = self._req_id
        # Notifications must OMIT the id field entirely. Sending
        # ``"id": null`` is rejected by pyright ("Received message which is
        # neither a response nor a notification"), which silently drops the
        # handshake and leaves later requests hanging.
        if expect_response:
            body = json.dumps({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }).encode("utf-8")
        else:
            body = json.dumps({
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
            }).encode("utf-8")
        self._write(body)

        if not expect_response:
            return None

        event = threading.Event()
        holder: Dict[str, Any] = {}
        self._responses[request_id] = (event, holder)
        event.wait(timeout=30.0)
        self._responses.pop(request_id, None)
        if not event.is_set():
            raise ToolError(f"LSP request '{method}' timed out")
        resp = holder.get("resp", {})
        if resp.get("error"):
            raise ToolError(f"LSP error ({resp['error'].get('code')}): {resp['error'].get('message')}")
        return resp.get("result")

    def notify(self, method: str, params: Any) -> None:
        self._send(method, params, expect_response=False)

    def request(self, method: str, params: Any) -> Any:
        return self._send(method, params, expect_response=True)

    def did_open(self, path: str) -> None:
        """Notify the server that a document is open.

        Several servers (pyright, rust-analyzer) will not answer
        ``textDocument/documentSymbol`` or ``textDocument/definition`` for a
        file they have not been told is open. We send ``didOpen`` once per
        path and ignore failures -- a server that does not care simply
        ignores the notification.
        """
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception:
            return
        try:
            self._write(json.dumps({
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file://" + path,
                        "languageId": _language_id_for(path),
                        "version": 1,
                        "text": text,
                    },
                },
            }).encode("utf-8"))
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            try:
                self._send("shutdown", None)
            except Exception:
                pass
            try:
                self.notify("exit", {})
            except Exception:
                pass
            try:
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            try:
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1)


# ---------------------------------------------------------------------------
# Server registry
# ---------------------------------------------------------------------------

#: Map of file extension -> (language id, command, root marker files)
LSP_SERVERS: Dict[str, Dict[str, Any]] = {
    ".py": {
        "language": "python",
        "command": ["pyright-langserver", "--stdio"],
        "fallback": [["pyls"], ["pylsp"]],
        "root_markers": ["pyproject.toml", "setup.py", "requirements.txt", "setup.cfg"],
    },
    ".go": {
        "language": "go",
        "command": ["gopls", "serve"],
        "fallback": [],
        "root_markers": ["go.mod", "go.sum"],
    },
    ".ts": {
        "language": "typescript",
        "command": ["typescript-language-server", "--stdio"],
        "fallback": [],
        "root_markers": ["tsconfig.json", "package.json"],
    },
    ".tsx": {
        "language": "typescript",
        "command": ["typescript-language-server", "--stdio"],
        "fallback": [],
        "root_markers": ["tsconfig.json", "package.json"],
    },
    ".js": {
        "language": "javascript",
        "command": ["typescript-language-server", "--stdio"],
        "fallback": [],
        "root_markers": ["package.json"],
    },
    ".jsx": {
        "language": "javascript",
        "command": ["typescript-language-server", "--stdio"],
        "fallback": [],
        "root_markers": ["package.json"],
    },
    ".rs": {
        "language": "rust",
        "command": ["rust-analyzer"],
        "fallback": [],
        "root_markers": ["Cargo.toml"],
    },
    ".java": {
        "language": "java",
        "command": ["jdtls"],
        "fallback": [],
        "root_markers": ["pom.xml", "build.gradle"],
    },
    ".c": {
        "language": "c",
        "command": ["clangd"],
        "fallback": [],
        "root_markers": ["compile_commands.json", "Makefile", "CMakeLists.txt"],
    },
    ".cpp": {
        "language": "cpp",
        "command": ["clangd"],
        "fallback": [],
        "root_markers": ["compile_commands.json", "Makefile", "CMakeLists.txt"],
    },
}


def _language_id_for(path: str) -> str:
    """Return the LSP ``languageId`` for a file based on its extension."""
    ext = os.path.splitext(path)[1].lower()
    spec = LSP_SERVERS.get(ext)
    if spec is None:
        return "text"
    return spec.get("language", "text")


def _find_server_root(root: str, markers: List[str]) -> str:
    """Walk up from `root` looking for one of the marker files."""
    current = root
    for _ in range(10):
        for marker in markers:
            if os.path.exists(os.path.join(current, marker)):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return root


def _resolve_binary(command: List[str]) -> Optional[List[str]]:
    """Return the command if the binary is available, else None."""
    if not command:
        return None
    if shutil.which(command[0]) is None:
        return None
    return command


def _try_resolve(spec: Dict[str, Any]) -> Optional[List[str]]:
    """Try the primary command, then each fallback, returning the first that exists."""
    primary = _resolve_binary(spec.get("command") or [])
    if primary is not None:
        return primary
    for fallback in spec.get("fallback", []) or []:
        if isinstance(fallback, list):
            resolved = _resolve_binary(fallback)
        else:
            resolved = _resolve_binary([fallback, "--stdio"])
        if resolved is not None:
            return resolved
    return None


class LSPManager:
    """Manages LSP server processes and routes requests to them."""

    def __init__(self, root: str) -> None:
        self.root = os.path.realpath(root)
        self._clients: Dict[str, _LSPClient] = {}

    def _language_for(self, path: str) -> Optional[str]:
        ext = os.path.splitext(path)[1].lower()
        spec = LSP_SERVERS.get(ext)
        if spec is None:
            return None
        return spec["language"]

    def _spec_for(self, path: str) -> Optional[Dict[str, Any]]:
        ext = os.path.splitext(path)[1].lower()
        return LSP_SERVERS.get(ext)

    async def _get_or_start(self, path: str) -> _LSPClient:
        ext = os.path.splitext(path)[1].lower()
        spec = LSP_SERVERS.get(ext)
        if spec is None:
            raise ToolError(f"No LSP server configured for extension '{ext}'")

        lang = spec["language"]
        if lang in self._clients:
            return self._clients[lang]

        command = _try_resolve(spec)
        if command is None:
            raise ToolError(
                f"No LSP server binary found for {ext} ({lang}). "
                f"Install one of: {', '.join(spec['command'] + spec.get('fallback', []))}"
            )

        server_root = _find_server_root(self.root, spec.get("root_markers", []))
        client = _LSPClient(command, server_root)
        await asyncio.to_thread(client.start)

        # Initialize
        await asyncio.to_thread(client.request, "initialize", {
            "processId": os.getpid(),
            "rootUri": "file://" + server_root,
            "capabilities": {
                "textDocument": {
                    "completion": {"completionItem": {"snippetSupport": True}},
                },
            },
        })
        await asyncio.to_thread(client.notify, "initialized", {})

        self._clients[lang] = client
        return client

    async def request(self, path: str, method: str, params: Any) -> Any:
        client = await self._get_or_start(path)
        return await asyncio.to_thread(client.request, method, params)

    async def close_all(self) -> None:
        for client in list(self._clients.values()):
            await asyncio.to_thread(client.close)
        self._clients.clear()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _location_to_str(loc: Dict[str, Any], root: str) -> str:
    """Convert an LSP Location to a human-readable path:line:col."""
    uri = loc.get("uri", "")
    path = uri.replace("file://", "")
    if path.startswith(root):
        path = path[len(root):].lstrip("/")
    rng = loc.get("range", {})
    start = rng.get("start", {})
    line = start.get("line", 0) + 1
    char = start.get("character", 0) + 1
    return f"{path}:{line}:{char}"


class GoToDefinitionTool(BaseTool):
    """Jump to the definition of a symbol using a language server."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = os.path.realpath(root or os.getcwd())
        self._manager: Optional[LSPManager] = None

    @property
    def name(self) -> str:
        return "go_to_definition"

    @property
    def description(self) -> str:
        return (
            "Find the definition of a symbol at a given position using a "
            "language server. Returns the file, line, and column where the "
            "symbol is defined. Much more accurate than searching for the name."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the source file containing the symbol.",
                },
                "line": {
                    "type": "integer",
                    "description": "1-indexed line number of the symbol.",
                },
                "character": {
                    "type": "integer",
                    "description": "1-indexed character offset of the symbol.",
                },
            },
            "required": ["path", "line", "character"],
        }

    def _get_manager(self) -> LSPManager:
        if self._manager is None:
            self._manager = LSPManager(self.root)
        return self._manager

    async def execute(self, path: str, line: int, character: int) -> str:
        full_path = path if os.path.isabs(path) else os.path.join(self.root, path)
        full_path = os.path.realpath(full_path)
        if not full_path.startswith(self.root):
            raise ToolError(f"Path escapes workspace directory: {path}")
        if not os.path.isfile(full_path):
            raise ToolError(f"File not found: {path}")

        manager = self._get_manager()
        client = await manager._get_or_start(full_path)
        await asyncio.to_thread(client.did_open, full_path)
        try:
            result = await asyncio.to_thread(
                client.request,
                "textDocument/definition",
                {
                    "textDocument": {"uri": "file://" + full_path},
                    "position": {"line": line - 1, "character": character - 1},
                },
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"LSP definition failed: {exc}")

        if not result:
            return f"No definition found for symbol at {path}:{line}:{character}."

        if isinstance(result, list):
            if not result:
                return f"No definition found for symbol at {path}:{line}:{character}."
            locations = [_location_to_str(loc, self.root) for loc in result]
            return "Definitions:\n" + "\n".join(locations)

        if isinstance(result, dict) and "uri" in result:
            return "Definition: " + _location_to_str(result, self.root)

        return f"Definition (raw): {json.dumps(result)}"


class FindReferencesTool(BaseTool):
    """Find all references to a symbol using a language server."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = os.path.realpath(root or os.getcwd())
        self._manager: Optional[LSPManager] = None

    @property
    def name(self) -> str:
        return "find_references"

    @property
    def description(self) -> str:
        return (
            "Find all references to a symbol at a given position using a "
            "language server. Returns a list of locations where the symbol "
            "is used. Useful before renaming or refactoring."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the source file containing the symbol.",
                },
                "line": {
                    "type": "integer",
                    "description": "1-indexed line number of the symbol.",
                },
                "character": {
                    "type": "integer",
                    "description": "1-indexed character offset of the symbol.",
                },
            },
            "required": ["path", "line", "character"],
        }

    def _get_manager(self) -> LSPManager:
        if self._manager is None:
            self._manager = LSPManager(self.root)
        return self._manager

    async def execute(self, path: str, line: int, character: int) -> str:
        full_path = path if os.path.isabs(path) else os.path.join(self.root, path)
        full_path = os.path.realpath(full_path)
        if not full_path.startswith(self.root):
            raise ToolError(f"Path escapes workspace directory: {path}")
        if not os.path.isfile(full_path):
            raise ToolError(f"File not found: {path}")

        manager = self._get_manager()
        client = await manager._get_or_start(full_path)
        await asyncio.to_thread(client.did_open, full_path)
        try:
            result = await asyncio.to_thread(
                client.request,
                "textDocument/references",
                {
                    "textDocument": {"uri": "file://" + full_path},
                    "position": {"line": line - 1, "character": character - 1},
                    "context": {"includeDeclaration": True},
                },
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"LSP references failed: {exc}")

        if not result:
            return f"No references found for symbol at {path}:{line}:{character}."

        locations = [_location_to_str(loc, self.root) for loc in result]
        header = f"Found {len(locations)} reference(s) for symbol at {path}:{line}:{character}:"
        return header + "\n" + "\n".join(locations)


class GetSymbolInfoTool(BaseTool):
    """Get the document symbols (outline) of a file using a language server."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = os.path.realpath(root or os.getcwd())
        self._manager: Optional[LSPManager] = None

    @property
    def name(self) -> str:
        return "get_symbol_info"

    @property
    def description(self) -> str:
        return (
            "Get the list of symbols (functions, classes, variables, etc.) "
            "in a file using a language server. Returns a structured outline "
            "with names, kinds, and line numbers."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the source file to inspect.",
                },
            },
            "required": ["path"],
        }

    def _get_manager(self) -> LSPManager:
        if self._manager is None:
            self._manager = LSPManager(self.root)
        return self._manager

    async def execute(self, path: str) -> str:
        full_path = path if os.path.isabs(path) else os.path.join(self.root, path)
        full_path = os.path.realpath(full_path)
        if not full_path.startswith(self.root):
            raise ToolError(f"Path escapes workspace directory: {path}")
        if not os.path.isfile(full_path):
            raise ToolError(f"File not found: {path}")

        manager = self._get_manager()
        client = await manager._get_or_start(full_path)
        await asyncio.to_thread(client.did_open, full_path)
        try:
            result = await asyncio.to_thread(
                client.request,
                "textDocument/documentSymbol",
                {"textDocument": {"uri": "file://" + full_path}},
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"LSP documentSymbol failed: {exc}")

        if not result:
            return f"No symbols found in {path}."

        # Handle both flat DocumentSymbol[] and flat SymbolInformation[]
        kind_names = {
            1: "File", 2: "Module", 3: "Namespace", 4: "Package", 5: "Class",
            6: "Method", 7: "Property", 8: "Field", 9: "Constructor", 10: "Enum",
            11: "Interface", 12: "Function", 13: "Variable", 14: "Constant",
            15: "String", 16: "Number", 17: "Boolean", 18: "Array", 19: "Object",
            20: "Key", 21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
            25: "Operator", 26: "TypeParameter",
        }

        lines = [f"Symbols in {path}:"]
        if isinstance(result, list):
            for sym in result:
                if "location" in sym:
                    # DocumentSymbol (has children)
                    name = sym.get("name", "?")
                    kind = kind_names.get(sym.get("kind", 0), "Unknown")
                    loc = sym["location"]["range"]["start"]
                    lines.append(f"  L{loc['line'] + 1:<5} [{kind:<12}] {name}")
                    for child in sym.get("children", []) or []:
                        cname = child.get("name", "?")
                        ckind = kind_names.get(child.get("kind", 0), "Unknown")
                        cloc = child["location"]["range"]["start"]
                        lines.append(f"    L{cloc['line'] + 1:<5} [{ckind:<12}] {cname}")
                else:
                    # SymbolInformation (flat)
                    name = sym.get("name", "?")
                    kind = kind_names.get(sym.get("kind", 0), "Unknown")
                    loc = sym.get("location", {}).get("range", {}).get("start", {})
                    lines.append(f"  L{loc.get('line', 0) + 1:<5} [{kind:<12}] {name}")

        return "\n".join(lines)


__all__ = [
    "GoToDefinitionTool",
    "FindReferencesTool",
    "GetSymbolInfoTool",
    "LSPManager",
]