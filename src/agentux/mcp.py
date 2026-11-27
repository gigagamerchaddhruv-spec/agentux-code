"""A minimal MCP client speaking JSON-RPC 2.0 over stdio.

Written directly against the protocol rather than depending on the ``mcp``
package, so Agentux keeps its small dependency list and its MCP support is
optional rather than load-bearing. The core agent never imports this module.

Transport, per the MCP stdio specification:

* One JSON-RPC message per line, newline-delimited, UTF-8. No ``Content-Length``
  headers -- that is LSP framing, and it is not used here.
* A server may write anything to stderr. It is drained separately and never
  parsed; a chatty server is not a broken server.
* The child's environment is an allowlist plus the config's ``env``. Agentux
  does not hand over its own environment, which would leak ``AGENTUX_API_KEY``
  to every server on the machine.
* Servers are launched without a shell, in their own process group, so a
  wrapper like ``npx`` cannot inject shell syntax and can be killed as a tree.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

from .errors import MCPError, MCPProtocolError
from .mcp_config import MCPServerConfig

#: The version we advertise. Servers answer with their own newest supported
#: version, so this is a request, not a demand.
PROTOCOL_VERSION = "2025-11-25"

#: Versions we are willing to talk to.
SUPPORTED_PROTOCOL_VERSIONS = {
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
}

#: How long to wait for a handshake, or for any single request.
DEFAULT_TIMEOUT = 60.0

#: Grace period between closing stdin and escalating to a signal.
TERMINATION_GRACE = 2.0

#: Bound on a single stdout line. A server that emits more than this in one
#: message is either broken or hostile; either way, stop buffering it.
MAX_LINE_BYTES = 10 * 1024 * 1024

#: Bound on tools/list pagination, so a server that returns a non-advancing
#: cursor cannot hang the startup path.
MAX_LIST_PAGES = 100

#: Only these variables reach the server process. Notably absent: AGENTUX_API_KEY
#: and anything else Agentux-specific or secret-bearing.
INHERITED_ENV = (
    ("APPDATA", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH", "PATHEXT",
     "PROCESSOR_ARCHITECTURE", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP",
     "USERNAME", "USERPROFILE", "COMSPEC", "NUMBER_OF_PROCESSORS", "OS")
    if sys.platform == "win32"
    else ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER", "TMPDIR", "LANG", "LC_ALL")
)


def _resolve_command(command: str) -> str:
    """Resolve a command to something ``execve`` can actually run.

    On POSIX a bare name is fine. On Windows ``npx`` is really ``npx.cmd`` and
    cannot be executed directly without a shell, which is not an option here --
    so look for the script extensions explicitly.
    """
    resolved = shutil.which(command)
    if resolved:
        return resolved
    if sys.platform == "win32":
        for extension in (".cmd", ".bat", ".exe", ".ps1"):
            candidate = shutil.which(command + extension)
            if candidate:
                return candidate
    return command


def _build_env(overrides: Optional[Dict[str, str]]) -> Dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in INHERITED_ENV and not value.startswith("()")
    }
    if overrides:
        env.update({str(k): str(v) for k, v in overrides.items()})
    return env


class MCPServer:
    """One MCP server subprocess."""

    def __init__(
        self,
        config: MCPServerConfig,
        timeout: float = DEFAULT_TIMEOUT,
        on_log: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.timeout = timeout
        self.on_log = on_log
        self.process: Optional[asyncio.subprocess.Process] = None
        self.server_info: Dict[str, Any] = {}
        self.protocol_version: Optional[str] = None
        self.instructions: Optional[str] = None
        self._pending: Dict[Any, asyncio.Future] = {}
        self._next_id = 0
        self._reader: Optional[asyncio.Task] = None
        self._stderr_reader: Optional[asyncio.Task] = None
        self._stderr_tail: List[str] = []
        self._closed = False
        self._failure: Optional[str] = None

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.config.name

    @property
    def connected(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def _log(self, message: str) -> None:
        if self.on_log:
            try:
                self.on_log(f"[{self.name}] {message}")
            except Exception:
                pass

    async def start(self) -> None:
        """Spawn the process and complete the initialize handshake."""
        if self.process is not None:
            return

        command = _resolve_command(self.config.command)
        kwargs: Dict[str, Any] = {
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "env": _build_env(self.config.env),
            "cwd": self.config.cwd or None,
        }
        # Own process group so the whole tree can be signalled, and no shell so
        # arguments cannot be reinterpreted.
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(__import__("subprocess"), "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True

        try:
            self.process = await asyncio.create_subprocess_exec(
                command, *self.config.args, **kwargs
            )
        except FileNotFoundError as exc:
            raise MCPError(
                f"MCP server {self.name!r}: command {self.config.command!r} was not found. "
                "Check that it is installed and on your PATH."
            ) from exc
        except OSError as exc:
            raise MCPError(
                f"MCP server {self.name!r}: could not start {self.config.display}: {exc}"
            ) from exc

        self._reader = asyncio.ensure_future(self._read_stdout())
        self._stderr_reader = asyncio.ensure_future(self._read_stderr())

        try:
            await self._handshake()
        except Exception:
            await self.close()
            raise

    async def _handshake(self) -> None:
        result = await self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agentux", "version": _agentux_version()},
            },
        )

        negotiated = result.get("protocolVersion")
        self.protocol_version = negotiated
        self.server_info = result.get("serverInfo") or {}
        self.instructions = result.get("instructions")

        if negotiated not in SUPPORTED_PROTOCOL_VERSIONS:
            # The spec says disconnect here. We log loudly and continue instead:
            # the handshake already succeeded, so the server understands us,
            # and refusing would cost the user their tools over a version label
            # we do not recognise.
            self._log(
                f"server requested protocol {negotiated!r}, which this build does "
                f"not know about; continuing anyway"
            )

        # A notification: no id, no params, and no reply. Sending a reply here
        # is a protocol violation that servers silently ignore or reject.
        await self.notify("notifications/initialized")

    async def close(self) -> None:
        """Shut the server down: close stdin, then SIGTERM, then SIGKILL.

        Closing stdin is the portable graceful-shutdown signal in MCP; there is
        no shutdown message.
        """
        self._closed = True
        process = self.process

        for task in (self._reader, self._stderr_reader):
            if task is not None and not task.done():
                task.cancel()

        if process is None:
            self._fail_pending(MCPError(f"MCP server {self.name!r} was never started."))
            return

        if process.returncode is None:
            try:
                if process.stdin is not None and not process.stdin.is_closing():
                    process.stdin.close()
            except (OSError, RuntimeError):
                pass

            try:
                await asyncio.wait_for(process.wait(), timeout=TERMINATION_GRACE)
            except asyncio.TimeoutError:
                self._signal(process, force=False)
                try:
                    await asyncio.wait_for(process.wait(), timeout=TERMINATION_GRACE)
                except asyncio.TimeoutError:
                    self._signal(process, force=True)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=TERMINATION_GRACE)
                    except asyncio.TimeoutError:
                        self._log("did not exit after SIGKILL")

        self._fail_pending(
            MCPError(f"MCP server {self.name!r} is closed; cannot send new requests.")
        )

    def _signal(self, process: asyncio.subprocess.Process, force: bool) -> None:
        import signal

        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            if sys.platform == "win32":
                process.terminate() if not force else process.kill()
            elif hasattr(os, "killpg"):
                # The process group, not just the leader: `npx` spawns a
                # grandchild node process that would otherwise survive and hold
                # the stdout pipe open forever.
                os.killpg(os.getpgid(process.pid), sig)
            else:
                process.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    # ------------------------------------------------------------------
    # Wire I/O
    # ------------------------------------------------------------------
    async def _read_stderr(self) -> None:
        """Drain stderr so the child never blocks on a full pipe.

        Content is kept only as a short tail for error messages. Per the spec,
        stderr output does not indicate an error condition.
        """
        stream = self.process.stderr if self.process else None
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._stderr_tail.append(text)
                    del self._stderr_tail[:-20]  # keep the last 20 lines
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _read_stdout(self) -> None:
        """Read newline-delimited JSON-RPC messages until EOF."""
        stream = self.process.stdout if self.process else None
        if stream is None:
            return

        try:
            while True:
                try:
                    line = await stream.readuntil(b"\n")
                except asyncio.IncompleteReadError as exc:
                    line = exc.partial
                    if not line:
                        break
                except asyncio.LimitOverrunError:
                    # A single message larger than the reader's buffer. It is
                    # not recoverable by continuing to parse this stream.
                    self._fail_pending(
                        MCPError(
                            f"MCP server {self.name!r} sent a message larger than "
                            f"{MAX_LINE_BYTES // (1024 * 1024)} MiB."
                        )
                    )
                    break
                except (ValueError, asyncio.CancelledError):
                    break

                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    # npx and hot-reload wrappers print banners to stdout.
                    # Skipping the line is what every SDK does, and it is the
                    # difference between working and not.
                    continue

                if isinstance(message, dict):
                    self._dispatch(message)

                if self.process is not None and self.process.stdout.at_eof():
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = str(exc)
        finally:
            # Reap first: the reason is built from the exit code, and it is
            # not populated yet at EOF. A cancelled reader (shutdown) skips the
            # wait -- there is nobody left to explain anything to.
            if not self._closed:
                await self._await_exit()
            reason = self._failure or (
                f"MCP server {self.name!r} closed its output stream"
                + f" ({self._describe_exit()})"
            )
            self._fail_pending(MCPError(reason))

    async def _await_exit(self) -> None:
        """Wait for the child to be reaped, so its exit code is readable.

        EOF on stdout arrives just before the process is reaped, so at the
        moment the reader gives up the return code is usually still ``None``.
        Waiting here is what makes the diagnostic say *why* the server died
        instead of only that it died -- without it the exit code appears or
        not depending on how the two events happened to interleave.
        """
        process = self.process
        if process is None or process.returncode is not None:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=TERMINATION_GRACE)
        except (asyncio.TimeoutError, asyncio.CancelledError, ProcessLookupError):
            # Still running, or the loop is shutting down. The message below
            # falls back to saying nothing was learned, which is honest.
            pass

    def _describe_exit(self) -> str:
        """The exit status to report, once the process has been reaped."""
        process = self.process
        if process is None or process.returncode is None:
            return "the process had not exited"
        return f"exit code {process.returncode}"

    def _dispatch(self, message: Dict[str, Any]) -> None:
        """Route a response to whoever is waiting on its id."""
        if "id" not in message or message.get("method"):
            return  # A notification; we do not subscribe to any.

        future = self._pending.pop(message["id"], None)
        if future is None or future.done():
            return

        if "error" in message:
            error = message["error"] or {}
            code = error.get("code", "unknown")
            detail = error.get("message", "no message")
            extra = error.get("data")
            future.set_exception(
                MCPProtocolError(
                    f"MCP server {self.name!r} returned error {code}: {detail}"
                    + (f" ({json.dumps(extra)[:300]})" if extra else "")
                )
            )
            return

        future.set_result(message.get("result") or {})

    def _fail_pending(self, error: BaseException) -> None:
        """Fail every in-flight request. They can never be answered now."""
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def _send(self, payload: Dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.returncode is not None:
            tail = "\n".join(self._stderr_tail[-5:])
            raise MCPError(
                f"MCP server {self.name!r} is not running"
                + (f". Last output: {tail}" if tail else ".")
            )
        try:
            process.stdin.write(
                (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            )
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise MCPError(f"MCP server {self.name!r} went away: {exc}") from exc

    async def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a request and wait for its response."""
        if self._closed:
            raise MCPError(f"MCP server {self.name!r} is closed.")

        self._next_id += 1
        request_id = self._next_id
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future

        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params

        try:
            await self._send(payload)
        except Exception:
            self._pending.pop(request_id, None)
            raise

        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise MCPError(
                f"MCP server {self.name!r} did not answer {method!r} within "
                f"{self.timeout:g}s."
            ) from None
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            raise

    async def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a notification. Nothing comes back."""
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self._send(payload)

    # ------------------------------------------------------------------
    # Protocol operations
    # ------------------------------------------------------------------
    async def list_tools(self) -> List[Dict[str, Any]]:
        """Return every tool the server offers, following pagination.

        ``nextCursor`` is optional; a server that returns everything at once
        simply omits it, so its absence -- not its presence -- ends the loop.
        """
        tools: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        for _ in range(MAX_LIST_PAGES):
            result = await self.request("tools/list", {"cursor": cursor} if cursor else {})
            page = result.get("tools")
            if not isinstance(page, list):
                raise MCPProtocolError(
                    f"MCP server {self.name!r}: tools/list returned no 'tools' array."
                )
            tools.extend(t for t in page if isinstance(t, dict))

            cursor = result.get("nextCursor")
            if not cursor:
                break
        else:
            self._log(f"stopped after {MAX_LIST_PAGES} pages of tools/list")

        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool.

        Both failure modes are distinct and are kept distinct. A JSON-RPC error
        means the *request* was wrong and is raised. ``isError: true`` means the
        tool ran and failed, and is returned to the caller so the model can read
        the message and correct itself.
        """
        result = await self.request("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise MCPProtocolError(
                f"MCP server {self.name!r}: tools/call returned {type(result).__name__}, "
                "expected an object."
            )
        return result


def render_tool_result(result: Dict[str, Any], max_chars: int = 60_000) -> str:
    """Flatten an MCP tool result into the text the model will read."""
    blocks = result.get("content")
    if not isinstance(blocks, list):
        # A server that returns only structuredContent is legal.
        structured = result.get("structuredContent")
        if structured is not None:
            return json.dumps(structured, indent=2, default=str)[:max_chars]
        return "(the server returned no content)"

    parts: List[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "image":
            parts.append(f"[image: {block.get('mimeType', 'unknown type')}]")
        elif kind == "audio":
            parts.append(f"[audio: {block.get('mimeType', 'unknown type')}]")
        elif kind == "resource":
            resource = block.get("resource") or {}
            body = resource.get("text")
            parts.append(
                f"[resource: {resource.get('uri', 'unknown')}]\n{body}"
                if body
                else f"[resource: {resource.get('uri', 'unknown')}]"
            )
        elif kind == "resource_link":
            parts.append(f"[link: {block.get('uri', 'unknown')}]")
        else:
            parts.append(f"[{kind}: {json.dumps(block, default=str)[:200]}]")

    text = "\n".join(p for p in parts if p).strip()
    if not text:
        return "(the server returned no text content)"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [{len(text) - max_chars} characters truncated]"
    return text


def _agentux_version() -> str:
    from . import __version__

    return __version__


__all__ = [
    "MCPServer",
    "render_tool_result",
    "PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
]
