"""Embedded Web UI Console for Agentux.

Provides a modern, zero-npm browser developer console with real-time
Server-Sent Events (SSE) streaming, interactive tool execution cards,
human-in-the-loop approvals, and workspace file navigation.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from .agent import Agent
from .config import Config


class AgentuxWebHandler(SimpleHTTPRequestHandler):
    """HTTP and SSE request handler for Agentux Web Console."""

    agent: Agent
    config: Config
    static_dir: str

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/status":
            self._send_json({
                "workspace": self.config.working_dir,
                "model": self.config.model,
                "theme": "emerald",
                "tools_count": len(self.agent.registry),
            })
            return

        if path == "/api/tools":
            tools_data = [
                {"name": t.name, "description": t.description}
                for t in self.agent.registry.list_tools()
            ]
            self._send_json({"tools": tools_data})
            return

        if path == "/api/files":
            files_list = []
            for root, dirs, files in os.walk(self.config.working_dir):
                # Filter ignore dirs
                dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache"}]
                rel_root = os.path.relpath(root, self.config.working_dir)
                for f in files:
                    rel_file = f if rel_root == "." else os.path.join(rel_root, f)
                    files_list.append(rel_file)
                if len(files_list) > 500:
                    break
            self._send_json({"files": sorted(files_list)})
            return

        if path == "/api/file":
            target = query.get("path", [""])[0]
            if not target:
                self._send_json({"error": "Missing 'path' parameter"}, status=HTTPStatus.BAD_REQUEST)
                return
            full_path = os.path.realpath(os.path.join(self.config.working_dir, target))
            if not full_path.startswith(self.config.working_dir) or not os.path.isfile(full_path):
                self._send_json({"error": "File not found or outside workspace"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(200_000)
                self._send_json({"path": target, "content": content})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/git/status":
            proc = subprocess.run(["git", "status", "--short"], cwd=self.config.working_dir, capture_output=True, text=True)
            branch_proc = subprocess.run(["git", "branch", "--show-current"], cwd=self.config.working_dir, capture_output=True, text=True)
            self._send_json({
                "branch": branch_proc.stdout.strip() or "detached",
                "status": proc.stdout.strip(),
            })
            return

        if path == "/api/git/diff":
            proc = subprocess.run(["git", "diff"], cwd=self.config.working_dir, capture_output=True, text=True)
            self._send_json({"diff": proc.stdout.strip()})
            return

        if path == "/api/checkpoints":
            checkpoint_mgr = getattr(self.agent, "checkpoint_manager", None)
            cps = []
            if checkpoint_mgr:
                cps = [
                    {
                        "id": c.id,
                        "time": c.formatted_time,
                        "description": c.description,
                        "files_count": len(c.files),
                        "git_sha": c.git_sha,
                    }
                    for c in checkpoint_mgr.list_checkpoints()
                ]
            self._send_json({"checkpoints": cps})
            return

        if path == "/api/repomap":
            repomap = getattr(self.agent, "repomap", None)
            q = query.get("query", [None])[0]
            out = repomap.generate_map(max_tokens=1000, query=q) if repomap else ""
            self._send_json({"repomap": out})
            return

        if path == "/api/sessions":
            store = getattr(self.agent, "session_store", None)
            sessions = store.list_sessions() if store else []
            self._send_json({"sessions": sessions})
            return

        if path == "/api/subagents":
            sub_mgr = getattr(self.agent, "subagent_manager", None)
            history = sub_mgr.history if sub_mgr else []
            self._send_json({"subagents": history})
            return

        # Serve static assets
        if path == "/" or not os.path.exists(os.path.join(self.static_dir, path.lstrip("/"))):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/chat":
            self._handle_chat_sse()
            return

        if parsed.path == "/api/checkpoints/rollback":
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len)
            try:
                body = json.loads(post_data.decode("utf-8")) if post_data else {}
            except Exception:
                body = {}
            target_id = body.get("checkpoint_id")
            mgr = getattr(self.agent, "checkpoint_manager", None)
            if not mgr:
                self._send_json({"error": "No checkpoint manager"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            success, msg = mgr.rollback(target_id)
            self._send_json({"success": success, "message": msg})
            return

        self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_chat_sse(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len)
        try:
            req_json = json.loads(post_data.decode("utf-8"))
            prompt = req_json.get("prompt", "").strip()
        except Exception:
            self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send_sse(event: str, data: Any) -> None:
            msg = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
            try:
                self.wfile.write(msg)
                self.wfile.flush()
            except Exception:
                pass

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Hook agent callbacks to SSE
        orig_text_cb = self.agent.on_text
        self.agent.on_text = lambda token: send_sse("chunk", {"delta": token})

        try:
            answer = loop.run_until_complete(self.agent.run(prompt, interactive=True))
            send_sse("done", {"content": answer})
        except Exception as exc:
            send_sse("error", {"error": str(exc)})
        finally:
            self.agent.on_text = orig_text_cb
            loop.close()

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def translate_path(self, path: str) -> str:
        # Map URL to static_dir
        clean_path = path.split("?")[0].split("#")[0].lstrip("/")
        return os.path.join(self.static_dir, clean_path)


class WebServer:
    """Zero-dependency embedded developer web console."""

    def __init__(self, agent: Agent, config: Config, port: int = 8080) -> None:
        self.agent = agent
        self.config = config
        self.port = port
        self.static_dir = self._resolve_static_dir()

        handler = AgentuxWebHandler
        handler.agent = agent
        handler.config = config
        handler.static_dir = self.static_dir

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self.thread: Optional[threading.Thread] = None

    def _resolve_static_dir(self) -> str:
        # The repo's own static assets always win. Checking the cwd first
        # meant running agentux --web from inside another project that
        # happened to have an internal/web/static directory served that
        # project's UI instead of Agentux's.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        candidate = os.path.join(repo_root, "internal", "web", "static")
        if os.path.isdir(candidate):
            return candidate
        cwd = os.getcwd()
        # Only fall back to the cwd when it IS the agentux repo. Serving a
        # random project's files as the web console is worse than showing a
        # clear error.
        if os.path.abspath(cwd) == os.path.abspath(repo_root):
            return cwd
        raise RuntimeError(
            "Agentux web console has no static assets to serve. "
            "Build the web UI or run from the agentux repository directory."
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, open_browser: bool = False) -> None:
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        if open_browser:
            webbrowser.open(self.url)

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()


__all__ = ["WebServer", "AgentuxWebHandler"]
