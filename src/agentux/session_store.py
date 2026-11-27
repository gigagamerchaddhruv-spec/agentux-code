"""Session persistence and resumption engine for Agentux.

Allows saving and restoring complete conversational and working sessions
across terminal closures, including full message histories, tool state,
working directories, and metadata.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional
from .models import Message


def _default_session_dir(working_dir: Optional[str] = None) -> str:
    """Session directory preference: project .agentux/sessions if exists, else user config."""
    if working_dir:
        project_dir = os.path.join(working_dir, ".agentux", "sessions")
        if os.path.isdir(os.path.join(working_dir, ".agentux")):
            os.makedirs(project_dir, exist_ok=True)
            return project_dir
    global_dir = os.path.expanduser("~/.config/agentux/sessions")
    os.makedirs(global_dir, exist_ok=True)
    return global_dir


class SessionStore:
    """Saves, lists, and restores Agentux sessions."""

    def __init__(self, directory: Optional[str] = None) -> None:
        self.directory = directory or _default_session_dir()
        os.makedirs(self.directory, exist_ok=True)

    def _file_path(self, session_id: str) -> str:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        return os.path.join(self.directory, f"{safe_id}.json")

    def save(
        self,
        session_id: str,
        history: List[Message],
        model: str,
        working_dir: str,
        title: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist a conversation session to disk."""
        path = self._file_path(session_id)
        first_user_prompt = ""
        for m in history:
            if m.role == "user" and m.content:
                first_user_prompt = m.content[:80].strip()
                break

        data: Dict[str, Any] = {
            "id": session_id,
            "title": title or first_user_prompt or f"Session {session_id}",
            "created_at": time.time(),
            "updated_at": time.time(),
            "model": model,
            "working_dir": working_dir,
            "turn_count": len([m for m in history if m.role == "user"]),
            "messages": [m.to_dict() for m in history],
            "extra": extra or {},
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data by ID or filename."""
        path = self._file_path(session_id)
        if not os.path.isfile(path):
            # Check if session_id is a partial match
            for f in os.listdir(self.directory):
                if f.endswith(".json") and session_id in f:
                    path = os.path.join(self.directory, f)
                    break
        if not os.path.isfile(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Reconstruct Message instances
            messages = [Message.from_dict(m) for m in data.get("messages", [])]
            data["messages"] = messages
            return data
        except Exception:
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions sorted by most recently modified."""
        sessions: List[Dict[str, Any]] = []
        if not os.path.isdir(self.directory):
            return sessions

        for filename in sorted(os.listdir(self.directory), reverse=True):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self.directory, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                sessions.append({
                    "id": meta.get("id", filename[:-5]),
                    "title": meta.get("title", "Untitled"),
                    "model": meta.get("model", ""),
                    "updated_at": meta.get("updated_at", os.path.getmtime(path)),
                    "turn_count": meta.get("turn_count", 0),
                    "working_dir": meta.get("working_dir", ""),
                })
            except Exception:
                continue

        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions

    def delete(self, session_id: str) -> bool:
        path = self._file_path(session_id)
        if os.path.isfile(path):
            try:
                os.remove(path)
                return True
            except OSError:
                return False
        return False
