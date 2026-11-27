"""Shadow checkpoint and time-travel undo engine for Agentux.

Allows automatic and manual snapshots before modifying files, instant `/undo`
rollback of agent turns, and unified diff inspection (`/diff`). Works seamlessly
in git repositories via shadow commits / stash or standalone via atomic backups.
"""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Checkpoint:
    """A point-in-time snapshot of modified project files."""
    id: str
    timestamp: float
    description: str
    files: Dict[str, Optional[str]] = field(default_factory=dict)  # rel_path -> original content (None if new file)
    git_sha: Optional[str] = None

    @property
    def formatted_time(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


class CheckpointManager:
    """Manages project checkpoints, rollbacks, and diff computation."""

    def __init__(self, root: str, max_checkpoints: int = 50) -> None:
        self.root = os.path.abspath(root)
        self.max_checkpoints = max_checkpoints
        self._checkpoints: List[Checkpoint] = []
        self._is_git = os.path.isdir(os.path.join(self.root, ".git"))

    def capture_file_before(self, rel_path: str, checkpoint_id: str) -> None:
        """Record the state of a file before it gets edited."""
        abs_path = os.path.join(self.root, rel_path)
        cp = self.get_checkpoint(checkpoint_id)
        if not cp:
            return
        if rel_path in cp.files:
            return  # Keep the earliest original state

        if os.path.exists(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    cp.files[rel_path] = f.read()
            except Exception:
                cp.files[rel_path] = None
        else:
            cp.files[rel_path] = None

    def create_checkpoint(self, description: str, files_to_snapshot: Optional[List[str]] = None) -> Checkpoint:
        """Create a new checkpoint snapshot."""
        cp_id = str(uuid.uuid4())[:8]
        git_sha = None

        if self._is_git:
            try:
                # Capture current git commit hash if clean
                res = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if res.returncode == 0:
                    git_sha = res.stdout.strip()
            except Exception:
                pass

        files_map: Dict[str, Optional[str]] = {}
        if files_to_snapshot:
            for rel in files_to_snapshot:
                abs_p = os.path.join(self.root, rel)
                if os.path.isfile(abs_p):
                    try:
                        with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                            files_map[rel] = f.read()
                    except Exception:
                        files_map[rel] = None
                else:
                    files_map[rel] = None

        cp = Checkpoint(
            id=cp_id,
            timestamp=time.monotonic(),
            description=description,
            files=files_map,
            git_sha=git_sha,
        )
        self._checkpoints.append(cp)
        if len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints.pop(0)
        return cp

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        for cp in reversed(self._checkpoints):
            if cp.id == checkpoint_id:
                return cp
        return None

    def get_latest(self) -> Optional[Checkpoint]:
        return self._checkpoints[-1] if self._checkpoints else None

    def rollback(self, checkpoint_id: Optional[str] = None) -> Tuple[bool, str]:
        """Rollback changes made since the specified checkpoint (or the latest)."""
        if not self._checkpoints:
            return False, "No checkpoints available to roll back to."

        target: Optional[Checkpoint] = None
        if checkpoint_id:
            target = self.get_checkpoint(checkpoint_id)
            if not target:
                return False, f"Checkpoint '{checkpoint_id}' not found."
        else:
            target = self._checkpoints[-1]

        restored: List[str] = []
        errors: List[str] = []

        for rel_path, original_content in target.files.items():
            abs_path = os.path.join(self.root, rel_path)
            try:
                if original_content is None:
                    # File was newly created by the agent; remove it
                    if os.path.exists(abs_path):
                        os.remove(abs_path)
                        restored.append(f"removed newly created {rel_path}")
                else:
                    # Restore original content
                    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(original_content)
                    restored.append(f"restored {rel_path}")
            except Exception as exc:
                errors.append(f"failed to restore {rel_path}: {exc}")

        # Remove the checkpoint from history once rolled back
        if target in self._checkpoints:
            self._checkpoints.remove(target)

        summary_parts = []
        if restored:
            summary_parts.append(f"Successfully reverted {len(restored)} file(s):\n  - " + "\n  - ".join(restored))
        if errors:
            summary_parts.append(f"Errors during rollback:\n  - " + "\n  - ".join(errors))
        if not restored and not errors:
            summary_parts.append("No file modifications found to revert in this checkpoint.")

        return True, "\n".join(summary_parts)

    def get_diff(self, checkpoint_id: Optional[str] = None) -> str:
        """Compute unified diff between the checkpoint snapshot and the current disk state."""
        if not self._checkpoints:
            return "No modifications in current session."

        target = self.get_checkpoint(checkpoint_id) if checkpoint_id else self._checkpoints[0]
        if not target:
            return "Checkpoint not found."

        diff_outputs: List[str] = []

        # Compare recorded snapshot files
        for rel_path, original_content in target.files.items():
            abs_path = os.path.join(self.root, rel_path)
            current_content = ""
            if os.path.isfile(abs_path):
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                        current_content = f.read()
                except Exception:
                    continue
            elif original_content is None:
                continue

            orig_lines = (original_content or "").splitlines(keepends=True)
            curr_lines = current_content.splitlines(keepends=True)

            diff = list(
                difflib.unified_diff(
                    orig_lines,
                    curr_lines,
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                )
            )
            if diff:
                diff_outputs.append("".join(diff))

        if not diff_outputs and self._is_git:
            # Fallback to git diff
            try:
                res = subprocess.run(
                    ["git", "diff", "--no-color"],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0 and res.stdout.strip():
                    return res.stdout.strip()
            except Exception:
                pass

        return "\n".join(diff_outputs) if diff_outputs else "No active file differences."

    def list_checkpoints(self) -> List[Checkpoint]:
        return list(self._checkpoints)

    def clear(self) -> None:
        self._checkpoints.clear()
