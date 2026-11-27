"""Autonomous test-driven verification and self-healing workflow engine for Agentux.

Provides automated test execution, failure diagnosis, surgical repair loops,
and rollback safety for software projects.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple


def detect_test_runner(root: str) -> Optional[str]:
    """Detect the most likely test command for the current project root."""
    if os.path.isfile(os.path.join(root, "pyproject.toml")) or os.path.isfile(os.path.join(root, "setup.py")):
        if os.path.isdir(os.path.join(root, "tests")):
            return "python3 -m pytest"
    if os.path.isfile(os.path.join(root, "go.mod")):
        return "go test ./..."
    if os.path.isfile(os.path.join(root, "package.json")):
        return "npm test"
    if os.path.isfile(os.path.join(root, "Cargo.toml")):
        return "cargo test"
    return None


class SelfHealingWorkflow:
    """Manages autonomous test execution, error diagnosis, and self-repair cycles."""

    def __init__(self, agent: Any, test_command: Optional[str] = None) -> None:
        self.agent = agent
        self.root = agent.config.working_dir or os.getcwd()
        self.test_command = test_command or detect_test_runner(self.root) or "pytest"

    def run_tests_sync(self, cmd: Optional[str] = None) -> Tuple[int, str]:
        """Execute test command and capture exit code and combined output."""
        command = cmd or self.test_command
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = (res.stdout or "") + (res.stderr or "")
            return res.returncode, output.strip()
        except subprocess.TimeoutExpired:
            return 124, f"Tests timed out after 60 seconds (command: {command})."
        except Exception as exc:
            return 1, f"Failed to execute test command '{command}': {exc}"

    async def execute_repair_loop(
        self,
        cmd: Optional[str] = None,
        max_attempts: int = 3,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute test-driven self-healing loop until tests pass or attempts exhausted."""
        command = cmd or self.test_command
        checkpoint_mgr = getattr(self.agent, "checkpoint_manager", None)
        initial_cp = None
        if checkpoint_mgr:
            initial_cp = checkpoint_mgr.create_checkpoint(f"Pre-repair snapshot before '{command}'")

        def emit(msg: str) -> None:
            if on_progress:
                on_progress(msg)
            elif hasattr(self.agent, "_emit"):
                self.agent._emit("narration", msg)

        emit(f"🧪 Running test suite: `{command}`")
        code, output = self.run_tests_sync(command)

        if code == 0:
            return {
                "success": True,
                "attempts": 0,
                "command": command,
                "output": output,
                "message": f"All tests passed on initial run ({command}).",
            }

        emit(f"❌ Test run failed with code {code}. Beginning autonomous repair loop (max {max_attempts} attempts)...")

        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            emit(f"🔧 Repair cycle {attempt}/{max_attempts}: Analyzing tracebacks and applying fixes...")

            prompt = (
                f"The test command `{command}` failed with exit code {code}.\n\n"
                f"--- Test Output ---\n{output[-2000:]}\n-------------------\n\n"
                "Please analyze the failing test(s) or traceback, inspect the relevant files, "
                "and apply a surgical fix to resolve the error. Keep changes focused only on the failure."
            )

            # Let agent run to repair
            await self.agent.run(prompt, interactive=False)

            # Re-run tests to verify fix
            emit(f"🧪 Re-running tests to verify attempt {attempt}...")
            code, output = self.run_tests_sync(command)
            if code == 0:
                emit(f"✅ Success! All tests passing after attempt {attempt}.")
                return {
                    "success": True,
                    "attempts": attempt,
                    "command": command,
                    "output": output,
                    "message": f"Successfully repaired and verified passing tests after {attempt} attempt(s).",
                }

        emit(f"⚠️ Tests still failing after {max_attempts} attempts.")
        return {
            "success": False,
            "attempts": attempt,
            "command": command,
            "output": output,
            "initial_checkpoint": initial_cp.id if initial_cp else None,
            "message": f"Tests still failing after {max_attempts} repair attempts. You can run `/undo` to revert.",
        }
