"""Task tracking for multi-step work.

An agent given a five-part task will quietly do three of them and report
success. Writing the plan down first, and updating it as it goes, is the
cheapest thing that keeps the whole task in view.

The list lives in memory and is rendered back to the model on every update, so
the current state is always in the conversation rather than something the model
has to remember.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..errors import ToolError
from .base import BaseTool

#: The states a task moves through. Kept small on purpose.
STATUSES = ("pending", "in_progress", "completed", "cancelled")

#: The columns a rendered board uses, in order.
ORDER = ("in_progress", "pending", "completed", "cancelled")

MAX_TASKS = 50


class TodoTool(BaseTool):
    """Maintain a checklist of what the current task requires."""

    def __init__(self) -> None:
        self._items: List[Dict[str, str]] = []
        self._next_id = 1

    @property
    def name(self) -> str:
        return "todo"

    @property
    def description(self) -> str:
        return (
            "Track a checklist of steps for the current task. Create the full "
            "list before starting a multi-step job, keep exactly one item "
            "'in_progress' at a time, and mark items completed as you finish "
            "them. The rendered list comes back on every update, so you do not "
            "need to track it yourself. Use it for anything with more than "
            "about three steps; skip it for single actions."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "update", "list", "clear"],
                    "description": (
                        "add: append tasks. update: change one task's status. "
                        "list: render the current checklist. clear: remove all."
                    ),
                },
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Task descriptions. Required for 'add'.",
                },
                "id": {
                    "type": "integer",
                    "description": "Task id to update. Required for 'update'.",
                },
                "status": {
                    "type": "string",
                    "enum": list(STATUSES),
                    "description": "New status for a task. Required for 'update'.",
                },
            },
            "required": ["action"],
        }

    # ------------------------------------------------------------------
    def _find(self, task_id: Any) -> Dict[str, str]:
        try:
            wanted = int(task_id)
        except (TypeError, ValueError):
            raise ToolError(f"Task id must be an integer, got {task_id!r}.") from None

        for item in self._items:
            if int(item["id"]) == wanted:
                return item
        valid = ", ".join(str(i["id"]) for i in self._items) or "(none)"
        raise ToolError(f"No task with id {wanted}. Existing ids: {valid}")

    def _render(self) -> str:
        if not self._items:
            return "The checklist is empty."

        done = sum(1 for i in self._items if i["status"] == "completed")
        cancelled = sum(1 for i in self._items if i["status"] == "cancelled")
        total = len(self._items)

        summary = f"{done}/{total} complete"
        if cancelled:
            summary += f", {cancelled} cancelled"
        lines = [f"Checklist ({summary}):"]

        for status in ORDER:
            group = [i for i in self._items if i["status"] == status]
            if not group:
                continue
            label = {"in_progress": "IN PROGRESS", "pending": "TODO",
                     "completed": "DONE", "cancelled": "SKIPPED"}[status]
            lines.append(f"\n{label}")
            for item in group:
                text = item["task"]
                if len(text) > 120:
                    text = text[:120] + "..."
                lines.append(f"  [{str(item['id']):>2}] {text}")

        remaining = total - done - cancelled
        if remaining:
            lines.append(f"\n{remaining} item(s) still open.")
        else:
            lines.append("\nEverything on the list is finished.")
        return "\n".join(lines)

    def _add(self, tasks: Optional[List[str]]) -> str:
        if not tasks:
            raise ToolError("'add' needs a 'tasks' list of descriptions.")
        room = MAX_TASKS - len(self._items)
        if room <= 0:
            raise ToolError(
                f"The checklist is full ({MAX_TASKS} items). Update or clear "
                "existing tasks before adding more."
            )
        for text in tasks[:room]:
            self._items.append({
                "id": str(self._next_id),
                "task": str(text),
                "status": "pending",
            })
            self._next_id += 1
        if len(tasks) > room:
            return self._render() + f"\n\n... {len(tasks) - room} task(s) dropped: the list is full."
        return self._render()

    def _update(self, task_id: Any, status: Optional[str]) -> str:
        if task_id is None:
            raise ToolError("'update' needs a task 'id'.")
        if status not in STATUSES:
            raise ToolError(
                f"Invalid status {status!r}. Use one of: {', '.join(STATUSES)}."
            )

        item = self._find(task_id)
        previous = item["status"]
        item["status"] = status

        # Keep at most one task in progress so "what am I doing now" stays
        # unambiguous as the list gets long.
        cleared = ""
        if status == "in_progress":
            for other in self._items:
                if other is not item and other["status"] == "in_progress":
                    other["status"] = "pending"
                    cleared = f"\nMoved task [{other['id']}] back to pending."

        return (
            f"Task [{item['id']}]: {previous} -> {status}.{cleared}\n\n" + self._render()
        )

    async def execute(
        self,
        action: str,
        tasks: Optional[List[str]] = None,
        id: Optional[int] = None,  # noqa: A002 - matches the JSON Schema name
        status: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        if action == "list":
            return self._render()
        if action == "add":
            return self._add(tasks)
        if action == "update":
            return self._update(id, status)
        if action == "clear":
            count = len(self._items)
            self._items.clear()
            return f"Cleared {count} task(s)."

        raise ToolError(
            f"Unsupported action {action!r}. Use 'add', 'update', 'list', or 'clear'."
        )


__all__ = ["TodoTool"]
