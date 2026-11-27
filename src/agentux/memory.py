"""Persistent knowledge store for Agentux Solo.

A small model cannot hold what it learns. This module is the place it puts
things so they survive the session: a SQLite database with a full-text index,
written by the model through the ``memory_save`` tool and read back every turn
by the agent.

The design is deliberately dependency-free. SQLite is in the standard library
and FTS5 is compiled into it, so the store costs nothing to run and nothing to
install. Scope is a property of each memory, not of the store: a memory tagged
``project`` is only reachable from that working directory, and one tagged
``global`` is reachable from anywhere. Retrieval ranks project memories above
global ones, so a local gotcha wins over a general lesson.

The full-text index does the heavy lifting on recall; there is no embedding
model and no vector database. The schema keeps an ``id`` column so a later
vector column can be added without a migration of the existing rows.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: Scope values a memory can carry.
SCOPE_PROJECT = "project"
SCOPE_GLOBAL = "global"
VALID_SCOPES = (SCOPE_PROJECT, SCOPE_GLOBAL)

#: Bodies are stored whole but never injected whole.
MAX_BODY_CHARS = 4_000
MAX_INJECT_CHARS = 400

#: Retrieval defaults.
FTS_MAX_TERMS = 6
DEFAULT_LIMIT = 5

#: Words that carry no signal for recall. Kept small on purpose: a longer list
#: risks dropping a term that was the whole point of the sentence.
_STOPWORDS = frozenset(
    """
    the a an and or but if then else for to of in on at by with from as is are
    was were be been being it its this that these those i you he she we they me
    my your his her our their do does did can could will would should may might
    must not no so than too very just about into over after before what which
    who how when where why have has had having up out off
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9_]+")

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5(
    body,
    scope UNINDEXED,
    project UNINDEXED,
    source UNINDEXED,
    tags UNINDEXED,
    id UNINDEXED,
    created_at UNINDEXED,
    updated_at UNINDEXED,
    accesses UNINDEXED,
    last_accessed UNINDEXED,
    tokenize = 'porter unicode61'
)
"""

# rank is FTS5's BM25 score; more negative is a better match.
_SELECT = """
SELECT id, body, scope, project, source, tags, created_at, updated_at,
       accesses, last_accessed, rank
FROM memories
WHERE memories MATCH ?
  AND (scope = 'global' OR project = ? OR project IS NULL)
ORDER BY rank
LIMIT ?
"""

_INSERT = """
INSERT INTO memories
    (body, scope, project, source, tags, id, created_at, updated_at,
     accesses, last_accessed)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
"""

_UPDATE = """
UPDATE memories
SET body = ?, scope = ?, project = ?, source = ?, tags = ?, updated_at = ?
WHERE id = ?
"""

_FIND_BY_BODY = "SELECT id FROM memories WHERE body = ? LIMIT 1"


@dataclass
class Memory:
    """One thing the model learned."""

    id: str
    body: str
    scope: str
    project: Optional[str]
    source: str
    tags: str
    created_at: float
    updated_at: float
    accesses: int
    last_accessed: Optional[float]


def default_memory_path() -> str:
    """Where the store lives when no path is given."""
    return os.path.join(os.path.expanduser("~"), ".agentux", "memory.db")


def _normalize(body: str) -> str:
    return re.sub(r"\s+", " ", str(body).strip())


def build_query(text: str, max_terms: int = FTS_MAX_TERMS) -> str:
    """Turn a user message into an FTS5 ``MATCH`` expression.

    Returns ``""`` when nothing usable remains. Every term is double-quoted, so
    characters that FTS5 would otherwise treat as syntax are matched as
    literal text; a user message containing quotes, parentheses or asterisks
    cannot break the query.
    """
    terms: List[str] = []
    for raw in _TOKEN.findall(str(text).lower()):
        if len(raw) < 3 or raw in _STOPWORDS:
            continue
        if raw not in terms:
            terms.append(raw)
        if len(terms) >= max_terms:
            break
    return " OR ".join(f'"{term}"' for term in terms)


def render_memory_block(
    memories: Sequence[Tuple[Memory, float]],
    max_chars: int = MAX_INJECT_CHARS,
) -> str:
    """Render recalled memories as the block injected into the system prompt.

    Empty when there is nothing to say, so the caller can append the result
    unconditionally.
    """
    if not memories:
        return ""
    lines = ["## Things you already know", ""]
    for memory, _score in memories:
        body = memory.body
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "..."
        if memory.scope == SCOPE_GLOBAL:
            where = "global"
        else:
            where = memory.project or "this project"
        lines.append(f"- {body}  [{memory.id} · {where}]")
    return "\n".join(lines)


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        body=row["body"],
        scope=row["scope"],
        project=row["project"],
        source=row["source"],
        tags=row["tags"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        accesses=row["accesses"],
        last_accessed=row["last_accessed"],
    )


class MemoryStore:
    """A SQLite-backed memory store with a full-text index."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = os.path.abspath(
            os.path.expanduser(path) if path else default_memory_path()
        )
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def save(
        self,
        body: str,
        *,
        scope: str = SCOPE_PROJECT,
        project: Optional[str] = None,
        source: str = "model",
        tags: str = "",
        memory_id: Optional[str] = None,
    ) -> Memory:
        """Store a memory, or update it if the same body is already there.

        Updating rather than duplicating is what keeps the store usable: a
        model that re-learns a fact it already recorded would otherwise bury
        the original under copies and flood the context with repeats.
        """
        normalized = _normalize(body)
        if not normalized:
            raise ValueError("A memory needs a non-empty body.")
        if scope not in VALID_SCOPES:
            raise ValueError(
                f"Scope must be one of {', '.join(VALID_SCOPES)}, got {scope!r}."
            )
        if len(normalized) > MAX_BODY_CHARS:
            normalized = normalized[:MAX_BODY_CHARS].rstrip()

        existing = self.get(memory_id) if memory_id else None
        if existing is None:
            row = self._conn.execute(_FIND_BY_BODY, (normalized,)).fetchone()
            existing = self.get(row["id"]) if row is not None else None

        now = time.time()
        if existing is not None:
            self._conn.execute(
                _UPDATE,
                (normalized, scope, project, source, tags, now, existing.id),
            )
            self._conn.commit()
            return self.get(existing.id)  # type: ignore[return-value]

        new_id = memory_id or uuid.uuid4().hex[:12]
        self._conn.execute(
            _INSERT,
            (normalized, scope, project, source, tags, new_id, now, now),
        )
        self._conn.commit()
        return self.get(new_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        project: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
    ) -> List[Tuple[Memory, float]]:
        """Return the memories that best match ``query``, best first.

        ``query`` must already be an FTS5 ``MATCH`` expression; use
        :func:`build_query` to make one from plain text. An empty query
        returns nothing rather than raising, so a turn with no usable terms
        costs no recall and no error.
        """
        if not query or limit <= 0:
            return []

        rows = self._conn.execute(
            _SELECT, (query, project, max(1, limit) * 3)
        ).fetchall()

        now = time.time()
        scored: List[Tuple[Memory, float]] = []
        for row in rows:
            memory = _row_to_memory(row)
            # BM25: FTS5's rank is negative, so negate it to get "higher is
            # better" and add the tie-breakers on top.
            score = -float(row["rank"])
            if memory.scope == SCOPE_PROJECT and memory.project == project:
                score += 2.0
            if memory.last_accessed:
                age_days = max(0.0, (now - memory.last_accessed) / 86_400.0)
                score += max(0.0, 1.0 - age_days / 30.0)
            score += min(memory.accesses, 20) * 0.05
            scored.append((memory, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def get(self, memory_id: str) -> Optional[Memory]:
        row = self._conn.execute(
            "SELECT id, body, scope, project, source, tags, created_at, "
            "updated_at, accesses, last_accessed FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return _row_to_memory(row) if row is not None else None

    def delete(self, memory_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self, *, project: Optional[str] = None) -> int:
        if project is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM memories "
                "WHERE scope = 'global' OR project = ?",
                (project,),
            ).fetchone()
        return int(row["n"])

    def touch(self, memory_ids: Sequence[str]) -> None:
        """Record that these memories were used, so ranking learns from it."""
        now = time.time()
        for memory_id in memory_ids:
            self._conn.execute(
                "UPDATE memories SET accesses = accesses + 1, "
                "last_accessed = ? WHERE id = ?",
                (now, memory_id),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "Memory",
    "MemoryStore",
    "SCOPE_PROJECT",
    "SCOPE_GLOBAL",
    "VALID_SCOPES",
    "build_query",
    "render_memory_block",
    "default_memory_path",
]
