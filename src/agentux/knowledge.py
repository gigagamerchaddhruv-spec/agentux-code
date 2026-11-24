"""Persistent knowledge database for Agentux Solo.

This is not conversation history. It is a growing store of facts Solo has
discovered, each with the source it came from, a confidence score, and a
verification status. The schema follows the Agentux Solo specification::

    knowledge          - the facts themselves
    sources            - where each fact came from
    research_history   - every search Solo performed
    metadata           - key/value store for engine state

SQLite keeps it dependency-free and it survives restarts because the file
lives on disk, not in memory.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

#: Where the store lives when nothing is configured.
DEFAULT_KNOWLEDGE_PATH = os.path.join(
    os.path.expanduser("~"), ".agentux", "knowledge.db"
)

#: Verification states a fact can carry.
VERIFICATION_UNVERIFIED = "unverified"
VERIFICATION_VERIFIED = "verified"
VERIFICATION_CONFLICTING = "conflicting"
VALID_VERIFICATION = (VERIFICATION_UNVERIFIED, VERIFICATION_VERIFIED, VERIFICATION_CONFLICTING)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,
    category        TEXT DEFAULT '',
    tags            TEXT DEFAULT '',
    confidence      REAL DEFAULT 0.5,
    verification    TEXT DEFAULT 'unverified',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    knowledge_id TEXT NOT NULL,
    url         TEXT NOT NULL,
    title       TEXT DEFAULT '',
    retrieved_at REAL NOT NULL,
    FOREIGN KEY (knowledge_id) REFERENCES knowledge(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS research_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query           TEXT NOT NULL,
    results_summary TEXT DEFAULT '',
    status          TEXT DEFAULT 'ok',
    completed_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category);
CREATE INDEX IF NOT EXISTS idx_knowledge_updated ON knowledge(updated_at);
CREATE INDEX IF NOT EXISTS idx_sources_knowledge ON sources(knowledge_id);
"""

_SELECT_KNOWLEDGE = """
SELECT id, title, content, category, tags, confidence, verification,
       created_at, updated_at
FROM knowledge
WHERE id = ?
"""

_SEARCH_KNOWLEDGE = """
SELECT id, title, content, category, tags, confidence, verification,
       created_at, updated_at
FROM knowledge
WHERE (content LIKE ? OR title LIKE ? OR category LIKE ? OR tags LIKE ?)
ORDER BY updated_at DESC
LIMIT ?
"""

_INSERT_KNOWLEDGE = """
INSERT INTO knowledge
    (id, title, content, category, tags, confidence, verification, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE_KNOWLEDGE = """
UPDATE knowledge
SET title = ?, content = ?, category = ?, tags = ?,
    confidence = ?, verification = ?, updated_at = ?
WHERE id = ?
"""

_INSERT_SOURCE = """
INSERT INTO sources (knowledge_id, url, title, retrieved_at)
VALUES (?, ?, ?, ?)
"""

_INSERT_RESEARCH = """
INSERT INTO research_history (query, results_summary, status, completed_at)
VALUES (?, ?, ?, ?)
"""


@dataclass
class Knowledge:
    """One fact Solo discovered and stored."""

    id: str
    title: str
    content: str
    category: str
    tags: str
    confidence: float
    verification: str
    created_at: float
    updated_at: float
    sources: List[Dict[str, Any]]


@dataclass
class ResearchRecord:
    """One web research operation."""

    id: int
    query: str
    results_summary: str
    status: str
    completed_at: float


def _row_to_knowledge(row: sqlite3.Row, sources: List[Dict[str, Any]]) -> Knowledge:
    return Knowledge(
        id=row["id"],
        title=row["title"],
        content=row["content"],
        category=row["category"],
        tags=row["tags"],
        confidence=row["confidence"],
        verification=row["verification"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sources=sources,
    )


class KnowledgeStore:
    """SQLite-backed knowledge base for Agentux Solo."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = os.path.abspath(
            os.path.expanduser(path) if path else DEFAULT_KNOWLEDGE_PATH
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
        title: str,
        content: str,
        *,
        category: str = "",
        tags: str = "",
        confidence: float = 0.5,
        verification: str = VERIFICATION_UNVERIFIED,
        sources: Optional[List[Dict[str, Any]]] = None,
        knowledge_id: Optional[str] = None,
    ) -> Knowledge:
        """Insert or update a knowledge entry.

        Updating rather than duplicating keeps the store readable: a model
        that re-learns a fact it already recorded would bury the original.
        """
        title = (title or "").strip() or "Untitled"
        content = (content or "").strip()
        if not content:
            raise ValueError("A knowledge entry needs non-empty content.")
        if verification not in VALID_VERIFICATION:
            raise ValueError(
                f"verification must be one of {', '.join(VALID_VERIFICATION)}"
            )
        confidence = max(0.0, min(1.0, float(confidence)))

        existing = self.get(knowledge_id) if knowledge_id else self._find_by_content(content)
        now = time.time()

        if existing is not None:
            self._conn.execute(
                _UPDATE_KNOWLEDGE,
                (title, content, category, tags, confidence, verification, now, existing.id),
            )
            self._conn.commit()
            return self.get(existing.id)  # type: ignore[return-value]

        new_id = knowledge_id or uuid.uuid4().hex[:12]
        self._conn.execute(
            _INSERT_KNOWLEDGE,
            (new_id, title, content, category, tags, confidence, verification, now, now),
        )
        if sources:
            for src in sources:
                self._conn.execute(
                    _INSERT_SOURCE,
                    (new_id, src.get("url", ""), src.get("title", ""), src.get("retrieved_at", now)),
                )
        self._conn.commit()
        return self.get(new_id)  # type: ignore[return-value]

    def _find_by_content(self, content: str) -> Optional[Knowledge]:
        row = self._conn.execute(
            "SELECT id FROM knowledge WHERE content = ? LIMIT 1", (content,)
        ).fetchone()
        return self.get(row["id"]) if row is not None else None

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def get(self, knowledge_id: str) -> Optional[Knowledge]:
        row = self._conn.execute(_SELECT_KNOWLEDGE, (knowledge_id,)).fetchone()
        if row is None:
            return None
        src_rows = self._conn.execute(
            "SELECT url, title, retrieved_at FROM sources WHERE knowledge_id = ?",
            (knowledge_id,),
        ).fetchall()
        sources = [
            {"url": s["url"], "title": s["title"], "retrieved_at": s["retrieved_at"]}
            for s in src_rows
        ]
        return _row_to_knowledge(row, sources)

    def search(self, query: str, *, category: Optional[str] = None, limit: int = 10) -> List[Knowledge]:
        """Keyword search across title, content, category and tags."""
        if not query or limit <= 0:
            return []
        like = f"%{query.strip()}%"
        rows = self._conn.execute(
            _SEARCH_KNOWLEDGE, (like, like, like, like, max(1, limit) * 3)
        ).fetchall()
        results: List[Knowledge] = []
        for row in rows:
            if category and row["category"] != category:
                continue
            src_rows = self._conn.execute(
                "SELECT url, title, retrieved_at FROM sources WHERE knowledge_id = ?",
                (row["id"],),
            ).fetchall()
            sources = [
                {"url": s["url"], "title": s["title"], "retrieved_at": s["retrieved_at"]}
                for s in src_rows
            ]
            results.append(_row_to_knowledge(row, sources))
        return results[:limit]

    def list_all(self, *, category: Optional[str] = None, limit: int = 50) -> List[Knowledge]:
        if category:
            rows = self._conn.execute(
                "SELECT id FROM knowledge WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                (category, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id FROM knowledge ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.get(r["id"]) for r in rows if self.get(r["id"]) is not None]  # type: ignore[misc]

    def delete(self, knowledge_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM knowledge").fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------
    # Research history
    # ------------------------------------------------------------------
    def record_research(self, query: str, results_summary: str = "", status: str = "ok") -> ResearchRecord:
        cursor = self._conn.execute(
            _INSERT_RESEARCH, (query, results_summary, status, time.time())
        )
        self._conn.commit()
        return ResearchRecord(
            id=cursor.lastrowid,
            query=query,
            results_summary=results_summary,
            status=status,
            completed_at=time.time(),
        )

    def recent_research(self, limit: int = 10) -> List[ResearchRecord]:
        rows = self._conn.execute(
            "SELECT id, query, results_summary, status, completed_at "
            "FROM research_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            ResearchRecord(
                id=r["id"], query=r["query"], results_summary=r["results_summary"],
                status=r["status"], completed_at=r["completed_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row is not None else default

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "Knowledge",
    "KnowledgeStore",
    "ResearchRecord",
    "VERIFICATION_UNVERIFIED",
    "VERIFICATION_VERIFIED",
    "VERIFICATION_CONFLICTING",
    "DEFAULT_KNOWLEDGE_PATH",
]