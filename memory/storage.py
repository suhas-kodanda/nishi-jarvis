from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import MemoryRecord


class MemoryStorage:
    """Persistence layer for P1 memory.

    SQLite is used for the first integration so the team can test
    the memory system without setting up PostgreSQL immediately.
    """

    def __init__(self, db_path: str = "data/memory.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        self._create_table()

    def _create_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                stable_key TEXT NOT NULL,
                layer TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL,
                confidence REAL NOT NULL,
                source_json TEXT,
                metadata_json TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                deleted INTEGER NOT NULL DEFAULT 0
            )
            """
        )

        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_active_memory_key
            ON memories(owner_id, stable_key)
            WHERE deleted = 0
            """
        )

        self.conn.commit()

    def upsert(self, memory: MemoryRecord) -> MemoryRecord:
        existing = self.get_by_stable_key(
            memory.owner_id,
            memory.stable_key,
        )
        
        next_version = (
            existing.version + 1 
            if existing 
            else 1
        )
        
        memory = memory.model_copy(
            update={"version": next_version}
        )

        if existing and not existing.deleted:
            self.conn.execute(
                """
                UPDATE memories
                SET deleted = 1, updated_at = ?
                WHERE id = ?
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    existing.id,
                ),
            )

        self.conn.execute(
            """
            INSERT INTO memories (
                id,
                owner_id,
                stable_key,
                layer,
                kind,
                content,
                importance,
                confidence,
                source_json,
                metadata_json,
                version,
                created_at,
                updated_at,
                expires_at,
                deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.owner_id,
                memory.stable_key,
                memory.layer.value,
                memory.kind.value,
                memory.content,
                memory.importance,
                memory.confidence,
                (
                    memory.source.model_dump_json()
                    if memory.source
                    else None
                ),
                json.dumps(memory.metadata),
                memory.version,
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                (
                    memory.expires_at.isoformat()
                    if memory.expires_at
                    else None
                ),
                int(memory.deleted),
            ),
        )

        self.conn.commit()

        return memory

    def get(
        self,
        memory_id: str,
        owner_id: str,
    ) -> MemoryRecord | None:

        row = self.conn.execute(
            """
            SELECT *
            FROM memories
            WHERE id = ? AND owner_id = ?
            """,
            (memory_id, owner_id),
        ).fetchone()

        return self._row_to_model(row) if row else None

    def get_by_stable_key(
        self,
        owner_id: str,
        stable_key: str,
    ) -> MemoryRecord | None:

        row = self.conn.execute(
            """
            SELECT *
            FROM memories
            WHERE owner_id = ?
              AND stable_key = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (owner_id, stable_key),
        ).fetchone()

        return self._row_to_model(row) if row else None

    def list_active(
        self,
        owner_id: str,
    ) -> list[MemoryRecord]:

        rows = self.conn.execute(
            """
            SELECT *
            FROM memories
            WHERE owner_id = ?
              AND deleted = 0
            ORDER BY importance DESC, updated_at DESC
            """,
            (owner_id,),
        ).fetchall()

        return [
            self._row_to_model(row)
            for row in rows
        ]

    def delete(
        self,
        memory_id: str,
        owner_id: str,
    ) -> bool:

        cur = self.conn.execute(
            """
            UPDATE memories
            SET deleted = 1,
                updated_at = ?
            WHERE id = ?
              AND owner_id = ?
              AND deleted = 0
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                memory_id,
                owner_id,
            ),
        )

        self.conn.commit()

        return cur.rowcount > 0

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> MemoryRecord:

        source = (
            json.loads(row["source_json"])
            if row["source_json"]
            else None
        )

        metadata = json.loads(row["metadata_json"])

        return MemoryRecord(
            id=row["id"],
            owner_id=row["owner_id"],
            stable_key=row["stable_key"],
            layer=row["layer"],
            kind=row["kind"],
            content=row["content"],
            importance=row["importance"],
            confidence=row["confidence"],
            source=source,
            metadata=metadata,
            version=row["version"],
            created_at=datetime.fromisoformat(
                row["created_at"]
            ),
            updated_at=datetime.fromisoformat(
                row["updated_at"]
            ),
            expires_at=(
                datetime.fromisoformat(row["expires_at"])
                if row["expires_at"]
                else None
            ),
            deleted=bool(row["deleted"]),
        )