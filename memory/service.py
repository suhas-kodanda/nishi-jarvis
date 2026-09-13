from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
from uuid import uuid4

import memory

from .models import (
    MemoryCandidate,
    MemoryContext,
    MemoryKind,
    MemoryLayer,
    MemoryRecord,
    MemorySource,
)
from .storage import MemoryStorage


class MemoryService:
    """Main P1 Memory Engine interface.

    P2 should use this service instead of accessing the
    database directly.
    """

    def __init__(self, storage: MemoryStorage | None = None):
        self.storage = storage or MemoryStorage()

    def save_memory(
        self,
        *,
        owner_id: str,
        stable_key: str,
        layer: MemoryLayer,
        kind: MemoryKind,
        content: str,
        importance: float = 0.5,
        confidence: float = 1.0,
        source: MemorySource | None = None,
        metadata: dict | None = None,
        expires_at: datetime | None = None,
        session_id: str | None = None,
    ) -> MemoryRecord:

        metadata = dict(metadata or {})

        if session_id is not None:
            metadata["session_id"] = session_id

        memory = MemoryRecord(
            id=str(uuid4()),
            owner_id=owner_id,
            stable_key=stable_key,
            layer=layer,
            kind=kind,
            content=content.strip(),
            importance=importance,
            confidence=confidence,
            source=source,
            metadata=metadata or {},
            expires_at=expires_at,
        )

        return self.storage.upsert(memory)

    def save_conversation_turn(
        self,
        *,
        owner_id: str,
        user_message: str,
        assistant_response: str,
        session_id: str,
        timestamp: datetime | None = None,
        turn_id: str | None = None,
        importance: float = 0.5,
        confidence: float = 1.0,
    ) -> MemoryRecord:

        timestamp = timestamp or datetime.now(timezone.utc)
        turn_id = turn_id or str(uuid4())

        stable_key = f"history_turn_{turn_id}"

        metadata = {
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.date().isoformat(),
            "record_type": "conversation_turn",
        }

        return self.save_memory(
            owner_id=owner_id,
            stable_key=stable_key,
            layer=MemoryLayer.L3,
            kind=MemoryKind.HISTORY,
            content=(
                f"User: {user_message.strip()}\n"
                f"Nishi: {assistant_response.strip()}"
            ),
            importance=importance,
            confidence=confidence,
            source=MemorySource(
                source_type="user_message",
                source_id=turn_id,
            ),
            metadata=metadata,
        )

    def save_verified_action(
        self,
        *,
        owner_id: str,
        session_id: str,
        action_type: str,
        action_result: str,
        importance: float = 0.8,
        confidence: float = 1.0,
        timestamp: datetime | None = None,
        event_id: str | None = None,
    ) -> MemoryRecord:

        timestamp = timestamp or datetime.now(timezone.utc)
        event_id = event_id or str(uuid4())

        stable_key = f"action_event_{event_id}"

        metadata = {
            "session_id": session_id,
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.date().isoformat(),
            "record_type": "verified_action",
            "action_type": action_type,
        }

        return self.save_memory(
            owner_id=owner_id,
            stable_key=stable_key,
            layer=MemoryLayer.L4,
            kind=MemoryKind.ACTION_EVENT,
            content=(
                f"Action: {action_type}\n"
                f"Result: {action_result.strip()}"
            ),
            importance=importance,
            confidence=confidence,
            source=MemorySource(
                source_type="action_result",
                source_id=event_id,
            ),
            metadata=metadata,
        )

    def get_memory(
        self,
        *,
        owner_id: str,
        memory_id: str,
    ) -> MemoryRecord | None:

        return self.storage.get(
            memory_id,
            owner_id,
        )

    def delete_memory(
        self,
        *,
        owner_id: str,
        memory_id: str,
    ) -> bool:

        return self.storage.delete(
            memory_id,
            owner_id,
        )

    def candidate_retrieval(
        self,
        *,
        owner_id: str,
        query: str,
        limit: int = 20,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> list[MemoryCandidate]:
        """Return candidate memories to P2.

        P2 is responsible for reranking and disambiguation.

        This first version uses transparent lexical matching.
        Later, embeddings/pgvector can be added behind this
        same interface.
        """

        query_tokens = set(self._tokens(query))
        candidates: list[MemoryCandidate] = []

        for memory in self.storage.list_active(owner_id):

            if self._expired(memory):
                continue

            if session_id is not None:
                if memory.metadata.get("session_id") != session_id:
                    continue

            if metadata is not None:
                if any(
                    memory.metadata.get(key) != value
                    for key, value in metadata.items()
                ):
                    continue

            content_tokens = set(
                self._tokens(memory.content)
            )

            overlap = len(
                query_tokens & content_tokens
            )

            lexical_score = overlap / max(
                len(query_tokens),
                1,
            )

            score = (
                0.75 * lexical_score
                + 0.15 * memory.importance
                + 0.10 * memory.confidence
            )

            if overlap > 0 or memory.layer in (MemoryLayer.L1, MemoryLayer.L2):
                candidates.append(
                    MemoryCandidate(
                        memory_id=memory.id,
                        stable_key=memory.stable_key,
                        layer=memory.layer,
                        kind=memory.kind,
                        content=memory.content,
                        importance=memory.importance,
                        confidence=memory.confidence,
                        score=score,
                        metadata=memory.metadata,
                    )
                )

        candidates.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return candidates[:limit]

    def build_memory_context(
        self,
        *,
        owner_id: str,
        query: str,
        selected_memory_ids: list[str],
    ) -> MemoryContext:
        """Build final memory context after P2 selection."""

        selected: list[MemoryCandidate] = []

        for memory_id in selected_memory_ids:

            memory = self.get_memory(
                owner_id=owner_id,
                memory_id=memory_id,
            )

            if (
                memory
                and not memory.deleted
                and not self._expired(memory)
            ):
                selected.append(
                    MemoryCandidate(
                        memory_id=memory.id,
                        stable_key=memory.stable_key,
                        layer=memory.layer,
                        kind=memory.kind,
                        content=memory.content,
                        importance=memory.importance,
                        confidence=memory.confidence,
                        score=1.0,
                        metadata=memory.metadata,
                    )
                )

        return MemoryContext(
            query=query,
            memories=selected,
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        punctuation = ".,!?;:()[]{}'\""

        return [
            token.strip(punctuation).lower()
            for token in text.split()
            if token.strip(punctuation)
        ]

    @staticmethod
    def _expired(
        memory: MemoryRecord,
    ) -> bool:

        return bool(
            memory.expires_at
            and memory.expires_at
            <= datetime.now(timezone.utc)
        )

    def get_l1_personality(self, *, owner_id: str):
        return [
            memory
            for memory in self.storage.list_active(owner_id)
            if memory.layer == MemoryLayer.L1
            and not self._expired(memory)
        ]

    def get_l2_active_goals(self, *, owner_id: str):
        return [
            memory
            for memory in self.storage.list_active(owner_id)
            if memory.layer == MemoryLayer.L2
            and memory.metadata.get("status", "active") == "active"
            and not self._expired(memory)
        ]

    def _within_time_range(
        self,
        memory: MemoryRecord,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> bool:
        """Return True when a memory falls inside the requested time range."""

        if start_time is None and end_time is None:
            return True

        timestamp_value = memory.metadata.get("timestamp")

        if not timestamp_value:
            return False

        try:
            memory_time = datetime.fromisoformat(timestamp_value)
        except (TypeError, ValueError):
            return False

        if memory_time.tzinfo is None:
            memory_time = memory_time.replace(tzinfo=timezone.utc)

        if start_time is not None:
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)

            if memory_time < start_time:
                return False

        if end_time is not None:
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)

            if memory_time > end_time:
                return False

        return True

    def get_l3_history(
        self,
        *,
        owner_id: str,
        session_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:

        memories = self.storage.list_active(
            owner_id=owner_id,
            layer=MemoryLayer.L3,
        )

        results: list[MemoryRecord] = []

        for memory in memories:
            if session_id is not None:
                if memory.metadata.get("session_id") != session_id:
                    continue

            if not self._within_time_range(
                memory,
                start_time=start_time,
                end_time=end_time,
            ):
                continue

            results.append(memory)

        results.sort(
            key=lambda memory: memory.metadata.get("timestamp", ""),
            reverse=True,
        )

        return results[:limit]

    def get_l4_current_state(
        self,
        *,
        owner_id: str,
        session_id: str,
        limit: int = 4,
    ):
        memories = [
            memory
            for memory in self.storage.list_active(owner_id)
            if memory.layer == MemoryLayer.L4
            and memory.metadata.get("session_id") == session_id
            and not self._expired(memory)
        ]

        return memories[:limit]

    