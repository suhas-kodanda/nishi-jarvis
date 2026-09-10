from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

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
    ) -> MemoryRecord:

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

            if overlap > 0:
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