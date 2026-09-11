"""P1 <-> P2 interface.

P2 should depend on these interfaces rather than directly accessing
P1's storage/database.
"""

from typing import Protocol

from .models import MemoryCandidate, MemoryContext


class MemoryCandidateProvider(Protocol):
    def candidate_retrieval(
        self,
        *,
        owner_id: str,
        query: str,
        limit: int = 20,
    ) -> list[MemoryCandidate]:
        ...


class MemoryContextProvider(Protocol):
    def build_memory_context(
        self,
        *,
        owner_id: str,
        query: str,
        selected_memory_ids: list[str],
    ) -> MemoryContext:
        ...