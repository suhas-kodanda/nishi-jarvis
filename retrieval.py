"""
NISHI P2 - Memory Retrieval and Selection

P2 responsibilities:
1. Receive candidate memories from P1.
2. Rerank candidates.
3. Select the most relevant memories.
4. Return selected memory IDs.
5. Send selected IDs back to P1.
"""

from memory.models import MemoryCandidate


def calculate_p2_score(memory: MemoryCandidate) -> float:
    """
    Calculate the P2 score.

    P1 has already calculated an initial relevance score.
    P2 uses that score along with importance and confidence.
    """

    score = (
        0.70 * memory.score
        + 0.20 * memory.importance
        + 0.10 * memory.confidence
    )

    return score


def rerank_candidates(
    candidates: list[MemoryCandidate],
) -> list[MemoryCandidate]:

    scored_candidates = []

    for memory in candidates:

        p2_score = calculate_p2_score(memory)

        scored_candidates.append(
            (memory, p2_score)
        )

    scored_candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return [
        item[0]
        for item in scored_candidates
    ]


def select_memories(
    candidates: list[MemoryCandidate],
    top_n: int = 5,
) -> list[MemoryCandidate]:

    ranked = rerank_candidates(candidates)

    return ranked[:top_n]


def get_selected_memory_ids(
    memories: list[MemoryCandidate],
) -> list[str]:

    return [
        memory.memory_id
        for memory in memories
    ]


def retrieve_and_select(
    provider,
    owner_id: str,
    query: str,
    top_n: int = 5,
):
    """
    P1 -> P2 pipeline.

    P1 gives candidate memories.
    P2 reranks and selects the best memories.
    """

    candidates = provider.candidate_retrieval(
        owner_id=owner_id,
        query=query,
        limit=20,
    )

    selected_memories = select_memories(
        candidates,
        top_n,
    )

    selected_ids = get_selected_memory_ids(
        selected_memories
    )

    return selected_memories, selected_ids


def retrieve_final_context(
    provider,
    owner_id: str,
    query: str,
    top_n: int = 5,
):
    """
    Complete P1 -> P2 -> P1 pipeline.

    1. P1 retrieves candidates.
    2. P2 reranks them.
    3. P2 selects top N.
    4. P2 sends selected IDs back to P1.
    5. P1 creates the final MemoryContext.
    """

    selected_memories, selected_ids = retrieve_and_select(
        provider=provider,
        owner_id=owner_id,
        query=query,
        top_n=top_n,
    )

    final_context = provider.build_memory_context(
        owner_id=owner_id,
        query=query,
        selected_memory_ids=selected_ids,
    )

    return final_context