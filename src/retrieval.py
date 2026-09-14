"""
NISHI P2 - Memory Retrieval and Selection

P2 responsibilities:
1. Receive candidate memories from P1.
2. Identify the type of memory query.
3. Rerank candidates.
4. Give priority to relevant memory layers.
5. Select the most relevant memories.
6. Return selected memory IDs.
"""

from memory.models import MemoryCandidate, MemoryLayer


# ---------------------------------------------------------
# 1. Detect what kind of memory the user is asking for
# ---------------------------------------------------------

def is_history_query(query: str) -> bool:
    """
    Check whether the user is asking about previous conversations.
    """

    query = query.lower()

    history_words = [
        "conversation",
        "conversations",
        "talk",
        "talked",
        "discuss",
        "discussed",
        "chat",
        "chats",
        "history",
        "what did we talk",
        "what did we discuss",
        "what have we talked",
    ]

    for word in history_words:
        if word in query:
            return True

    return False


def is_today_query(query: str) -> bool:
    """
    Check whether the user is asking specifically about today.
    """

    query = query.lower()

    today_words = [
        "today",
        "this day",
        "earlier today",
        "so far today",
    ]

    for word in today_words:
        if word in query:
            return True

    return False


# ---------------------------------------------------------
# 2. Calculate the normal P2 score
# ---------------------------------------------------------

def calculate_p2_score(memory: MemoryCandidate) -> float:
    """
    Normal P2 ranking score.

    P1 already gives us a relevance score.
    Importance and confidence are also considered.
    """

    score = (
        0.70 * memory.score
        + 0.20 * memory.importance
        + 0.10 * memory.confidence
    )

    return score


# ---------------------------------------------------------
# 3. Rerank candidates
# ---------------------------------------------------------

def rerank_candidates(
    candidates: list[MemoryCandidate],
    query: str = "",
) -> list[MemoryCandidate]:

    history_query = is_history_query(query)

    scored_candidates = []

    for memory in candidates:

        score = calculate_p2_score(memory)

        # -------------------------------------------------
        # If this is a conversation-history query,
        # strongly prefer L3 memories.
        # -------------------------------------------------

        if history_query and memory.layer == MemoryLayer.L3:
            score += 1.0

        scored_candidates.append((memory, score))

    scored_candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return [item[0] for item in scored_candidates]


# ---------------------------------------------------------
# 4. Select memories
# ---------------------------------------------------------

def select_memories(
    candidates: list[MemoryCandidate],
    top_n: int = 5,
    query: str = "",
) -> list[MemoryCandidate]:

    ranked = rerank_candidates(
        candidates,
        query=query,
    )

    return ranked[:top_n]


# ---------------------------------------------------------
# 5. Get IDs
# ---------------------------------------------------------

def get_selected_memory_ids(
    memories: list[MemoryCandidate],
) -> list[str]:

    return [
        memory.memory_id
        for memory in memories
    ]


# ---------------------------------------------------------
# 6. Main P2 retrieval function
# ---------------------------------------------------------

def retrieve_and_select(
    provider,
    owner_id: str,
    query: str,
    top_n: int = 5,
):

    if is_history_query(query):
        metadata = {
            "record_type": "conversation_turn",
        }

        if is_today_query(query):
            from datetime import datetime, timezone
            metadata["date"] = datetime.now(timezone.utc).date().isoformat()

        candidates = provider.candidate_retrieval(
            owner_id=owner_id,
            query=query,
            limit=50,
            metadata=metadata,
        )
    else:
        candidates = provider.candidate_retrieval(
            owner_id=owner_id,
            query=query,
            limit=20,
        )

    selected_memories = select_memories(
        candidates,
        top_n=top_n,
        query=query,
    )

    selected_ids = get_selected_memory_ids(
        selected_memories
    )

    return selected_memories, selected_ids


# ---------------------------------------------------------
# 7. Build final context through P1
# ---------------------------------------------------------

def retrieve_final_context(
    provider,
    owner_id: str,
    query: str,
    top_n: int = 5,
):

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