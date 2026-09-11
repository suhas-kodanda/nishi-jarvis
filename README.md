# NISHI - P2 Memory Retrieval

## Responsibility

P2 is responsible for retrieving relevant memories for the current user query.

P2 does not manage memory storage.

P1 owns memory creation, updating, deletion and storage.

## Input

P2 receives:

- Current user query
- Current conversation
- Stored memories from P1

## Retrieval Pipeline

Query
↓
Tokenization
↓
Stop-word removal
↓
Basic synonym expansion
↓
Relevance calculation
↓
Importance scoring
↓
Ranking
↓
Top-N retrieval
↓
Layer organization
↓
Final memory context

## Memory Layers

L1 - User personality and preferences

L2 - User goals and ambitions

L3 - Relevant conversation/history

L4 - Current conversation

## Output

P2 returns:

{
    "L1": [...],
    "L2": [...],
    "L3": [...],
    "L4": [...]
}

## Important Design Decision

The current implementation does not use:

- Embeddings
- Vector databases
- LLM calls
- LangChain memory

This keeps retrieval lightweight and reduces unnecessary API/token cost.

## P1 Interface

P1 should eventually provide the stored memories through a function/interface such as:

get_memories()

P2 only reads the returned memories and does not modify them.

## P2 Interface

Main function:

build_context(
    query,
    current_conversation,
    memories,
    top_n=5
)

This returns the final memory context for P3.