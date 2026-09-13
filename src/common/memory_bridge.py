"""
Wires the real P1/P2 memory system into the get_memory_context /
update_memory hooks handle_message() already expects.

Two real gotchas this handles, not just plumbing:

1. IMPORT PATH: memory/ lives at the repo root; retrieval.py has moved
   at least three times over the course of this project (repo root ->
   src/memory/ -> src/). Rather than keep re-guessing its location every
   time it shifts, both the repo root AND src/ go on sys.path here, so
   this keeps working regardless of which of those two retrieval.py
   currently sits in. If it moves again to somewhere neither of these
   covers, this will need another update -- worth settling the folder
   layout with the team once rather than patching this reactively again.

2. DATABASE PATH: MemoryStorage defaults to the *relative* path
   "data/memory.db". A relative path resolves against whatever directory
   you happen to run Python from -- so P1's own tests and this bridge
   could each silently create their OWN separate database despite both
   looking like they're using "the same" default path. Anchoring it to
   the repo root explicitly avoids that split-brain problem.
"""

import sys
from pathlib import Path
from uuid import uuid4

# --- Gotcha 1: cover both known locations for retrieval.py ---
_SRC_DIR = Path(__file__).resolve().parents[1]    # src/common/ -> src/
_REPO_ROOT = _SRC_DIR.parent                        # src/ -> repo root
for _path in (_REPO_ROOT, _SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from memory.models import MemoryContext, MemoryKind, MemoryLayer  # noqa: E402
from memory.service import MemoryService  # noqa: E402
from memory.storage import MemoryStorage  # noqa: E402
from retrieval import retrieve_final_context  # noqa: E402

from schema import Decision, Query  # noqa: E402

# --- Gotcha 2: anchor the DB to the repo root, not the current working dir ---
_DB_PATH = str(_REPO_ROOT / "data" / "memory.db")

# Single shared connection/service for the whole process -- matches how
# test_retrieval.py uses one `service` throughout, and avoids reopening
# the SQLite connection on every call.
_memory_service = MemoryService(storage=MemoryStorage(db_path=_DB_PATH))

# No real user accounts anywhere in this project yet -- single fixed
# owner_id is the right simplification for a single-user hackathon demo.
# Swap this out the moment there's any real notion of "who's talking."
OWNER_ID = "default_user"


def _format_memories(memories) -> str:
    if not memories:
        return "None."

    return "\n".join(
        f"- {memory.content}"
        for memory in memories
    )


def get_memory_context(
    query: Query,
    session_id: str,
) -> str:
    # Always include stable context.
    personality = _memory_service.get_l1_personality(
        owner_id=OWNER_ID
    )

    active_goals = _memory_service.get_l2_active_goals(
        owner_id=OWNER_ID
    )

    current_state = _memory_service.get_l4_current_state(
        owner_id=OWNER_ID,
        session_id=session_id,
    )

    # Retrieve only relevant L3 history.
    history_context = retrieve_final_context(
        provider=_memory_service,
        owner_id=OWNER_ID,
        query=query.interpreted_query,
        top_n=4,
    )

    relevant_history = history_context.memories

    return f"""
[PERSISTENT PERSONALITY — L1]
{_format_memories(personality)}

[ACTIVE GOALS — L2]
{_format_memories(active_goals)}

[CURRENT SESSION STATE — L4]
{_format_memories(current_state)}

[RELEVANT HISTORY — L3]
{_format_memories(relevant_history)}
""".strip()


def update_memory(
    user_input: str,
    response: str,
    query: Query,
    decision: Decision,
    session_id: str,
) -> None:
    """Real implementation of the write-side hook. Logs every completed
    turn as an L4 verified action/event -- the safe, mechanical default.

    NOT implemented here: deciding whether something from this turn is
    ALSO worth promoting to L1 (a stable preference) or L2 (a goal).
    That needs real judgment (an LLM call, or explicit user commands like
    "remember that...") -- a separate, bigger feature, not something to
    invent silently inside a write-back hook.

    stable_key is a fresh uuid per call, not a fixed string -- reusing
    the same stable_key would make each new turn's event silently
    overwrite the previous one (see storage.py's upsert(), which retires
    the old record whenever the same (owner_id, stable_key) is reused).
    L4 is meant to accumulate, not overwrite.
    """
    _memory_service.save_memory(
        owner_id=OWNER_ID,
        stable_key=f"turn_{uuid4().hex}",
        layer=MemoryLayer.L4,
        kind=MemoryKind.ACTION_EVENT,
        content=f"User said: {user_input!r} -> Nishi responded: {response!r}",
        importance=0.3,
        confidence=1.0,
        metadata={"session_id": session_id},
    )


def close() -> None:
    """Call on shutdown to close the SQLite connection cleanly."""
    _memory_service.storage.close()

