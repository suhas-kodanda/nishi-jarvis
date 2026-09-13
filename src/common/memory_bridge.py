import uuid
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
from typing import Literal, Optional
from uuid import uuid4

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

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

from common.schema import Decision, Query  # noqa: E402

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

class _ExtractedFact(BaseModel):
    contains_fact: bool
    layer: Optional[Literal["L1", "L2"]] = None
    fact: Optional[str] = None
    stable_key: Optional[str] = Field(
        default=None,
        description=(
            "A short stable category key for this fact, such as "
            "'user_name', 'response_style', 'main_goal', or 'current_project'."
        ),
    )


_FACT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract only stable user facts worth remembering.

Use L1 for personality/preferences and L2 for ongoing goals.
Return contains_fact=False for ordinary chat or one-time requests.

When contains_fact is true, also return a stable_key. Reuse the same
stable_key when a later fact updates the same topic.

If true, return:
- fact: a short fact
- stable_key: a reusable category such as user_name,
  response_style, main_goal, or current_project."""
    ),
    ("human", "User: {user_input}\nNishi: {response}"),
])

_fact_llm = None
_fact_chain = None


def get_fact_chain():
    global _fact_llm, _fact_chain

    if _fact_chain is None:
        _fact_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0,
        )

        _fact_chain = _FACT_PROMPT | _fact_llm.with_structured_output(
            _ExtractedFact,
            method="json_schema",
        )

    return _fact_chain

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
    """Real implementation of the write-side hook.

    Two things happen here, not one:

    1. Every turn logs to L4 as before, tagged with session_id -- the
       safe, mechanical default, unchanged.
    2. A separate, cheap classification call decides if THIS turn also
       contains something durable worth promoting to L1 (personality) or
       L2 (goal). Skipped entirely for execute-mode tasks (scheduling,
       sending an email) -- those are inherently unlikely to contain a
       personal fact, and skipping saves both the API call and the worse
       of the two ways this classifier can be wrong: a missed fact just
       stays in L4 (low stakes, another chance if restated); a
       wrongly-flagged fact permanently pollutes L1/L2, which now gets
       included in every future turn unconditionally via
       get_l1_personality/get_l2_active_goals above.

    stable_key is a fresh uuid per call for both L4 and any promoted
    fact -- reusing the same stable_key would make each new save
    silently overwrite the previous one (see storage.py's upsert()).
    Facts currently accumulate rather than merge/update -- a known
    simplification, not something this pass covers.
    """
    _memory_service.save_memory(
        owner_id=OWNER_ID,
        stable_key=f"l4_{uuid.uuid4().hex}",
        layer=MemoryLayer.L4,
        kind=MemoryKind.ACTION_EVENT,
        content=f"User said: {user_input!r} -> Nishi responded: {response!r}",
        importance=0.3,
        confidence=1.0,
        metadata={"session_id": session_id},
    )

    try:
        skip = decision.execution_mode == "execute"
        extracted = None if skip else get_fact_chain().invoke(
            {"user_input": user_input, "response": response}
        )
    except Exception:
        extracted = None  # classification failing shouldn't affect anything else

    _memory_service.save_memory(
        owner_id=OWNER_ID,
        stable_key=f"l3_{uuid.uuid4().hex}",
        layer=MemoryLayer.L3,
        kind=MemoryKind.HISTORY,
        content=f"User: {user_input}\nNishi: {response}",
        importance=0.5,
        confidence=1.0,
        metadata={"session_id": session_id},
    )

    try:
        extracted = get_fact_chain().invoke({
            "user_input": user_input,
            "response": response,
        })
    except Exception:
        extracted = None

    if (
        extracted
        and extracted.contains_fact
        and extracted.layer in {"L1", "L2"}
        and extracted.fact
        and extracted.stable_key
    ):
        layer = (
            MemoryLayer.L1
            if extracted.layer == "L1"
            else MemoryLayer.L2
        )

        kind = (
            MemoryKind.PERSONALITY
            if extracted.layer == "L1"
            else MemoryKind.GOAL
        )

        _memory_service.save_memory(
            owner_id=OWNER_ID,
            stable_key=f"{layer.value.lower()}_{extracted.stable_key}",
            layer=layer,
            kind=kind,
            content=extracted.fact,
            importance=0.8,
            confidence=1.0,
            metadata={"status": "active"} if layer == MemoryLayer.L2 else {},
        )


def close() -> None:
    """Call on shutdown to close the SQLite connection cleanly."""
    _memory_service.storage.close()