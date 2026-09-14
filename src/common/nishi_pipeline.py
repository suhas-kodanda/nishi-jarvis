"""
P3's actual job: wire the schemas/prompts into working calls, and connect
the Decision output to the agent loop.

Two Gemini calls happen per turn at most:
  1. understand_query   -- cheap, Flash-Lite
  2. make_decision      -- Flash, since it's also drafting the reply

If the decision needs real execution, a third+ set of calls happens inside
the agent loop (reason_node), each cycle: reason -> act -> observe -> repeat.
"""

from typing import Callable, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from dotenv import load_dotenv

from common.agent_loop import build_graph, execute_tool
from common.memory_bridge import get_memory_context
from common.prompt_templets import DECISION_PROMPT, QUERY_PROMPT
from common.schema import Decision, Query

load_dotenv()  # reads .env in the current folder and sets os.environ from it

# Which sessions currently have a task paused mid-way, waiting on an
# ask_user answer. In-memory, matching MemorySaver's own scope in
# agent_loop.py -- both live only as long as this one chat.py process.
_pending_interrupts: set[str] = set()

# --- Stage 1: understand the query ---
# _query_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
# query_chain = QUERY_PROMPT | _query_llm.with_structured_output(Query, method="json_schema")
_query_llm = None
_query_chain = None


def get_query_chain():
    global _query_llm, _query_chain

    if _query_chain is None:
        _query_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0,
        )

        _query_chain = QUERY_PROMPT | _query_llm.with_structured_output(
            Query,
            method="json_schema",
        )

    return _query_chain


def understand_query(user_input: str) -> Query:
    return get_query_chain().invoke({"user_input": user_input})


# --- Stage 2: decide what to do about it (and draft the reply, if talk) ---
# Using flash-lite here too, not gemini-3.7-flash: the free tier caps
# 3.7-flash at just 20 requests/day, while flash-lite's daily quota is
# roughly 25x higher (~500/day) -- far more headroom for a hackathon's
# worth of iteration. Swap back to gemini-3.7-flash once billing is linked
# or if you want the quality upgrade specifically for the final demo.
# _decision_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
# decision_chain = DECISION_PROMPT | _decision_llm.with_structured_output(Decision, method="json_schema")
_decision_llm = None
_decision_chain = None


def get_decision_chain():
    global _decision_llm, _decision_chain

    if _decision_chain is None:
        _decision_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0,
        )

        _decision_chain = DECISION_PROMPT | _decision_llm.with_structured_output(
            Decision,
            method="json_schema",
        )

    return _decision_chain


def make_decision(query: Query, memory_context: str, recent_context: str) -> Decision:
    """Wraps decision_chain.invoke with ONE retry on failure.

    Decision's own model_validator has rejected real output a few times
    in practice (e.g. execution_mode='direct' with a tool set, or an
    empty 'answer') -- each one previously needed a manual prompt fix
    after the fact. This gives the model one chance to see its own
    actual error and self-correct, before falling back to the existing,
    already-tested failure path in handle_message(). Same underlying
    principle as the reasoning loop already self-correcting an invented
    tool name -- just applied one stage earlier, to the schema itself.

    Deliberately ONE retry, not a loop -- if it fails twice, something
    more fundamental is wrong, and handle_message()'s existing generic
    error message is the right response, not another silent attempt.
    """
    payload = {
        "query": query.model_dump_json(),
        "memory_context": memory_context,
        "recent_context": recent_context,
    }
    try:
        return get_decision_chain().invoke(payload)
    except Exception as first_error:
        retry_payload = {
            **payload,
            "recent_context": (
                f"{recent_context}\n\n[SYSTEM: your previous attempt at "
                f"this exact decision failed schema validation -- fix it "
                f"this time. Error: {first_error}]"
            ),
        }
        try:
            return get_decision_chain().invoke(retry_payload)
        except Exception:
            raise first_error  # preserve the ORIGINAL error, not the retry's


# --- Stage 3: orchestrate the whole turn ---
_agent_app = build_graph()


def _handle_agent_result(result: dict, session_id: str) -> str:
    """Shared handling for whatever _agent_app.invoke() returns, whether
    from a fresh start or a resume -- ANY invocation can pause again (a
    multi-step task might need more than one clarifying question), so
    this can't be special-cased to just one call site."""
    if "__interrupt__" in result:
        _pending_interrupts.add(session_id)
        return result["__interrupt__"][0].value

    if result.get("final_answer"):
        return result["final_answer"]

    # Loop exhausted its attempts without completing -- say so honestly.
    # Silently falling back to decision.answer here would present a real
    # failure as if it succeeded, which is exactly what got fixed.
    recent = "; ".join(result.get("scratchpad", [])[-3:]) or "no progress was made"
    return f"I wasn't able to finish this after a few attempts. What I tried: {recent}"


def handle_message(
    user_input: str,
    get_memory_context: Callable[[Query, str], str],
    update_memory: Optional[Callable[[str, str, Query, Decision, str], None]] = None,
    session_id: str = "default_session",
    recent_context: str = "(start of conversation)",
    verbose: bool = False,
) -> str:
    """
    get_memory_context: Person 2's read-side function. Only called when
    query.memory_required is True, so a plain "how's it going" never
    triggers a memory lookup.

    update_memory: Person 1/2's write-side function, symmetric with
    get_memory_context above. Called once, after the response is fully
    decided, with (user_input, response, query, decision).

    recent_context: the last couple of raw exchanges from THIS chat
    session, supplied by the caller (chat.py keeps this, not stored in
    the database). Deliberately separate from get_memory_context/L1-L3 --
    this is for "what did I just say" (needs the literal recent text,
    not a lexical search that misses pronouns), not long-term recall.
    Zero extra API calls: just more text in the existing make_decision
    call, not a new request. Keep the caller's buffer small (a handful
    of turns) -- every character here is paid for on every single call.

    verbose: if True, prints the intermediate Query and Decision -- handy
    for testing without burning extra API calls re-deriving them separately.
    """
    if session_id in _pending_interrupts:
        # This session has a task paused mid-way, waiting on exactly
        # this answer -- it's not a new request to classify, it's the
        # missing piece of an existing one. Skip understand_query/
        # make_decision entirely and resume the paused graph directly.
        _pending_interrupts.discard(session_id)
        config = {"configurable": {"thread_id": session_id}}
        result = _agent_app.invoke(Command(resume=user_input), config=config)
        response = _handle_agent_result(result, session_id)

        if update_memory is not None:
            try:
                # No real Query/Decision exists for a resumed turn -- a
                # minimal placeholder keeps update_memory's existing
                # contract intact without inventing a fake classification.
                placeholder_query = Query(
                    interpreted_query=user_input, memory_required=False,
                    intent="task_completion", tone="neutral", response_type="task_result",
                )
                placeholder_decision = Decision(type="task", execution_mode="execute", answer=response)
                update_memory(user_input, response, placeholder_query, placeholder_decision, session_id)
            except Exception as e:
                if verbose:
                    print(f"  [update_memory failed on resume, response still returned: {e}]")
        return response

    try:
        query = understand_query(user_input)
    except Exception as e:
        if verbose:
            print(f"  [understand_query failed: {e}]")
        return "Sorry, I'm having trouble understanding that right now -- please try again in a moment."

    if verbose:
        print(
            f"  Query -> intent={query.intent}, tone={query.tone}, "
            f"memory_required={query.memory_required}"
        )

    try :
        memory_context = get_memory_context(
            query,
            session_id,
        )
    except Exception as e:
        if verbose:
            print(f"  [get_memory_context failed: {e}]")
        memory_context = "Persistent memory is unavailable right now."

    try:
        decision = make_decision(query, memory_context, recent_context)
    except Exception as e:
        if verbose:
            print(f"  [make_decision failed: {e}]")
        return "Sorry, I'm having trouble deciding how to respond right now -- please try again in a moment."

    if verbose:
        print(
            f"  Decision -> type={decision.type}, "
            f"execution_mode={decision.execution_mode}, tool={decision.tool}"
        )

    try:
        response = _resolve_response(decision, memory_context, session_id, verbose)
    except Exception as e:
        if verbose:
            print(f"  [_resolve_response failed: {e}]")
        # Deliberately NOT falling back to decision.answer here -- that's
        # the placeholder acknowledgment, and returning it after a genuine
        # crash would be the same "looks like success" problem as the
        # loop-timeout case below.
        response = "Something went wrong while I was working on that -- please try again."

    if update_memory is not None:
        try:
            update_memory(user_input, response, query, decision,session_id)
        except Exception as e:
            # The response is already correct and ready -- a broken
            # write-back shouldn't cost the user the answer they were
            # waiting for. Surface it in verbose mode, don't raise.
            if verbose:
                print(f"  [update_memory failed, response still returned as normal: {e}]")

    return response


def _goal_context(decision: Decision) -> str:
    """Formats the goal/objective/success_condition trio for the reasoning
    loop's scratchpad, guarding against any of them being None -- Decision
    marks all three Optional, and an unguarded f-string would otherwise
    literally inject the word "None" into what gets fed back to the LLM."""
    goal = decision.goal or "not specified"
    objective = decision.objective or "not specified"
    success = decision.success_condition or "not specified"
    return f"Goal: {goal}. Objective: {objective}. Success looks like: {success}."


def _resolve_response(decision: Decision, memory_context: str, session_id: str, verbose: bool) -> str:
    """Everything after Decision is made: figure out the actual response
    text, whichever path (direct answer, fast tool call, or full loop)
    it takes. Split out from handle_message so update_memory above has
    exactly one place to hook in, regardless of which path ran."""

    # Conversation, or a task answerable without a tool: Decision already
    # has the full answer -- no further calls needed.
    if decision.type == "conversation" or decision.execution_mode == "direct":
        return decision.answer

    # --- Section 6: simple task -- try Decision's own tool call directly. ---
    # Zero extra LLM calls: Decision already picked the tool and arguments,
    # so just run it and check the Observation.
    if decision.tool is not None:
        observation = execute_tool(decision.tool, decision.tool_arguments or {})
        if verbose:
            print(
                f"  Direct tool call -> success={observation.success}, "
                f"result={observation.result}"
            )
        if observation.success:
            return observation.result or decision.answer

        # Falls through to Section 7 below, seeded with what already failed
        # so reason_node doesn't just blindly repeat the same attempt.
        seed_scratchpad = [
            _goal_context(decision),
            f"First attempt: called {decision.tool}({decision.tool_arguments}) "
            f"-> failed: {observation.error or 'unknown error'}",
        ]
        initial_attempts = 1
    else:
        # Decision didn't settle on an initial tool -- nothing to try
        # directly, so go straight to reasoning.
        seed_scratchpad = [_goal_context(decision)]
        initial_attempts = 0

    # --- Section 7: agentic execution -- only reached when the direct
    # attempt failed, or there was no initial tool to try at all. ---
    result = _agent_app.invoke(
        {
            "goal": decision.goal or "not specified -- infer from progress so far",
            "next_action": None,
            "scratchpad": seed_scratchpad,
            "is_done": False,
            "final_answer": None,
            "attempts": initial_attempts,
            "memory_context": memory_context,
        },
        config={"configurable": {"thread_id": session_id}},
    )

    return _handle_agent_result(result, session_id)


if __name__ == "__main__":
    def fake_memory(query: Query, session_id: str) -> str:
        return "User's name is not yet known. No prior goals recorded."

    def fake_update(user_input: str, response: str, query: Query, decision: Decision, session_id: str) -> None:
        print(f"  [would update memory here: heard '{user_input}', decided type={decision.type}]")

    print(handle_message("How was your day?", fake_memory, fake_update))
    print(handle_message("Add 'buy groceries' to my task list for tomorrow", fake_memory, fake_update))