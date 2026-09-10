"""
P3's actual job: wire the schemas/prompts into working calls, and connect
the Decision output to the agent loop.

Two Gemini calls happen per turn at most:
  1. understand_query   -- cheap, Flash-Lite
  2. make_decision      -- Flash, since it's also drafting the reply

If the decision needs real execution, a third+ set of calls happens inside
the agent loop (reason_node), each cycle: reason -> act -> observe -> repeat.
"""

from typing import Callable

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from agent_loop import TOOLS, build_graph, execute_tool
from prompt_templets import DECISION_PROMPT, QUERY_PROMPT
from schema import Decision, Query

load_dotenv()  # reads .env in the current folder and sets os.environ from it

# --- Stage 1: understand the query ---
_query_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
query_chain = QUERY_PROMPT | _query_llm.with_structured_output(Query, method="json_schema")


def understand_query(user_input: str) -> Query:
    return query_chain.invoke({"user_input": user_input})


# --- Stage 2: decide what to do about it (and draft the reply, if talk) ---
# Using flash-lite here too, not gemini-3.7-flash: the free tier caps
# 3.7-flash at just 20 requests/day, while flash-lite's daily quota is
# roughly 25x higher (~500/day) -- far more headroom for a hackathon's
# worth of iteration. Swap back to gemini-3.7-flash once billing is linked
# or if you want the quality upgrade specifically for the final demo.
_decision_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
decision_chain = DECISION_PROMPT | _decision_llm.with_structured_output(Decision, method="json_schema")


def _describe_tools() -> str:
    """Builds a readable tool list from tools.TOOLS + each function's own
    docstring, so DECISION_PROMPT always reflects whatever's actually
    registered -- no separate list to keep in sync by hand."""
    if not TOOLS:
        return "(no tools currently available)"
    lines = []
    for name, fn in TOOLS.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "no description"
        lines.append(f"- {name}: {doc}")
    return "\n".join(lines)


def make_decision(query: Query, memory_context: str) -> Decision:
    return decision_chain.invoke(
        {
            "query": query.model_dump_json(),
            "memory_context": memory_context,
            "available_tools": _describe_tools(),
        }
    )


# --- Stage 3: orchestrate the whole turn ---
_agent_app = build_graph()


def handle_message(
    user_input: str,
    get_memory_context: Callable[[Query], str],
    verbose: bool = False,
) -> str:
    """
    get_memory_context: Person 2's function. Only called when
    query.memory_required is True, so a plain "how's it going" never
    triggers a memory lookup.

    verbose: if True, prints the intermediate Query and Decision -- handy
    for testing without burning extra API calls re-deriving them separately.
    """
    query = understand_query(user_input)
    if verbose:
        print(
            f"  Query -> intent={query.intent}, tone={query.tone}, "
            f"memory_required={query.memory_required}"
        )

    memory_context = get_memory_context(query) if query.memory_required else "none needed"
    decision = make_decision(query, memory_context)
    if verbose:
        print(
            f"  Decision -> type={decision.type}, "
            f"execution_mode={decision.execution_mode}, tool={decision.tool}"
        )

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
            f"Goal: {decision.goal}. Objective: {decision.objective}. "
            f"Success looks like: {decision.success_condition}.",
            f"First attempt: called {decision.tool}({decision.tool_arguments}) "
            f"-> failed: {observation.error or 'unknown error'}",
        ]
        initial_attempts = 1
    else:
        # Decision didn't settle on an initial tool -- nothing to try
        # directly, so go straight to reasoning.
        seed_scratchpad = [
            f"Goal: {decision.goal}. Objective: {decision.objective}. "
            f"Success looks like: {decision.success_condition}."
        ]
        initial_attempts = 0

    # --- Section 7: agentic execution -- only reached when the direct
    # attempt failed, or there was no initial tool to try at all. ---
    result = _agent_app.invoke(
        {
            "goal": decision.goal,
            "next_action": None,
            "observation": None,
            "scratchpad": seed_scratchpad,
            "is_done": False,
            "final_answer": None,
            "attempts": initial_attempts,
        }
    )
    return result["final_answer"] or decision.answer


if __name__ == "__main__":
    def fake_memory(query: Query) -> str:
        return "User's name is not yet known. No prior goals recorded."

    print(handle_message("How was your day?", fake_memory))
    print(handle_message("Add 'buy groceries' to my task list for tomorrow", fake_memory))