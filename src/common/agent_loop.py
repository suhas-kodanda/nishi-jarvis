"""
P3 + P4 -- Agent Loop

P3 owns `reason_node`      (decides the next action, via native tool calling)
P4 owns the tool bodies    (in tools.py -- what actually happens when called)

`AgentState` is the contract between you two -- as long as you both read and
write it the same way, you can build your halves independently and wire them
together at the end.

Reasoning uses real native tool calling (.bind_tools()), not a prompt
describing tools in prose -- the model is structurally bound to the
functions registered in tools.py's TOOLS dict, including their real
argument names and types.

Run standalone for a quick manual test (uses tools.py's real, working
task-management tools):
    python agent_loop.py
"""

from typing import Literal, Optional, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from dotenv import load_dotenv

from common.schema import Observation
from common.tools import TOOLS

load_dotenv()  # reads .env in the current folder and sets os.environ from it


class AgentState(TypedDict):
    goal: str                      # what we're trying to accomplish (from the router)
    next_action: Optional[dict]    # P3 writes this: {"tool": str, "input": dict}
    scratchpad: list               # running log of action -> observation, fed back each turn
    is_done: bool                  # P3 sets this True once the goal is satisfied
    final_answer: Optional[str]    # P3 writes this when done
    attempts: int                  # safety valve against infinite loops
    memory_context: str      # P3 reads this, set by the router (P1) from memory_bridge

REASON_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Nishi's goal reasoner. Choose ONE appropriate tool "
            "that directly advances the goal. Use the tool's description "
            "and arguments. After each result, assess progress: continue, "
            "adapt/replan, or finish when the goal is satisfied. Never use "
            "unrelated tools or invent information."
        ),
        (
            "human",
            "Persistent context:\n{memory_context}\n\n"
            "Goal: {goal}\n\n"
            "Progress:\n{scratchpad}",
        ),
    ]
)

# Wrap tools.py's plain functions into real tool-calling schemas, purely
# here -- tools.py itself stays free of any LangChain-specific code, so
# P4 only ever needs to write plain, typed Python functions.
_LC_TOOLS = [
    StructuredTool.from_function(func=fn, name=name, description=(fn.__doc__ or "").strip())
    for name, fn in TOOLS.items()
]

# Same reasoning as nishi_pipeline.py's _decision_llm: gemini-3.7-flash's
# free tier is only 20 requests/day, and this shares that same quota.
# _reason_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
# reason_chain = REASON_PROMPT | _reason_llm.bind_tools(_LC_TOOLS)
_reason_llm = None
reason_chain = None


def get_reason_chain():
    global _reason_llm, reason_chain

    if reason_chain is None:
        _reason_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0,
        )
        reason_chain = REASON_PROMPT | _reason_llm.bind_tools(_LC_TOOLS)

    return reason_chain


def reason_node(state: AgentState) -> AgentState:
    """P3: decide what happens next -- either call a tool or finish.

    Native tool calling replaces the old NextStep schema entirely: the
    model either returns tool_calls (it wants to act) or plain text
    content (it's done, and that text IS the final answer). One less
    thing to keep in sync -- is_done/final_answer used to be separate
    fields to maintain by hand; now they're just what "no tool call"
    naturally means.
    """
    scratchpad_text = "\n".join(state["scratchpad"]) or "(nothing yet)"
    response = get_reason_chain().invoke({"goal": state["goal"], "scratchpad": scratchpad_text, "memory_context": state["memory_context"]},)

    if not response.tool_calls:
        # response.content isn't guaranteed to be a plain string here --
        # confirmed in practice: Gemini's tool-calling responses can
        # return it as a list of content blocks (e.g.
        # [{"type": "text", "text": "..."}]) instead. Fixed at the
        # source so every caller gets a clean string, not just wherever
        # display happens to patch it afterward.
        content = response.content
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return {**state, "is_done": True, "final_answer": content}

    call = response.tool_calls[0]  # one action per turn, same as before
    return {
        **state,
        "next_action": {"tool": call["name"], "input": call["args"]},
        "scratchpad": state["scratchpad"] + [f"Called {call['name']}({call['args']})"],
        "attempts": state["attempts"] + 1,
    }


def should_continue(state: AgentState) -> Literal["act", "ask_user", "end"]:
    if state["is_done"] or state["attempts"] >= 6:
        return "end"
    if state["next_action"]["tool"] == "ask_user":
        return "ask_user"
    return "act"


def ask_user_node(state: AgentState) -> AgentState:
    """Dedicated, isolated node for pausing -- deliberately contains
    NOTHING except the interrupt() call itself.

    This matters: LangGraph re-executes a node FROM THE TOP on resume,
    not from mid-function -- confirmed by testing, not assumed. Putting
    interrupt() inside reason_node (the first design tried) meant every
    resume would re-run the LLM call too, and only correctly pick up the
    answer if the model happened to re-request the exact same ask_user
    call with the exact same question on that replay -- not guaranteed,
    especially given how inconsistent flash-lite has already proven to
    be elsewhere in this project. An isolated node makes the replay
    trivial and deterministic: there's nothing here to go wrong except
    the interrupt() call itself.
    """
    question = state["next_action"]["input"].get("question", "Can you clarify?")
    answer = interrupt(question)
    return {
        **state,
        "scratchpad": state["scratchpad"] + [
            f"Asked user: {question}",
            f"User answered: {answer}",
        ],
    }


# execute_tool now dispatches for real through tools.TOOLS -- nothing left
# here for P4 to fill in. Adding a tool means editing tools.py only.
def execute_tool(tool: str, tool_input: dict) -> Observation:
    """
    Looks up `tool` by name in tools.TOOLS and calls it with tool_input
    as keyword arguments. Tool functions just return a result string or
    raise an exception on failure (see tools.py's contract) -- this is
    the one place that wraps either outcome into a proper Observation.
    """
    fn = TOOLS.get(tool)
    if fn is None:
        return Observation(
            success=False,
            error=f"Unknown tool '{tool}'. Available tools: {', '.join(TOOLS) or 'none registered'}",
        )
    try:
        result = fn(**(tool_input or {}))
        return Observation(success=True, result=result)
    except Exception as e:
        return Observation(success=False, error=str(e))


def act_node(state: AgentState) -> AgentState:
    """P3: unpack the action reason_node chose, run it via execute_tool,
    and record what happened in the scratchpad."""
    action = state["next_action"]
    obs = execute_tool(action["tool"], action["input"])
    outcome = obs.result if obs.success else f"FAILED: {obs.error}"
    return {
        **state,
        "scratchpad": state["scratchpad"] + [f"Result: {outcome}"],
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)
    graph.add_node("ask_user", ask_user_node)
    graph.add_conditional_edges("reason", should_continue, {"act": "act", "ask_user": "ask_user", "end": END})
    graph.add_edge("act", "reason")
    graph.add_edge("ask_user", "reason")
    graph.set_entry_point("reason")
    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke(
        {
            "goal": "Add 'buy groceries' to my task list for tomorrow",
            "next_action": None,
            "scratchpad": [],
            "is_done": False,
            "final_answer": None,
            "attempts": 0,
            "memory_context": "none",
        },
        config={"configurable": {"thread_id": "manual-test"}},
    )
    print(result["final_answer"])
    print("\n".join(result["scratchpad"]))