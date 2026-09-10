"""
P3 + P4 -- Agent Loop

P3 owns `reason_node`      (decides the next action, or that the goal is met)
P4 owns `act_node`         (actually executes tools, returns an observation)

`AgentState` is the contract between you two -- as long as you both read and
write it the same way, you can build your halves independently and wire them
together at the end.

Run standalone for a quick manual test (uses the placeholder act_node, so it
won't actually do anything real yet -- that's expected until P4 fills it in):
    python agent_loop.py
"""

from typing import Literal, Optional, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from schema import Observation
from tools import TOOLS

load_dotenv()  # reads .env in the current folder and sets os.environ from it


class AgentState(TypedDict):
    goal: str                      # what we're trying to accomplish (from the router)
    next_action: Optional[dict]    # P3 writes this: {"tool": str, "input": dict}
    observation: Optional[dict]    # P4 writes this: an Observation, as a dict
    scratchpad: list                # running log of thought/action/observation, for the demo trace
    is_done: bool                  # P3 sets this True once the goal is satisfied
    final_answer: Optional[str]    # P3 writes this when done
    attempts: int                  # safety valve against infinite loops


class NextStep(BaseModel):
    """What P3's reasoning step decides to do next."""

    is_done: bool = Field(description="True if the goal has been fully accomplished.")
    final_answer: Optional[str] = Field(default=None, description="Set only if is_done is True.")
    tool: Optional[str] = Field(default=None, description="Name of the tool to call next, if not done.")
    tool_input: Optional[dict] = Field(default=None, description="Arguments for that tool.")
    thought: str = Field(description="One short sentence of reasoning, for the demo trace.")


REASON_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Nishi's task-execution reasoner. Given a goal and the results of "
            "actions taken so far, decide the single next tool call needed, or declare "
            "the goal done. Available tools: {tool_names}.",
        ),
        ("human", "Goal: {goal}\n\nProgress so far:\n{scratchpad}"),
    ]
)

# Same reasoning as nishi_pipeline.py's _decision_llm: gemini-3.7-flash's
# free tier is only 20 requests/day, and this shares that same quota.
_reason_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
reason_chain = REASON_PROMPT | _reason_llm.with_structured_output(
    NextStep, method="json_schema"
)

# Derived from tools.py -- adding a tool there is all it takes for the
# reasoning loop to know about it too. Nothing to keep in sync by hand.
TOOL_NAMES = list(TOOLS.keys())


def reason_node(state: AgentState) -> AgentState:
    """P3: decide what happens next, given the goal and what's happened so far."""
    scratchpad_text = "\n".join(state["scratchpad"]) or "(nothing yet)"
    step = reason_chain.invoke(
        {
            "goal": state["goal"],
            "scratchpad": scratchpad_text,
            "tool_names": ", ".join(TOOL_NAMES),
        }
    )

    if step.is_done:
        return {**state, "is_done": True, "final_answer": step.final_answer}

    return {
        **state,
        "next_action": {"tool": step.tool, "input": step.tool_input},
        "scratchpad": state["scratchpad"]
        + [f"Thought: {step.thought} -> calling {step.tool}({step.tool_input})"],
        "attempts": state["attempts"] + 1,
    }


def should_continue(state: AgentState) -> Literal["act", "end"]:
    if state["is_done"] or state["attempts"] >= 6:
        return "end"
    return "act"


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
    and record the resulting Observation."""
    action = state["next_action"]
    obs = execute_tool(action["tool"], action["input"])
    return {
        **state,
        "observation": obs.model_dump(),
        "scratchpad": state["scratchpad"]
        + [f"Observation: success={obs.success}, result={obs.result}, error={obs.error}"],
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)
    graph.add_conditional_edges("reason", should_continue, {"act": "act", "end": END})
    graph.add_edge("act", "reason")
    graph.set_entry_point("reason")
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke(
        {
            "goal": "Add 'buy groceries' to my task list for tomorrow",
            "next_action": None,
            "observation": None,
            "scratchpad": [],
            "is_done": False,
            "final_answer": None,
            "attempts": 0,
        }
    )
    print(result["final_answer"])
    print("\n".join(result["scratchpad"]))