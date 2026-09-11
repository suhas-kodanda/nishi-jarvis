"""
Tool implementations.

CONTRACT for adding a new tool (read this before writing one):

  1. Write a function with real typed parameters, not **kwargs:
         def my_tool(some_arg: str, optional_arg: str | None = None) -> str
     Typed parameters matter now, not just style -- agent_loop.py wraps
     these into real tool-calling schemas the model is structurally
     bound to (via .bind_tools()), so the parameter names and types
     ARE what the model is told to provide. **kwargs would show up as
     an opaque, useless "object" with no argument names at all.
       - Return a short, user-facing sentence describing what happened.
         This string becomes Observation.result directly and may be
         shown to the user as-is (e.g. "Added 'buy groceries' for
         tomorrow.", not "{'status': 'ok', 'id': 4}").
       - If something goes wrong, just `raise` a normal exception
         (ValueError, KeyError, a requests exception, whatever fits
         naturally). You do NOT need to import Observation, and you
         do NOT need to catch anything yourself -- agent_loop.py's
         execute_tool() wraps every call in try/except and turns a
         raised exception into a failed Observation automatically.

  2. Register it in TOOLS at the bottom of this file:
         TOOLS["my_tool_name"] = my_tool
     Use a clear, descriptive name -- this is literally what the LLM
     sees and picks from when deciding which tool to call, so
     "check_calendar" is good, "tool3" is not.

That's the whole contract. Nothing outside this file needs to change
when you add a tool -- agent_loop.py, nishi_pipeline.py, and the
prompts all read from the TOOLS dict below, not from anything hardcoded.

Five real, working tools are implemented below -- a genuine in-memory
task manager (create/list/update/complete/delete), not placeholders.
Replace the in-memory list with a real database whenever that's ready;
the function signatures and TOOLS registration don't need to change.
"""

from typing import Callable, Optional

# --- Task management: real logic, in-memory storage for now ---

_TASKS: list[dict] = []
_next_id = [1]  # boxed in a list so it's mutable from inside the functions below


def create_task(title: str, due_date: Optional[str] = None) -> str:
    """Adds a task to the task list."""
    if not title:
        raise ValueError("A task title is required.")
    task_id = str(_next_id[0])
    _next_id[0] += 1
    due = due_date or "no due date"
    _TASKS.append({"id": task_id, "title": title, "due_date": due, "done": False})
    return f"Added '{title}' (id={task_id}, due: {due})."


def list_tasks() -> str:
    """Lists everything currently on the task list."""
    if not _TASKS:
        return "Your task list is empty."
    lines = [
        f"- [{t['id']}] {t['title']} (due: {t['due_date']})" + (" \u2713" if t["done"] else "")
        for t in _TASKS
    ]
    return "Your tasks:\n" + "\n".join(lines)


def _find_task(task_id: str) -> dict:
    for t in _TASKS:
        if t["id"] == task_id:
            return t
    raise ValueError(f"No task found with id {task_id}. Use list_tasks to see valid ids.")


def update_task(task_id: str, title: Optional[str] = None, due_date: Optional[str] = None) -> str:
    """Updates an existing task's title or due date."""
    task = _find_task(task_id)
    if title:
        task["title"] = title
    if due_date:
        task["due_date"] = due_date
    return f"Updated task {task_id}: '{task['title']}' (due: {task['due_date']})."


def complete_task(task_id: str) -> str:
    """Marks a task as complete."""
    task = _find_task(task_id)
    task["done"] = True
    return f"Marked '{task['title']}' as complete."


def delete_task(task_id: str) -> str:
    """Deletes a task from the list."""
    task = _find_task(task_id)
    _TASKS.remove(task)
    return f"Deleted task '{task['title']}'."


def find_calendar_events(date: str, search_term: Optional[str] = None) -> str:
    """Finds calendar events matching a date or search term."""
    raise NotImplementedError("find_calendar_events: P4 to implement.")


def create_calendar_event(title: str, start_time: str, end_time: Optional[str] = None) -> str:
    """Creates a new calendar event."""
    raise NotImplementedError("create_calendar_event: P4 to implement.")


def update_calendar_event(event_id: str, title: Optional[str] = None, start_time: Optional[str] = None) -> str:
    """Updates an existing calendar event's time or details."""
    raise NotImplementedError("update_calendar_event: P4 to implement.")


def delete_calendar_event(event_id: str) -> str:
    """Deletes a calendar event."""
    raise NotImplementedError("delete_calendar_event: P4 to implement.")


def search_emails(query: str) -> str:
    """Searches emails matching a sender, subject, or keyword."""
    raise NotImplementedError("search_emails: P4 to implement.")


def read_email(email_id: str) -> str:
    """Reads the full content of a specific email."""
    raise NotImplementedError("read_email: P4 to implement.")


def draft_email(to: str, subject: str, body: str) -> str:
    """Creates a draft email without sending it."""
    raise NotImplementedError("draft_email: P4 to implement.")


def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email."""
    raise NotImplementedError("send_email: P4 to implement.")


# Real implementation, not a stub -- safe by construction (only numeric
# literals and +-*/() are ever evaluated; no eval() on raw input).
import ast
import operator

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only numbers and + - * / ** ( ) are allowed.")


def calculate(expression: str) -> str:
    """Evaluates a math expression precisely (for anything beyond simple mental math)."""
    if not expression:
        raise ValueError("An 'expression' argument is required.")
    result = _safe_eval(ast.parse(expression, mode="eval").body)
    return f"{expression} = {result}"


# NOT registered below yet -- ask_user doesn't fit the other tools' contract
# (return a string / raise). It needs the loop to genuinely pause for a real
# human reply, which requires either LangGraph's interrupt()+checkpointer
# support (the real fix) or treating it as an early is_done with the
# question as final_answer (simpler, loses loop continuity). Team decision,
# not made here -- see the message this was discussed in.


# --- P4: add your real tools above this line ---


# Single source of truth: agent_loop.py dispatches through this dict,
# and derives the tool-name list it tells the LLM about from its keys.
# Adding a tool means adding one line here -- nothing else to touch.
# Only registering what's confirmed needed right now. Every tool here
# costs real tokens on every decision + reasoning call (see describe_tools()
# below) -- a tool should exist because the LLM genuinely can't do the
# thing itself (external state, side effect, or private data), not because
# it's a traditional "agent tool." create_task/list_tasks/update_task/
# complete_task/delete_task are all real external state changes -- Nishi's
# own task list, which nothing else can read or modify.
#
# Deliberately NOT registered yet, though the functions exist above,
# ready to add with one line each once actually needed:
#   calculate               -- LLM handles ordinary math fine; add back
#                               only if complex/precise computation shows
#                               up as a real need in testing.
#   search_emails/read_email/draft_email/send_email
#                            -- only meaningful once Gmail is actually
#                               connected. No point exposing tools that
#                               can't do anything real yet.
#   find_calendar_events/create_calendar_event/update_calendar_event/
#   delete_calendar_event   -- same logic: only register once a real
#                               calendar integration exists. Note the
#                               distinction that matters here: "what's
#                               next Monday's date" needs no tool at all,
#                               the LLM just knows that -- only reading or
#                               modifying the user's *actual* calendar
#                               needs one.
#
# ask_user isn't listed at all -- see the architecture note above it.
TOOLS: dict[str, Callable[..., str]] = {
    "create_task": create_task,
    "list_tasks": list_tasks,
    "update_task": update_task,
    "complete_task": complete_task,
    "delete_task": delete_task,
}


def describe_tools() -> str:
    """One line per tool, pulled from each function's own docstring --
    kept deliberately short since this text gets sent to the LLM on every
    call. Used by both DECISION_PROMPT and REASON_PROMPT so there's one
    place tool descriptions live, not two lists that can drift apart."""
    if not TOOLS:
        return "(no tools currently available)"
    lines = []
    for name, fn in TOOLS.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "no description"
        lines.append(f"- {name}: {doc}")
    return "\n".join(lines)