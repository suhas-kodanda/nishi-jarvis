"""
Tool implementations.

CONTRACT for adding a new tool (read this before writing one):

  1. Write a plain function: def my_tool(**kwargs) -> str
       - Pull whatever arguments you need out of kwargs.
       - Return a short, user-facing sentence describing what happened.
         This string becomes Observation.result directly and may be
         shown to the user as-is, so write it as a sentence, not raw
         data (e.g. "Added 'buy groceries' for tomorrow.", not
         "{'status': 'ok', 'id': 4}").
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

Two real (if simple) tools are implemented below as a working example
of the pattern -- an in-memory task list, not a placeholder. Replace
the in-memory list with a real database/API call whenever that's ready;
the function signature and TOOLS registration don't need to change.
"""

from typing import Callable

# --- Example tools: real logic, in-memory storage for now ---

_TASKS: list[dict] = []


def create_task(**kwargs) -> str:
    """Adds a task to the task list."""
    title = kwargs.get("title") or kwargs.get("task")
    if not title:
        raise ValueError("A task title is required (pass 'title' or 'task').")
    due_date = kwargs.get("due_date", "no due date")
    _TASKS.append({"title": title, "due_date": due_date})
    return f"Added '{title}' to your task list (due: {due_date})."


def list_tasks(**kwargs) -> str:
    """Lists everything currently on the task list."""
    if not _TASKS:
        return "Your task list is empty."
    lines = [f"- {t['title']} (due: {t['due_date']})" for t in _TASKS]
    return "Your tasks:\n" + "\n".join(lines)


# --- P4: add your real tools above this line ---


# Single source of truth: agent_loop.py dispatches through this dict,
# and derives the tool-name list it tells the LLM about from its keys.
# Adding a tool means adding one line here -- nothing else to touch.
TOOLS: dict[str, Callable[..., str]] = {
    "create_task": create_task,
    "list_tasks": list_tasks,
    # "check_calendar": check_calendar,
    # "web_search": web_search,
}
