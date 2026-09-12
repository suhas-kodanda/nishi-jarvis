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
from pathlib import Path

# --- Google Calendar + Gmail: real OAuth + API integration ---
# credentials.json (downloaded from Google Cloud Console) and token.json
# (generated automatically on first successful authorization) both live
# at the repo root -- same anchoring reasoning as memory_bridge.py's DB
# path, so this works regardless of which directory you run from.
#
# SETUP REQUIRED before this works (can't be done from code):
#   1. Google Cloud Console -> enable the Google Calendar API AND the
#      Gmail API (APIs & Services > Library, search + Enable each)
#   2. Google Auth Platform -> Audience: External, add your test users
#   3. Google Auth Platform -> Data access: add the scopes listed below
#   4. Google Auth Platform -> Clients -> Create Client -> Desktop app
#   5. Download the JSON, save it as credentials.json at the repo root
#   6. NEVER commit credentials.json or token.json -- both in .gitignore
#
# One shared authorization for both APIs -- requesting both scopes
# together means one browser consent, not two separate logins. Only
# gmail.send is requested, not gmail.readonly/gmail.modify: sending is
# all that's needed right now, and it sits in a lighter Google review
# tier than reading inbox content would.
#
# First real call will open a browser for one-time authorization;
# token.json gets written automatically after that and reused (with
# automatic refresh) on every call after, no repeated browser prompts.
_GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.send",
]
_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/common/ -> src/ -> repo root
_CREDENTIALS_PATH = _REPO_ROOT / "credentials.json"
_TOKEN_PATH = _REPO_ROOT / "token.json"


def _get_google_credentials():
    """Shared OAuth handling for every Google API this file uses --
    Calendar and Gmail both call this, not separate copies of the same
    logic. Reuses/refreshes a cached token if possible, otherwise runs
    the one-time browser authorization."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), _GOOGLE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not _CREDENTIALS_PATH.exists():
                raise RuntimeError(
                    f"credentials.json not found at {_CREDENTIALS_PATH}. "
                    "Download OAuth credentials from Google Cloud Console "
                    "(Desktop app type) and save it there first."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(_CREDENTIALS_PATH), _GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
        _TOKEN_PATH.write_text(creds.to_json())

    return creds


# Hardcoded rather than auto-detected -- reliable IANA timezone
# auto-detection isn't consistent across platforms. Change this to
# match your actual timezone.
_DEFAULT_TIMEZONE = "Asia/Kolkata"


def _get_calendar_service():
    """Builds the Calendar API client from the shared credentials."""
    from googleapiclient.discovery import build

    return build("calendar", "v3", credentials=_get_google_credentials())


def _parse_datetime(text: str, default_hour: int = 9):
    """LLM-provided date/time strings won't reliably be in one exact
    format -- dateutil handles ABSOLUTE dates well ('October 11', 'Oct
    11 2026 3pm') but doesn't understand relative expressions at all
    ('next Friday', 'tomorrow') -- confirmed by testing, not assumed.
    Those are handled manually below before falling through to dateutil
    for everything else. If no year is given and the resulting date has
    already passed this year, assumes next year -- 'schedule October 11'
    almost always means the upcoming one, not one that already happened.
    """
    from datetime import datetime, timedelta
    from dateutil import parser as dateutil_parser

    now = datetime.now()
    lowered = text.lower().strip()

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    time_part = default_hour, 0
    for wd_name in [w for w in weekdays if w in lowered]:
        target_wd = weekdays.index(wd_name)
        days_ahead = (target_wd - now.weekday()) % 7
        if days_ahead == 0 or "next" in lowered:
            days_ahead = days_ahead if days_ahead > 0 else 7
        base = now + timedelta(days=days_ahead)
        # Reuse dateutil just to pull a time-of-day out of the same
        # string if one's present (e.g. "next Friday 3pm" -> 3pm)
        try:
            with_time = dateutil_parser.parse(lowered.replace(wd_name, "").replace("next", "").strip() or "9am")
            time_part = with_time.hour, with_time.minute
        except (ValueError, dateutil_parser.ParserError):
            pass
        return base.replace(hour=time_part[0], minute=time_part[1], second=0, microsecond=0)

    if "tomorrow" in lowered:
        return (now + timedelta(days=1)).replace(hour=default_hour, minute=0, second=0, microsecond=0)
    if "today" in lowered:
        return now.replace(hour=default_hour, minute=0, second=0, microsecond=0)

    parsed = dateutil_parser.parse(text, default=now.replace(hour=default_hour, minute=0, second=0, microsecond=0))
    if parsed < now and "%Y" not in text and str(now.year) not in text:
        parsed = parsed.replace(year=parsed.year + 1)
    return parsed


def create_calendar_event(title: str, start_time: str, end_time: Optional[str] = None) -> str:
    """Creates a new event on the user's Google Calendar."""
    service = _get_calendar_service()

    start_dt = _parse_datetime(start_time)
    end_dt = _parse_datetime(end_time) if end_time else start_dt.replace(hour=start_dt.hour + 1)

    event = {
        "summary": title,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": _DEFAULT_TIMEZONE},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": _DEFAULT_TIMEZONE},
    }
    created = service.events().insert(calendarId="primary", body=event).execute()
    return f"Created '{title}' on your calendar for {start_dt.strftime('%B %d, %Y at %I:%M %p')}. Link: {created.get('htmlLink')}"


def find_calendar_events(date: str, search_term: Optional[str] = None) -> str:
    """Finds events on the user's calendar for a given date."""
    from datetime import timedelta

    service = _get_calendar_service()
    day = _parse_datetime(date)
    day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    events_result = service.events().list(
        calendarId="primary",
        timeMin=day_start.isoformat() + "Z",
        timeMax=day_end.isoformat() + "Z",
        singleEvents=True,
        orderBy="startTime",
        q=search_term,
    ).execute()

    events = events_result.get("items", [])
    if not events:
        return f"No events found on {day_start.strftime('%B %d, %Y')}."
    lines = [f"- {e['summary']} at {e['start'].get('dateTime', e['start'].get('date'))}" for e in events]
    return f"Events on {day_start.strftime('%B %d, %Y')}:\n" + "\n".join(lines)

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
    """Sends an email immediately from the user's Gmail account."""
    import base64
    from email.mime.text import MIMEText
    from googleapiclient.discovery import build

    service = build("gmail", "v1", credentials=_get_google_credentials())

    message = MIMEText(body)
    message["To"] = to
    message["Subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return f"Sent email to {to} with subject '{subject}'. (id: {sent.get('id')})"


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
    "create_calendar_event": create_calendar_event,
    "find_calendar_events": find_calendar_events,
    "send_email": send_email,
    # "calculate": calculate,  -- see earlier discussion, not registered yet
    # "ask_user": ask_user,  -- see architecture note above before adding
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