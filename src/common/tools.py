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
from dotenv import load_dotenv

load_dotenv()

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
    "https://www.googleapis.com/auth/gmail.compose",  # covers send + draft management (superset of gmail.send)
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/tasks",
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
def get_current_datetime() -> str:
    """Returns the current date and time -- Gemini can't reliably know 'now' on its own."""
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%A, %B %d, %Y, %I:%M %p")

# ============================================================
# GOOGLE TASKS TOOLS
# ============================================================

def _get_tasks_service():
    """Builds the Google Tasks API client from shared credentials."""
    from googleapiclient.discovery import build

    return build(
        "tasks",
        "v1",
        credentials=_get_google_credentials()
    )


def _get_default_tasklist_id() -> str:
    """Returns the user's default Google Tasks list ID."""
    return "@default"


def create_task(
    title: str,
    due_date: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """Creates a task in the user's Google Tasks list."""

    if not title:
        raise ValueError("A task title is required.")

    service = _get_tasks_service()

    body = {
        "title": title,
    }

    if notes:
        body["notes"] = notes

    if due_date:
        due_dt = _parse_datetime(due_date)
        body["due"] = due_dt.isoformat() + "Z"

    task = service.tasks().insert(
        tasklist=_get_default_tasklist_id(),
        body=body,
    ).execute()

    due_text = ""
    if task.get("due"):
        due_text = f" Due: {task['due'][:10]}."

    return (
        f"Created Google Task '{task.get('title')}'."
        f"{due_text}"
        f" Task ID: {task.get('id')}."
    )


def list_tasks(
    show_completed: bool = False,
) -> str:
    """Lists tasks from the user's Google Tasks list."""

    service = _get_tasks_service()

    result = service.tasks().list(
        tasklist=_get_default_tasklist_id(),
        showCompleted=show_completed,
        showHidden=False,
        maxResults=100,
    ).execute()

    tasks = result.get("items", [])

    if not tasks:
        return "No Google Tasks found."

    lines = []

    for task in tasks:
        status = (
            "✓ completed"
            if task.get("status") == "completed"
            else "pending"
        )

        due = task.get("due")
        due_text = f", due {due[:10]}" if due else ""

        lines.append(
            f"- [{task.get('id')}] "
            f"{task.get('title')} "
            f"({status}{due_text})"
        )

    return "Google Tasks:\n" + "\n".join(lines)


def get_task(task_id: str) -> str:
    """Gets a specific Google Task by ID."""

    if not task_id:
        raise ValueError("A task ID is required.")

    service = _get_tasks_service()

    task = service.tasks().get(
        tasklist=_get_default_tasklist_id(),
        task=task_id,
    ).execute()

    due = task.get("due")
    due_text = due[:10] if due else "No due date"

    return (
        f"Task: {task.get('title')}\n"
        f"ID: {task.get('id')}\n"
        f"Status: {task.get('status')}\n"
        f"Due: {due_text}\n"
        f"Notes: {task.get('notes', 'None')}"
    )


def update_task(
    task_id: str,
    title: Optional[str] = None,
    due_date: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """Updates an existing Google Task."""

    if not task_id:
        raise ValueError("A task ID is required.")

    service = _get_tasks_service()

    task = service.tasks().get(
        tasklist=_get_default_tasklist_id(),
        task=task_id,
    ).execute()

    if title is not None:
        task["title"] = title

    if due_date is not None:
        due_dt = _parse_datetime(due_date)
        task["due"] = due_dt.isoformat() + "Z"

    if notes is not None:
        task["notes"] = notes

    updated = service.tasks().update(
        tasklist=_get_default_tasklist_id(),
        task=task_id,
        body=task,
    ).execute()

    return (
        f"Updated Google Task '{updated.get('title')}'. "
        f"Task ID: {updated.get('id')}."
    )


def complete_task(task_id: str) -> str:
    """Marks a Google Task as completed."""

    if not task_id:
        raise ValueError("A task ID is required.")

    service = _get_tasks_service()

    task = service.tasks().get(
        tasklist=_get_default_tasklist_id(),
        task=task_id,
    ).execute()

    task["status"] = "completed"

    completed = service.tasks().update(
        tasklist=_get_default_tasklist_id(),
        task=task_id,
        body=task,
    ).execute()

    return (
        f"Completed Google Task '{completed.get('title')}'."
    )


def delete_task(task_id: str) -> str:
    """Deletes a Google Task."""

    if not task_id:
        raise ValueError("A task ID is required.")

    service = _get_tasks_service()

    task = service.tasks().get(
        tasklist=_get_default_tasklist_id(),
        task=task_id,
    ).execute()

    title = task.get("title", "Unknown task")

    service.tasks().delete(
        tasklist=_get_default_tasklist_id(),
        task=task_id,
    ).execute()

    return f"Deleted Google Task '{title}'."


def list_task_lists() -> str:
    """Lists the user's Google Tasks lists."""

    service = _get_tasks_service()

    result = service.tasklists().list(
        maxResults=100
    ).execute()

    task_lists = result.get("items", [])

    if not task_lists:
        return "No Google Task lists found."

    lines = [
        f"- [{task_list.get('id')}] {task_list.get('title')}"
        for task_list in task_lists
    ]

    return "Google Task lists:\n" + "\n".join(lines)


def create_task_list(title: str) -> str:
    """Creates a new Google Tasks list."""

    if not title:
        raise ValueError("A task list title is required.")

    service = _get_tasks_service()

    task_list = service.tasklists().insert(
        body={"title": title}
    ).execute()

    return (
        f"Created Google Task list '{task_list.get('title')}'. "
        f"List ID: {task_list.get('id')}."
    )


def clear_completed_tasks() -> str:
    """Clears completed tasks from the default Google Tasks list."""

    service = _get_tasks_service()

    service.tasks().clear(
        tasklist=_get_default_tasklist_id()
    ).execute()

    return "Cleared completed Google Tasks."

def update_calendar_event(event_id: str, title: Optional[str] = None, start_time: Optional[str] = None) -> str:
    """Updates an existing calendar event's title or start time."""
    service = _get_calendar_service()

    updates = {}
    if title is not None:
        updates["summary"] = title
    if start_time is not None:
        start_dt = _parse_datetime(start_time)
        updates["start"] = {"dateTime": start_dt.isoformat(), "timeZone": _DEFAULT_TIMEZONE}
    if not updates:
        return "No changes were specified -- nothing was updated."

    updated = service.events().patch(calendarId="primary", eventId=event_id, body=updates).execute()
    return f"Updated event '{updated.get('summary', event_id)}'."


def delete_calendar_event(event_id: str) -> str:
    """Deletes a calendar event."""
    service = _get_calendar_service()
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return "Deleted the event."


def _get_header(headers: list, name: str) -> str:
    """Gmail returns headers as a flat list of {name, value} dicts, not a
    dict -- this just makes lookup by name convenient."""
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return "(unknown)"


def _extract_email_body(payload: dict) -> str:
    """Gmail nests the actual body inside `parts` for multipart messages
    (most real emails: text + html versions together), or directly in
    body.data for simple ones. Recursively finds and decodes the first
    text/plain part, falling back to whatever's there if none is found."""
    import base64

    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        result = _extract_email_body(part)
        if result:
            return result

    if payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    return ""


def search_emails(query: str) -> str:
    """Searches emails using Gmail search syntax (e.g. 'from:alice@example.com', 'subject:invoice', or plain keywords)."""
    from googleapiclient.discovery import build

    service = build("gmail", "v1", credentials=_get_google_credentials())
    results = service.users().messages().list(userId="me", q=query, maxResults=5).execute()
    messages = results.get("messages", [])

    if not messages:
        return f"No emails found matching '{query}'."

    lines = []
    for msg in messages:
        detail = service.users().messages().get(
            userId="me", id=msg["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()
        headers = detail.get("payload", {}).get("headers", [])
        lines.append(
            f"- [{msg['id']}] From: {_get_header(headers, 'From')} | "
            f"Subject: {_get_header(headers, 'Subject')} | {_get_header(headers, 'Date')}"
        )

    return f"Found {len(messages)} email(s) matching '{query}':\n" + "\n".join(lines)


def read_email(email_id: str) -> str:
    """Reads the full content of a specific email (use the id shown by search_emails)."""
    from googleapiclient.discovery import build

    service = build("gmail", "v1", credentials=_get_google_credentials())
    msg = service.users().messages().get(userId="me", id=email_id, format="full").execute()

    headers = msg.get("payload", {}).get("headers", [])
    body = _extract_email_body(msg.get("payload", {})).strip()
    # Capped, not sent to the LLM unbounded -- a long email body feeding
    # back into the reasoning loop is exactly the kind of per-call token
    # cost that's been worth watching throughout this project.
    if len(body) > 1500:
        body = body[:1500] + "... [truncated]"

    return (
        f"From: {_get_header(headers, 'From')}\n"
        f"Subject: {_get_header(headers, 'Subject')}\n"
        f"Date: {_get_header(headers, 'Date')}\n\n{body}"
    )


def draft_email(to: str, subject: str, body: str) -> str:
    """Creates a draft email without sending it."""
    import base64
    from email.mime.text import MIMEText
    from googleapiclient.discovery import build

    service = build("gmail", "v1", credentials=_get_google_credentials())

    message = MIMEText(body)
    message["To"] = to
    message["Subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    draft = service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
    return f"Created a draft to {to} with subject '{subject}'."


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

# ============================================================
# GOOGLE DRIVE TOOLS
# ============================================================

def _get_drive_service():
    """Returns an authenticated Google Drive API service."""
    from googleapiclient.discovery import build

    return build(
        "drive",
        "v3",
        credentials=_get_google_credentials()
    )


def search_drive(query: str) -> str:
    """Searches Google Drive for files by name."""
    if not query:
        raise ValueError("Search query cannot be empty.")

    service = _get_drive_service()

    safe_query = query.replace("\\", "\\\\").replace("'", "\\'")

    results = service.files().list(
        q=f"name contains '{safe_query}' and trashed = false",
        pageSize=20,
        orderBy="modifiedTime desc",
        fields=(
            "files(id,name,mimeType,size,modifiedTime,"
            "createdTime,webViewLink)"
        ),
    ).execute()

    files = results.get("files", [])

    if not files:
        return f"No Drive files found matching '{query}'."

    output = [f"Found {len(files)} file(s):"]

    for file in files:
        output.append(
            f"\nName: {file.get('name')}"
            f"\nID: {file.get('id')}"
            f"\nType: {file.get('mimeType')}"
            f"\nModified: {file.get('modifiedTime')}"
            f"\nLink: {file.get('webViewLink', 'N/A')}"
        )

    return "\n".join(output)


def list_drive_files(limit: int = 20) -> str:
    """Lists recent non-trashed files in Google Drive."""
    service = _get_drive_service()

    limit = max(1, min(int(limit), 100))

    results = service.files().list(
        q="trashed = false",
        pageSize=limit,
        orderBy="modifiedTime desc",
        fields=(
            "files(id,name,mimeType,size,modifiedTime,"
            "createdTime,webViewLink)"
        ),
    ).execute()

    files = results.get("files", [])

    if not files:
        return "No files found in Google Drive."

    output = [f"Found {len(files)} file(s):"]

    for file in files:
        output.append(
            f"\nName: {file.get('name')}"
            f"\nID: {file.get('id')}"
            f"\nType: {file.get('mimeType')}"
            f"\nModified: {file.get('modifiedTime')}"
            f"\nLink: {file.get('webViewLink', 'N/A')}"
        )

    return "\n".join(output)


def get_drive_file(file_id: str) -> str:
    """Gets metadata for a Google Drive file."""
    if not file_id:
        raise ValueError("file_id cannot be empty.")

    service = _get_drive_service()

    file = service.files().get(
        fileId=file_id,
        fields=(
            "id,name,mimeType,size,description,"
            "createdTime,modifiedTime,webViewLink,"
            "parents,owners"
        ),
    ).execute()

    owners = file.get("owners", [])
    owner_names = ", ".join(
        owner.get("displayName", "Unknown")
        for owner in owners
    )

    return (
        f"Name: {file.get('name')}\n"
        f"ID: {file.get('id')}\n"
        f"Type: {file.get('mimeType')}\n"
        f"Size: {file.get('size', 'N/A')}\n"
        f"Description: {file.get('description', 'N/A')}\n"
        f"Created: {file.get('createdTime')}\n"
        f"Modified: {file.get('modifiedTime')}\n"
        f"Owners: {owner_names or 'N/A'}\n"
        f"Link: {file.get('webViewLink', 'N/A')}"
    )


def read_drive_file(file_id: str) -> str:
    """
    Reads the contents of a Google Drive file.

    Supports:
    - Google Docs
    - Google Sheets
    - Google Slides
    - Plain text files
    - Other downloadable text files
    """
    if not file_id:
        raise ValueError("file_id cannot be empty.")

    service = _get_drive_service()

    file = service.files().get(
        fileId=file_id,
        fields="id,name,mimeType,size,webViewLink",
    ).execute()

    name = file.get("name", "Unknown")
    mime_type = file.get("mimeType", "")

    # Google Docs
    if mime_type == "application/vnd.google-apps.document":
        content = service.files().export(
            fileId=file_id,
            mimeType="text/plain",
        ).execute()

        return (
            f"File: {name}\n"
            f"Type: Google Doc\n\n"
            f"{content.decode('utf-8', errors='replace')}"
        )

    # Google Sheets
    if mime_type == "application/vnd.google-apps.spreadsheet":
        content = service.files().export(
            fileId=file_id,
            mimeType="text/csv",
        ).execute()

        return (
            f"File: {name}\n"
            f"Type: Google Sheet\n\n"
            f"{content.decode('utf-8', errors='replace')}"
        )

    # Google Slides
    if mime_type == "application/vnd.google-apps.presentation":
        content = service.files().export(
            fileId=file_id,
            mimeType="text/plain",
        ).execute()

        return (
            f"File: {name}\n"
            f"Type: Google Slides\n\n"
            f"{content.decode('utf-8', errors='replace')}"
        )

    # Normal downloadable files
    try:
        from googleapiclient.http import MediaIoBaseDownload
        import io

        request = service.files().get_media(fileId=file_id)

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        content = buffer.getvalue()

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return (
                f"File '{name}' is not a text-readable file.\n"
                f"Type: {mime_type}\n"
                f"Link: {file.get('webViewLink', 'N/A')}"
            )

        return (
            f"File: {name}\n"
            f"Type: {mime_type}\n\n"
            f"{text}"
        )

    except Exception as e:
        return f"Could not read '{name}': {e}"


def create_drive_file(
    name: str,
    content: str,
    mime_type: str = "text/plain",
) -> str:
    """Creates a new file in Google Drive."""
    if not name:
        raise ValueError("File name cannot be empty.")

    service = _get_drive_service()

    from googleapiclient.http import MediaIoBaseUpload
    import io

    file_metadata = {
        "name": name,
    }

    media = MediaIoBaseUpload(
        io.BytesIO(content.encode("utf-8")),
        mimetype=mime_type,
        resumable=False,
    )

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id,name,mimeType,webViewLink,createdTime",
    ).execute()

    return (
        f"File created successfully.\n"
        f"Name: {file.get('name')}\n"
        f"ID: {file.get('id')}\n"
        f"Type: {file.get('mimeType')}\n"
        f"Link: {file.get('webViewLink', 'N/A')}"
    )


def update_drive_file(
    file_id: str,
    content: str,
) -> str:
    """Replaces the contents of an existing Drive file."""
    if not file_id:
        raise ValueError("file_id cannot be empty.")

    service = _get_drive_service()

    from googleapiclient.http import MediaIoBaseUpload
    import io

    # Get existing file metadata
    existing_file = service.files().get(
        fileId=file_id,
        fields="id,name,mimeType,webViewLink",
    ).execute()

    mime_type = existing_file.get("mimeType", "text/plain")

    # Google Workspace files cannot be updated using normal
    # media upload. They need the appropriate Google API.
    if mime_type.startswith("application/vnd.google-apps."):
        return (
            f"'{existing_file.get('name')}' is a Google Workspace file "
            f"({mime_type}). Direct content replacement is not supported "
            f"by this generic update tool."
        )

    media = MediaIoBaseUpload(
        io.BytesIO(content.encode("utf-8")),
        mimetype=mime_type,
        resumable=False,
    )

    file = service.files().update(
        fileId=file_id,
        media_body=media,
        fields="id,name,mimeType,modifiedTime,webViewLink",
    ).execute()

    return (
        f"File updated successfully.\n"
        f"Name: {file.get('name')}\n"
        f"ID: {file.get('id')}\n"
        f"Modified: {file.get('modifiedTime')}\n"
        f"Link: {file.get('webViewLink', 'N/A')}"
    )


def delete_drive_file(file_id: str) -> str:
    """Moves a Google Drive file to the trash."""
    if not file_id:
        raise ValueError("file_id cannot be empty.")

    service = _get_drive_service()

    file = service.files().get(
        fileId=file_id,
        fields="id,name",
    ).execute()

    service.files().update(
        fileId=file_id,
        body={"trashed": True},
    ).execute()

    return (
        f"File moved to trash successfully.\n"
        f"Name: {file.get('name')}\n"
        f"ID: {file.get('id')}"
    )


def restore_drive_file(file_id: str) -> str:
    """Restores a trashed Google Drive file."""
    if not file_id:
        raise ValueError("file_id cannot be empty.")

    service = _get_drive_service()

    file = service.files().get(
        fileId=file_id,
        fields="id,name,trashed",
    ).execute()

    if not file.get("trashed"):
        return f"File '{file.get('name')}' is not in the trash."

    service.files().update(
        fileId=file_id,
        body={"trashed": False},
    ).execute()

    return (
        f"File restored successfully.\n"
        f"Name: {file.get('name')}\n"
        f"ID: {file.get('id')}"
    )


def create_drive_folder(name: str) -> str:
    """Creates a new folder in Google Drive."""
    if not name:
        raise ValueError("Folder name cannot be empty.")

    service = _get_drive_service()

    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }

    folder = service.files().create(
        body=file_metadata,
        fields="id,name,mimeType,webViewLink,createdTime",
    ).execute()

    return (
        f"Folder created successfully.\n"
        f"Name: {folder.get('name')}\n"
        f"ID: {folder.get('id')}\n"
        f"Link: {folder.get('webViewLink', 'N/A')}"
    )


def move_drive_file(
    file_id: str,
    folder_id: str,
) -> str:
    """Moves a Drive file into a specified folder."""
    if not file_id:
        raise ValueError("file_id cannot be empty.")

    if not folder_id:
        raise ValueError("folder_id cannot be empty.")

    service = _get_drive_service()

    file = service.files().get(
        fileId=file_id,
        fields="id,name,parents",
    ).execute()

    old_parents = ",".join(file.get("parents", []))

    service.files().update(
        fileId=file_id,
        addParents=folder_id,
        removeParents=old_parents,
        fields="id,name,parents",
    ).execute()

    return (
        f"File moved successfully.\n"
        f"Name: {file.get('name')}\n"
        f"ID: {file.get('id')}\n"
        f"Folder ID: {folder_id}"
    )


def list_drive_folder(folder_id: str, limit: int = 20) -> str:
    """Lists files inside a specific Google Drive folder."""
    if not folder_id:
        raise ValueError("folder_id cannot be empty.")

    service = _get_drive_service()

    limit = max(1, min(int(limit), 100))

    safe_folder_id = folder_id.replace("\\", "\\\\").replace("'", "\\'")

    results = service.files().list(
        q=(
            f"'{safe_folder_id}' in parents "
            f"and trashed = false"
        ),
        pageSize=limit,
        orderBy="name",
        fields=(
            "files(id,name,mimeType,size,"
            "modifiedTime,webViewLink)"
        ),
    ).execute()

    files = results.get("files", [])

    if not files:
        return "No files found in this folder."

    output = [
        f"Found {len(files)} file(s) in folder:"
    ]

    for file in files:
        output.append(
            f"\nName: {file.get('name')}"
            f"\nID: {file.get('id')}"
            f"\nType: {file.get('mimeType')}"
            f"\nLink: {file.get('webViewLink', 'N/A')}"
        )

    return "\n".join(output)

# ============================================================
# GITHUB TOOLS
# ============================================================

import os
import base64
import requests


_GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
_GITHUB_API = "https://api.github.com"


def _get_github_headers() -> dict:
    """Returns authenticated GitHub API headers."""

    if not _GITHUB_TOKEN:
        raise RuntimeError(
            "GITHUB_TOKEN is not configured."
        )

    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {_GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _github_request(
    method: str,
    endpoint: str,
    **kwargs,
):
    """Makes an authenticated GitHub API request."""

    response = requests.request(
        method,
        f"{_GITHUB_API}{endpoint}",
        headers=_get_github_headers(),
        timeout=20,
        **kwargs,
    )

    if not response.ok:
        raise RuntimeError(
            f"GitHub API error {response.status_code}: "
            f"{response.text[:500]}"
        )

    if response.status_code == 204:
        return None

    return response.json()


def github_list_repositories() -> str:
    """Lists repositories accessible to the user."""

    repos = _github_request(
        "GET",
        "/user/repos",
        params={
            "per_page": 100,
            "sort": "updated",
        },
    )

    if not repos:
        return "No GitHub repositories found."

    lines = []

    for repo in repos:
        visibility = (
            "private"
            if repo.get("private")
            else "public"
        )

        lines.append(
            f"- {repo.get('full_name')} "
            f"({visibility})"
        )

    return "GitHub repositories:\n" + "\n".join(lines)


def github_get_repository(
    owner: str,
    repo: str,
) -> str:
    """Gets information about a GitHub repository."""

    if not owner:
        raise ValueError("Repository owner is required.")

    if not repo:
        raise ValueError("Repository name is required.")

    data = _github_request(
        "GET",
        f"/repos/{owner}/{repo}",
    )

    return (
        f"Repository: {data.get('full_name')}\n"
        f"Description: "
        f"{data.get('description') or 'None'}\n"
        f"Default branch: "
        f"{data.get('default_branch')}\n"
        f"Language: "
        f"{data.get('language') or 'Unknown'}\n"
        f"Stars: {data.get('stargazers_count')}\n"
        f"Open issues: "
        f"{data.get('open_issues_count')}\n"
        f"URL: {data.get('html_url')}"
    )


def github_list_issues(
    owner: str,
    repo: str,
    state: str = "open",
) -> str:
    """Lists issues in a GitHub repository."""

    if state not in {"open", "closed", "all"}:
        raise ValueError(
            "State must be open, closed, or all."
        )

    issues = _github_request(
        "GET",
        f"/repos/{owner}/{repo}/issues",
        params={
            "state": state,
            "per_page": 50,
        },
    )

    # GitHub's issues endpoint can also return pull requests.
    issues = [
        issue
        for issue in issues
        if "pull_request" not in issue
    ]

    if not issues:
        return "No GitHub issues found."

    lines = []

    for issue in issues:
        labels = ", ".join(
            label.get("name", "")
            for label in issue.get("labels", [])
        )

        label_text = (
            f" | labels: {labels}"
            if labels
            else ""
        )

        lines.append(
            f"- #{issue.get('number')} "
            f"{issue.get('title')} "
            f"[{issue.get('state')}]"
            f"{label_text}"
        )

    return (
        f"GitHub issues for {owner}/{repo}:\n"
        + "\n".join(lines)
    )


def github_get_issue(
    owner: str,
    repo: str,
    issue_number: int,
) -> str:
    """Gets detailed information about a GitHub issue."""

    if issue_number <= 0:
        raise ValueError(
            "Issue number must be positive."
        )

    issue = _github_request(
        "GET",
        f"/repos/{owner}/{repo}/issues/{issue_number}",
    )

    labels = ", ".join(
        label.get("name", "")
        for label in issue.get("labels", [])
    )

    return (
        f"Issue #{issue.get('number')}: "
        f"{issue.get('title')}\n"
        f"State: {issue.get('state')}\n"
        f"Author: "
        f"{issue.get('user', {}).get('login')}\n"
        f"Labels: {labels or 'None'}\n"
        f"Description:\n"
        f"{issue.get('body') or 'No description'}\n"
        f"URL: {issue.get('html_url')}"
    )


def github_create_issue(
    owner: str,
    repo: str,
    title: str,
    body: str = "",
) -> str:
    """Creates a new GitHub issue."""

    if not title:
        raise ValueError(
            "Issue title is required."
        )

    issue = _github_request(
        "POST",
        f"/repos/{owner}/{repo}/issues",
        json={
            "title": title,
            "body": body,
        },
    )

    return (
        f"Created GitHub issue "
        f"#{issue.get('number')}: "
        f"{issue.get('title')}. "
        f"URL: {issue.get('html_url')}"
    )


def github_list_pull_requests(
    owner: str,
    repo: str,
    state: str = "open",
) -> str:
    """Lists pull requests in a GitHub repository."""

    if state not in {"open", "closed", "all"}:
        raise ValueError(
            "State must be open, closed, or all."
        )

    pulls = _github_request(
        "GET",
        f"/repos/{owner}/{repo}/pulls",
        params={
            "state": state,
            "per_page": 50,
        },
    )

    if not pulls:
        return "No GitHub pull requests found."

    lines = []

    for pull in pulls:
        lines.append(
            f"- #{pull.get('number')} "
            f"{pull.get('title')} "
            f"[{pull.get('state')}] "
            f"{pull.get('user', {}).get('login')}"
        )

    return (
        f"Pull requests for {owner}/{repo}:\n"
        + "\n".join(lines)
    )


def github_get_file(
    owner: str,
    repo: str,
    path: str,
    branch: Optional[str] = None,
) -> str:
    """Reads a file from a GitHub repository."""

    if not path:
        raise ValueError("File path is required.")

    params = {}

    if branch:
        params["ref"] = branch

    data = _github_request(
        "GET",
        f"/repos/{owner}/{repo}/contents/{path}",
        params=params,
    )

    if isinstance(data, list):
        return (
            "The requested path is a directory, "
            "not a file."
        )

    if data.get("type") != "file":
        return (
            f"GitHub returned a non-file resource "
            f"of type '{data.get('type')}'."
        )

    encoded = data.get("content", "")

    try:
        content = base64.b64decode(
            encoded
        ).decode("utf-8")
    except Exception:
        content = "[Unable to decode file as UTF-8]"

    return (
        f"File: {data.get('path')}\n"
        f"SHA: {data.get('sha')}\n"
        f"Content:\n{content}"
    )


def github_list_commits(
    owner: str,
    repo: str,
    branch: Optional[str] = None,
    limit: int = 20,
) -> str:
    """Lists recent commits and their authors."""

    if limit < 1 or limit > 100:
        raise ValueError(
            "Limit must be between 1 and 100."
        )

    params = {
        "per_page": limit,
    }

    if branch:
        params["sha"] = branch

    commits = _github_request(
        "GET",
        f"/repos/{owner}/{repo}/commits",
        params=params,
    )

    if not commits:
        return "No GitHub commits found."

    lines = []

    for commit in commits:
        commit_data = commit.get("commit", {})
        author = commit_data.get("author", {})

        author_name = (
            author.get("name")
            or commit.get("author", {}).get("login")
            or "Unknown"
        )

        date = author.get("date", "")

        if date:
            date = date.replace("T", " ")[:19]

        message = (
            commit_data.get("message", "")
            .split("\n")[0]
        )

        lines.append(
            f"- {commit.get('sha', '')[:7]} | "
            f"{message} | "
            f"by {author_name} | "
            f"{date}"
        )

    return (
        f"Recent commits for {owner}/{repo}:\n"
        + "\n".join(lines)
    )


def github_get_commit(
    owner: str,
    repo: str,
    sha: str,
) -> str:
    """Gets detailed information about a GitHub commit."""

    if not sha:
        raise ValueError("Commit SHA is required.")

    commit = _github_request(
        "GET",
        f"/repos/{owner}/{repo}/commits/{sha}",
    )

    commit_data = commit.get("commit", {})
    author = commit_data.get("author", {})

    lines = [
        f"Commit: {commit.get('sha')}",
        f"Message: {commit_data.get('message', '')}",
        f"Author: {author.get('name', 'Unknown')}",
        f"Date: {author.get('date', 'Unknown')}",
        "",
        "Changed files:",
    ]

    for file in commit.get("files", []):
        lines.append(
            f"- {file.get('filename')} "
            f"[{file.get('status')}] "
            f"+{file.get('additions', 0)} "
            f"-{file.get('deletions', 0)}"
        )

    return "\n".join(lines)
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
    "get_current_datetime": get_current_datetime,
    # Tasks
    "create_task": create_task,
    "list_tasks": list_tasks,
    "update_task": update_task,
    "complete_task": complete_task,
    "delete_task": delete_task,
    "get_task": get_task,
    "list_task_lists": list_task_lists,
    "create_task_list": create_task_list,
    "clear_completed_tasks": clear_completed_tasks,
    # Calendar
    "create_calendar_event": create_calendar_event,
    "find_calendar_events": find_calendar_events,
    "update_calendar_event": update_calendar_event,
    "delete_calendar_event": delete_calendar_event,
    # Gmail
    "send_email": send_email,
    "draft_email": draft_email,
    "search_emails": search_emails,
    "read_email": read_email,
    # Google Drive
    "search_drive": search_drive,
    "list_drive_files": list_drive_files,
    "get_drive_file": get_drive_file,
    "read_drive_file": read_drive_file,
    "create_drive_file": create_drive_file,
    "update_drive_file": update_drive_file,
    "delete_drive_file": delete_drive_file,
    "restore_drive_file": restore_drive_file,
    "create_drive_folder": create_drive_folder,
    "move_drive_file": move_drive_file,
    "list_drive_folder": list_drive_folder,
    # github
    "github_list_repositories": github_list_repositories,
    "github_get_repository": github_get_repository,
    "github_list_issues": github_list_issues,
    "github_get_issue": github_get_issue,
    "github_create_issue": github_create_issue,
    "github_list_pull_requests": github_list_pull_requests,
    "github_get_file": github_get_file,
    "github_list_commits": github_list_commits,
    "github_get_commit": github_get_commit,
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