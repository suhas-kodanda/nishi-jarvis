"""
Interactive chat loop -- talk to Nishi directly instead of running through
test_pipeline.py's fixed set of cases.

Run:
    python chat.py

Type 'exit' or 'quit' to end the session, or Ctrl+C.
"""
import warnings
import logging

# Existing warning suppression
warnings.filterwarnings(
    "ignore",
    message="Model .* uses fixed sampling defaults.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Direct use of automatic function calling.*",
    category=UserWarning,
)

from google.genai import models

models.Models._logged_afc_warning = True
models.AsyncModels._logged_afc_warning = True

from schema import Decision, Query

from memory_bridge import close as close_memory
from memory_bridge import get_memory_context, update_memory
from nishi_pipeline import handle_message

VERBOSE = False  # set True only when debugging -- shows Query/Decision
                  # internals per turn. Default is clean: just the final answer.

# Kept small on purpose: every line here gets sent on every single
# make_decision call for the rest of the session. More turns = better
# recall, more tokens, every time. 3 exchanges is a deliberate balance,
# not an arbitrary number -- tune it if it's not enough in practice.
MAX_RECENT_TURNS = 3
_recent_turns: list[str] = []


def main() -> None:
     print("\n  Nishi")
     print("  ─────────────────────────────────")
     print("  Your goal. Her actions.\n")

     while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            recent_context = "\n".join(_recent_turns) or "(start of conversation)"
            response = handle_message(
                user_input, get_memory_context, update_memory,
                recent_context=recent_context, verbose=VERBOSE,
            )
        except Exception as e:
            # One bad turn (a rate limit, a validation error, whatever)
            # shouldn't kill the whole session -- print it and keep going.
            print(f"[error this turn, session continues: {e}]\n")
            continue

        if isinstance(response, list):
             response = "".join(
             item.get("text", "")
              for item in response
             if isinstance(item, dict) and item.get("type") == "text"
              )

        print(f"Nishi: {response}\n")

        _recent_turns.append(f"User: {user_input}")
        _recent_turns.append(f"Nishi: {response}")
        _recent_turns[:] = _recent_turns[-(MAX_RECENT_TURNS * 2):]

     close_memory()


if __name__ == "__main__":
    main()