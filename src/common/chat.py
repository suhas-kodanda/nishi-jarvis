"""
Interactive chat loop -- talk to Nishi directly instead of running through
test_pipeline.py's fixed set of cases.

Run:
    python chat.py

Type 'exit' or 'quit' to end the session, or Ctrl+C.
"""

import warnings

# Harmless: gemini-3.5-flash-lite ignores the temperature parameter we set,
# and langchain_google_genai warns about it on every single call. Doesn't
# affect correctness -- just noise. Suppressed here, not by editing the
# library itself, so it stays suppressed regardless of which file actually
# triggers it (nishi_pipeline.py or agent_loop.py).
warnings.filterwarnings(
    "ignore",
    message="Model .* uses fixed sampling defaults.*",
    category=UserWarning,
)

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
    print("Nishi is ready. Type 'exit' or 'quit' to stop.")
    print(
        "Note: there's no conversation memory across turns yet (see the L4 "
        "gap discussed earlier) -- each message is handled independently, "
        "so it won't recall what you said two messages ago within this chat.\n"
    )

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

        print(f"Nishi: {response}\n")

        _recent_turns.append(f"User: {user_input}")
        _recent_turns.append(f"Nishi: {response}")
        _recent_turns[:] = _recent_turns[-(MAX_RECENT_TURNS * 2):]

    close_memory()


if __name__ == "__main__":
    main()