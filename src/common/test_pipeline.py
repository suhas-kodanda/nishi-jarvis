"""
Manual test harness for the Query -> Decision -> (answer or agent loop) pipeline.

This isn't automated pass/fail testing -- the LLM's exact wording will vary
run to run. It's a fixed set of inputs chosen to exercise each branch, so
you can eyeball whether the routing and reasoning actually look right.

Run:
    python test_pipeline.py
"""

import time

from nishi_pipeline import handle_message
from schema import Query

TEST_CASES = [
    {
        "label": "Pure conversation",
        "input": "How was your day?",
        "watch_for": "type=conversation, tool=None, answer reads like a normal chat reply",
    },
    {
        "label": "Emotional / venting",
        "input": "I'm feeling really overwhelmed with everything right now.",
        "watch_for": "intent=emotional_support or venting, tone=empathetic/supportive, type=conversation",
    },
    {
        "label": "Task, no tool needed",
        "input": "What's 15% of 240?",
        "watch_for": "type=task, execution_mode=direct, the answer already has the number (36)",
    },
    {
        "label": "Task, needs a real tool",
        "input": "Add 'buy groceries' to my task list for tomorrow.",
        "watch_for": (
            "type=task, execution_mode=execute. Since create_task actually "
            "works now, this will likely resolve via the fast path (zero "
            "extra LLM calls) rather than entering the reasoning loop -- "
            "that's expected, not a bug. Force a real failure (e.g. break "
            "create_task temporarily) if you want to see the loop kick in."
        ),
    },
    {
        "label": "Should trigger a memory lookup",
        "input": "What did I say my main goal was last time we talked?",
        "watch_for": (
            "Query.memory_required=True (even though fake_memory below just "
            "returns a stub for now, until Person 2's real retrieval exists)"
        ),
    },
]


def fake_memory(query: Query) -> str:
    """Stand-in for Person 2's real retrieval logic. Swap this out once that exists."""
    return "User's name is not yet known. No prior goals recorded."


def run_case(case: dict) -> None:
    print("=" * 72)
    print(f"CASE: {case['label']}")
    print(f"INPUT: {case['input']}")
    print(f"WATCH FOR: {case['watch_for']}")
    print("-" * 72)
    start = time.time()
    final = handle_message(case["input"], fake_memory, verbose=True)
    elapsed = time.time() - start
    print(f"FINAL RESPONSE ({elapsed:.1f}s): {final}\n")


if __name__ == "__main__":
    for i, case in enumerate(TEST_CASES):
        run_case(case)
        if i < len(TEST_CASES) - 1:
            time.sleep(5)  # avoid stacking calls into the same rate-limit window