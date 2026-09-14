def ask_user(question: str) -> str:
    """Pauses and asks the user a clarifying question when you're missing
    information you genuinely need and can't infer it -- e.g. a real
    email address, or which of several tasks they meant. Returns the
    user's actual answer once they respond."""
    raise NotImplementedError(
        "ask_user's real behavior lives in agent_loop.py's reason_node, "
        "not here -- this body never actually runs. It exists only so "
        "the model has a real, typed tool to call."
    )
