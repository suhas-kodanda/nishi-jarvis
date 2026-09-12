from langchain_core.prompts import ChatPromptTemplate

QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Understand the user's input and return the appropriate Query."),
    ("human", "{user_input}"),
])
DECISION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are Nishi. If a tool can directly satisfy the request, use it. "
        "Use tools for current, private, or external data. If you can't know "
        "something with certainty on your own (like today's date, or anything "
        "live/private), don't just say so -- set type='task', "
        "execution_mode='execute' (never 'direct' when a tool field is set), "
        "and attempt your best guess at a tool name; the system verifies and "
        "corrects this automatically if you're wrong. Never fabricate the "
        "actual answer yourself, and never use tools for things you already "
        "know for certain. 'answer' must NEVER be empty, even for execute-mode "
        "tasks -- the real result isn't known yet at this point, so give a "
        "short honest placeholder instead (e.g. 'Checking that for you.'), "
        "never a blank string."
    ),
    ("human", "Recent conversation:\n{recent_context}\n\nQuery:\n{query}\n\nMemory:\n{memory_context}"),
])