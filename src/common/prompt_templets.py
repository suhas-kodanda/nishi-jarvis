from langchain_core.prompts import ChatPromptTemplate

QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Understand the user's input and return the appropriate Query."),
    ("human", "{user_input}"),
])
DECISION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are Nishi. If a tool can directly satisfy the request, use it. "
        "Use tools for current, private, or external data. Never guess, invent, "
        "or use unrelated tools."
    ),
    ("human", "Recent conversation:\n{recent_context}\n\nQuery:\n{query}\n\nMemory:\n{memory_context}"),
])