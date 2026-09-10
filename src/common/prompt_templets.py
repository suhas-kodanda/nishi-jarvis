from langchain_core.prompts import ChatPromptTemplate

QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Understand the user's input and return the appropriate Query."),
    ("human", "{user_input}"),
])
DECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are Nishi, a personal AI assistant."),
    ("human", "Query:\n{query}\n\nMemory:\n{memory_context}"),
])