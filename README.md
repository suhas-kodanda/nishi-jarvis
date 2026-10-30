# Nishi

**A modular personal AI assistant combining persistent memory, LLM-based reasoning, and tool-driven task execution.**

Nishi explores an architecture for personal AI assistants that can retain relevant context, interpret user intent, and execute tasks through external tools.

## Overview

Unlike a conventional chatbot, Nishi separates conversation processing, memory retrieval, decision-making, and task execution into modular components. This architecture provides a foundation for building an assistant that can move beyond generating responses to performing useful actions.

## Core Capabilities

- **Hierarchical Memory:** Organizes preferences, goals, and conversation history into distinct memory layers.
- **Context Retrieval:** Selects relevant memories using relevance, importance, and confidence scores.
- **Decision Pipeline:** Routes requests according to intent and execution requirements.
- **Tool-Based Execution:** Supports multi-step workflows through a reasoning and observation loop.
- **External Integrations:** Includes tools for Google services and GitHub.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Google Gemini | Language model |
| LangChain | Model and tool integration |
| LangGraph | Agent workflow orchestration |
| Pydantic | Data validation and structured schemas |
| SQLite | Persistent memory storage |

## Project Structure

```text
nishi-jarvis/
├── memory/       # Memory models, storage, and services
├── src/          # Retrieval, orchestration, and tool integrations
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- Google Gemini API key
- Additional credentials for external integrations

### Installation

```bash
git clone https://github.com/suhas-kodanda/nishi-jarvis.git
cd nishi-jarvis

python -m venv .venv
```

Activate the virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Configure the required environment variables and integration credentials before running the application.

## Current Scope

Nishi is an evolving prototype. It currently uses score-based memory retrieval rather than embedding-based semantic search. External integrations require individual configuration, and the project is not yet designed for production deployment.

## Project Goal

The long-term objective is to explore how persistent memory, structured decision-making, and tool execution can be combined into a cohesive personal AI assistant.

This returns the final memory context for P3.
