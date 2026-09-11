from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryLayer(str, Enum):
    L1 = "L1"  # Personality / stable preferences
    L2 = "L2"  # Goals / plans
    L3 = "L3"  # Conversation history / summaries
    L4 = "L4"  # Verified actions / events


class MemoryKind(str, Enum):
    PERSONALITY = "personality"
    GOAL = "goal"
    HISTORY = "history"
    ACTION_EVENT = "action_event"


class MemorySource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal[
        "user_message",
        "assistant_message",
        "stt_transcript",
        "action_result",
        "manual",
        "system",
    ]

    source_id: str | None = None
    excerpt: str | None = None


class MemoryRecord(BaseModel):
    """Canonical P1 memory object."""

    model_config = ConfigDict(extra="forbid")

    id: str
    owner_id: str
    stable_key: str

    layer: MemoryLayer
    kind: MemoryKind

    content: str = Field(min_length=1)

    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    source: MemorySource | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    version: int = Field(default=1, ge=1)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    expires_at: datetime | None = None

    deleted: bool = False


class MemoryCandidate(BaseModel):
    """Candidate returned from P1 to P2."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    stable_key: str

    layer: MemoryLayer
    kind: MemoryKind

    content: str

    importance: float
    confidence: float

    score: float

    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryContext(BaseModel):
    """Final memory package returned after P2 selection."""

    model_config = ConfigDict(extra="forbid")

    memories: list[MemoryCandidate] = Field(default_factory=list)

    query: str

    generated_at: datetime = Field(default_factory=utc_now)