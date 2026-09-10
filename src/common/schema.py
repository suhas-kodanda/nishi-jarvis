from typing import Literal
from pydantic import BaseModel, Field, model_validator
class Decision(BaseModel):
    type: Literal["conversation", "task"] = Field(
        description="Whether the user's request is a conversation or a task."
    )
    execution_mode: Literal["direct", "tool", "agent"] | None = Field(
        default=None,
        description=(
            "For tasks, determines how the task should be executed: "
            "'direct' for a direct LLM response, "
            "'tool' for external tool use, or "
            "'agent' for an iterative agentic loop. "
            "Use None for conversations."
        )
    )
    goal: str | None = Field(
        default=None,
        description=(
            "The user's overall desired outcome, when needed."
        )
    )
    objective: str | None = Field(
        default=None,
        description=(
            "The specific objective required to achieve the goal, when needed."
        )
    )
    success_condition: str | None = Field(
        default=None,
        description=(
            "The condition that determines whether the task "
            "has been successfully completed."
        )
    )
    answer: str = Field(
        description=(
            "The final user-facing answer. Always provide an answer. "
            "Give only the relevant result or response. Do not include "
            "internal reasoning, planning, tool calls, agent steps, "
            "execution history, or unnecessary background unless "
            "explicitly requested by the user."
        )
    )
    @model_validator(mode="after")
    def validate_decision(self):
        if not self.answer.strip():
            raise ValueError("Answer must not be empty.")
        if self.type == "conversation" and self.execution_mode is not None:
            raise ValueError(
                "Conversation must have execution_mode=None."
            )
        if self.type == "task" and self.execution_mode is None:
            raise ValueError(
                "Task must have an execution_mode."
            )
        return self
class Observation(BaseModel):
    success: bool = Field(
        description=(
            "Whether the action or tool execution "
            "completed successfully."
        )
    )
    result: str | None = Field(
        default=None,
        description=(
            "The useful result produced by the action or tool."
        )
    )
    error: str | None = Field(
        default=None,
        description=(
            "The error encountered during execution, if any."
        )
    )