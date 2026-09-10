from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator
class Query(BaseModel):
    interpreted_query: str = Field(
        description="The user's intended request."
    )
    memory_required: bool = Field(
        description="Whether stored memory is needed."
    )
    intent: Literal[
        "information",
        "advice",
        "motivation",
        "emotional_support",
        "companionship",
        "venting",
        "task_completion",
    ] = Field(
        description="What the user primarily wants."
    )
    tone: Literal[
        "neutral",
        "friendly",
        "casual",
        "supportive",
        "empathetic",
        "encouraging",
        "professional",
    ] = Field(
        description="How Nishi should respond."
    )
    response_type: Literal[
        "explanation",
        "guidance",
        "motivation",
        "comfort",
        "conversation",
        "direct_answer",
        "task_result",
    ] = Field(
        description="The appropriate response approach."
    )
class Decision(BaseModel):
    type: Literal["conversation", "task"] = Field(
        description="Whether the request is conversation or a task."
    )
    execution_mode: Literal["direct", "execute"] | None = Field(
        default=None,
        description=(
            "For tasks, 'direct' gives an answer and 'execute' "
            "uses an external action. None for conversations."
        )
    )
    goal: str | None = Field(
        default=None,
        description="The user's desired outcome."
    )
    objective: str | None = Field(
        default=None,
        description="The objective needed to achieve the goal."
    )
    success_condition: str | None = Field(
        default=None,
        description="What determines task success."
    )
    tool: str | None = Field(
        default=None,
        description="The initial tool to use, if needed."
    )
    tool_arguments: dict[str, Any] | None = Field(
        default=None,
        description="Arguments for the initial tool."
    )
    answer: str = Field(
        description=(
            "The final user-facing answer. "
            "Do not include internal reasoning or execution details."
        )
    )
    @model_validator(mode="after")
    def validate_decision(self): 
         if not self.answer.strip():
             raise ValueError("Answer must not be empty.")
         if self.type == "conversation":
            if self.execution_mode is not None:
                 raise ValueError(
                "Conversation must have execution_mode=None."
            )
         if self.tool is not None or self.tool_arguments is not None:
                 raise ValueError(
                "Conversation cannot have tool fields."
            )
         elif self.type == "task":
             if self.execution_mode is None:
                 raise ValueError(
                "Task must have an execution_mode."
            )
             if self.execution_mode == "direct":
                 if self.tool is not None or self.tool_arguments is not None:
                    raise ValueError(
                    "Direct tasks cannot have tool fields."
                )
         return self
class Observation(BaseModel):
    success: bool = Field(
        description="Whether the action completed successfully."
    )
    result: str | None = Field(
        default=None,
        description="The result produced by the action."
    )
    error: str | None = Field(
        default=None,
        description="The error encountered, if any."
    )