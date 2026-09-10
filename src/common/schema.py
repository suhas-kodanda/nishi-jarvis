from pydantic import BaseModel
from typing import Literal


class Decision(BaseModel):
    type: Literal["conversation", "task"]
    goal: str | None = None
    objective: str | None = None
    success_condition: str | None = None
    requires_agent: bool


class Observation(BaseModel):
    success: bool
    result: str | None = None
    error: str | None = None