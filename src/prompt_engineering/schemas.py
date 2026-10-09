from typing import Literal
from pydantic import BaseModel, ConfigDict

Role = Literal["system", "user", "assistant"]

class PromptMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Role
    content: str

class PromptStrategy(BaseModel):
    model_config = ConfigDict(frozen=True)
    pid: int
    name: str
    messages: tuple[PromptMessage, ...]