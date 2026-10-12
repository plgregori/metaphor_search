from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

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

class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[str] = Field(default_factory=list,
        description="Model aliases from models.json. Empty = every model with prompt_engineering enabled.")
    strategies: list[str] = Field(default_factory=list,
        description="Strategy names from the prompts file (see GET /prompt-engineering/strategies). Empty = all strategies.")
    repeats: int = Field(2, ge=1, le=20,
        description="Runs per (model, strategy, text). The notebook uses 2; the paper reports 5.")
    max_texts: int | None = Field(None, ge=1,
        description="Use only the first N dataset texts (for quick tests). Empty = all texts.")
    temperature: float | None = Field(0.0, ge=0, le=2,
        description="null = the provider's default. Use null for models that reject a custom temperature (e.g. o3, o4-mini).")
    openai_concurrency: int = Field(4, ge=1, le=16,
        description="Parallel requests for OpenAI models. Ollama models always run one at a time.")
    max_consecutive_errors: int = Field(5, ge=1,
        description="A model is abandoned after this many failed calls in a row (e.g. a wrong parameter or a dead server).")

class ModelProgress(BaseModel):
    total: int
    done: int = 0
    errors: int = 0
    state: Literal["pending", "running", "done", "aborted"] = "pending"
    detail: str | None = None

class RunStatus(BaseModel):
    run_id: str
    state: Literal["running", "completed", "failed", "cancelled", "interrupted"]
    created_at: datetime
    finished_at: datetime | None = None
    request: RunRequest                       # as resolved: the models and strategies actually used
    n_texts: int
    total_calls: int
    completed_calls: int = 0
    errors: int = 0
    models: dict[str, ModelProgress]
    folder: str
    message: str | None = None

class StrategySummary(BaseModel):
    name: str
    n_messages: int
    pid: int