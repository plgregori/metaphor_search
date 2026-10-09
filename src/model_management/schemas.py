from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, model_validator, field_validator
from datetime import datetime

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["ollama", "openai"]
    model_id: str               # name the provider's API uses (Ollama tag, OpenAI model name...)
    hf_id: str | None = None    # Hugging Face ID: only needed to fine-tune a local model
    fine_tuning: bool
    rag: bool
    prompt_engineering: bool

    @field_validator("hf_id", mode="before")
    @classmethod
    def _blank_hf_id_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value   # an empty form box means "not set"

    @model_validator(mode="after")
    def _hf_id_for_local_fine_tuning(self) -> Self:
        if self.provider == "ollama" and self.fine_tuning and not self.hf_id:
            raise ValueError("Hugging Face id is required for an ollama model with fine_tuning enabled")
        return self

class HistoryEntry(BaseModel):
    config: ModelConfig
    deleted_at: datetime
    removed_from_ollama: bool

class PullState(BaseModel):
    state: Literal["queued", "pulling", "done", "failed"]
    percent: float | None = None   # progress of the layer being downloaded
    detail: str | None = None