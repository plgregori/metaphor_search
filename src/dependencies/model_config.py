import asyncio
import json
from json import JSONDecodeError
import logging
import os
from pathlib import Path
from typing import Literal, Self
from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, model_validator

logger = logging.getLogger(__name__)

class ConfigError(ValueError):
    """Models configuration file missing, unreadable or invalid."""

class DuplicateModelError(ValueError):
    """The alias is already registered."""

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["ollama", "openai"]
    model_id: str               # name the provider's API uses (Ollama tag, OpenAI model name...)
    hf_id: str | None = None    # Hugging Face ID: only needed to fine-tune a local model
    fine_tuning: bool
    rag: bool
    prompt_engineering: bool

    @model_validator(mode="after")
    def _hf_id_for_local_fine_tuning(self) -> Self:
        if self.provider == "ollama" and self.fine_tuning and not self.hf_id:
            raise ValueError("Hugging Face id is required for an ollama model with fine_tuning enabled")
        return self

class PullState(BaseModel):
    state: Literal["pulling", "done", "failed"]
    percent: float | None = None   # progress of the layer being downloaded
    detail: str | None = None

_MODELS_ADAPTER = TypeAdapter(dict[str, ModelConfig])

class ModelManager():
    model_config_dict: dict[str, ModelConfig]
    model_config_path: str
    ollama_url: str
    ollama_client: AsyncClient

    def __init__(self, model_config_dict: dict[str, ModelConfig], model_config_path: str, ollama_url: str) -> None:
        self.model_config_dict = model_config_dict
        self.model_config_path = model_config_path
        self.ollama_url = ollama_url
        self.pulls: dict[str, PullState] = {}
        self._pull_tasks: dict[str, "asyncio.Task[None]"] = {}
        self.ollama_client = AsyncClient(host=self.ollama_url)

    @classmethod
    def load_from_json(cls, model_config_path: str, ollama_url: str) -> "ModelManager":
        path = Path(model_config_path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise ConfigError(f"Model configuration file not found: {path}") from e
        except OSError as e:
            raise ConfigError(f"Model configuration file could not be read ({path}): {e}") from e
        try:
            raw_data = json.loads(text)
        except JSONDecodeError as e:
            raise ConfigError(f"{path} is not valid JSON: {e}") from e
        try:
            models = _MODELS_ADAPTER.validate_python(raw_data)
        except ValidationError as e:
            raise ConfigError(f"{path} has an invalid model configuration:\n{e}") from e

        return cls(models, model_config_path, ollama_url)

    def save(self) -> None:
        path = Path(self.model_config_path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(_MODELS_ADAPTER.dump_json(self.model_config_dict, indent=4))
        os.replace(tmp, path)  

    async def list_ollama_models(self) -> set[str]:
        try:
            response = await asyncio.wait_for(self.ollama_client.list(), timeout=5)
        except asyncio.TimeoutError as e:
            raise ConnectionError(f"Ollama did not answer within 5 seconds at {self.ollama_url}.") from e
        except ResponseError as e:
            raise ConnectionError(f"Ollama answered with an error: {e}") from e
        return {m.model for m in response.models if m.model}

    @staticmethod
    def is_pulled(tag: str, available: set[str]) -> bool:
        # Ollama lists untagged models as "<name>:latest"
        return tag in available or (":" not in tag and f"{tag}:latest" in available)

    async def _pull(self, model_name: str) -> None:
        tag = self.model_config_dict[model_name].model_id
        try:
            async for p in await self.ollama_client.pull(tag, stream=True):
                pct = 100 * p.completed / p.total if p.total and p.completed is not None else None
                self.pulls[model_name] = PullState(state="pulling", percent=pct, detail=p.status)
            self.pulls[model_name] = PullState(state="done", percent=100)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Pull of %s failed", tag)
            self.pulls[model_name] = PullState(state="failed", detail=str(e))

    def start_pull(self, model_name: str) -> None:
        task = self._pull_tasks.get(model_name)
        if task is not None and not task.done():
            return                                          # already downloading
        self.pulls[model_name] = PullState(state="pulling")
        self._pull_tasks[model_name] = asyncio.create_task(self._pull(model_name))


    async def add_model(self, model_name: str, model_config: ModelConfig) -> bool:
        """Registers the model in memory and in models.json. Returns True if a download was started."""
        if model_name in self.model_config_dict:
            raise DuplicateModelError(f"Model {model_name} already loaded!")

        needs_pull = False
        if model_config.provider == "ollama":
            # Everything that can fail is checked BEFORE changing anything.
            # If Ollama is down, this raises ConnectionError and nothing is written.
            available = await self.list_ollama_models()
            needs_pull = not self.is_pulled(model_config.model_id, available)

        if model_name in self.model_config_dict:        # re-check: another request may have added it during the await
            raise DuplicateModelError(f"Model {model_name} already loaded!")

        self.model_config_dict[model_name] = model_config
        try:
            self.save()
        except OSError as e:
            del self.model_config_dict[model_name]          # keep memory and file consistent
            raise ConfigError(f"Could not write {self.model_config_path}: {e}") from e

        if needs_pull:
            self.start_pull(model_name)
        return needs_pull

    async def availability(self) -> dict[str, bool | None]:
        """True/False for ollama models; None for other providers or when Ollama is unreachable."""
        try:
            available: set[str] | None = await self.list_ollama_models()
        except ConnectionError:
            available = None
        result: dict[str, bool | None] = {}
        for alias, config in self.model_config_dict.items():
            if config.provider != "ollama" or available is None:
                result[alias] = None
            else:
                result[alias] = self.is_pulled(config.model_id, available)
        return result

    async def close(self) -> None:
        """Cancels downloads still running when the app shuts down."""
        for task in self._pull_tasks.values():
            task.cancel()
        await asyncio.gather(*self._pull_tasks.values(), return_exceptions=True)
        await self.ollama_client.close()