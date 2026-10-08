import asyncio
import json
from json import JSONDecodeError
import logging
import os
from pathlib import Path
from typing import Literal, Self
from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, model_validator, field_validator
from datetime import datetime, timezone

logger = logging.getLogger("uvicorn.error")

class ConfigError(ValueError):
    """Models configuration file missing, unreadable or invalid."""

class DuplicateModelError(ValueError):
    """The alias is already registered."""

class UnknownModelError(ValueError):
       """The alias is not registered."""

class DeleteBlockedError(ValueError):
    """The model cannot be deleted right now."""

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

_MODELS_ADAPTER = TypeAdapter(dict[str, ModelConfig])
_HISTORY_ADAPTER = TypeAdapter(dict[str, HistoryEntry])

class ModelManager():
    model_config_dict: dict[str, ModelConfig]
    model_config_path: str
    ollama_url: str
    ollama_client: AsyncClient

    def __init__(self, model_config_dict: dict[str, ModelConfig], model_config_path: str, ollama_url: str,
                 model_history_path: str, model_history: dict[str, HistoryEntry]) -> None:
        self.model_config_dict = model_config_dict
        self.model_config_path = model_config_path
        self.model_history_path = model_history_path
        self.model_history = model_history
        self.ollama_url = ollama_url
        self.pulls: dict[str, PullState] = {}
        self._pull_tasks: dict[str, "asyncio.Task[None]"] = {}
        self._startup_task: "asyncio.Task[None] | None" = None
        self.ollama_client = AsyncClient(host=self.ollama_url)

    @classmethod
    def load_from_json(cls, model_config_path: str, ollama_url: str, model_history_path: str) -> "ModelManager":
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

        return cls(models, model_config_path, ollama_url, model_history_path, cls._load_history(model_history_path))

    @staticmethod
    def _load_history(model_history_path: str) -> dict[str, HistoryEntry]:
        path = Path(model_history_path)
        if not path.exists():
            return {}                       # the file is created the first time a model is deleted
        try:
            return _HISTORY_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as e:
            raise ConfigError(f"Model history file could not be read ({path}): {e}") from e

    def save_history(self) -> None:
        path = Path(self.model_history_path)
        path.parent.mkdir(parents=True, exist_ok=True)      # creates data/ if needed
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(_HISTORY_ADAPTER.dump_json(self.model_history, indent=4))
        os.replace(tmp, path)

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
        current = self.pulls.get(model_name)
        if current is not None and current.state in ("queued", "pulling"):
            return                                          # already downloading, or waiting its turn at startup
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

    def edit_model(self, model_name: str, changes: dict[str, bool | str | None]) -> ModelConfig:
           """Changes fine_tuning / rag / prompt_engineering / hf_id of a registered model, in memory and in models.json.
           Raises UnknownModelError, pydantic.ValidationError (the result would be invalid) or ConfigError."""
           current = self.model_config_dict.get(model_name)
           if current is None:
               raise UnknownModelError(f"Unknown model {model_name}.")

           # A new ModelConfig, not model_copy(update=...): that would skip the validators
           updated = ModelConfig(**{**current.model_dump(), **changes})  # pyright: ignore[reportUnknownArgumentType]

           self.model_config_dict[model_name] = updated
           try:
               self.save()
           except OSError as e:
               self.model_config_dict[model_name] = current        # keep memory and file consistent
               raise ConfigError(f"Could not write {self.model_config_path}: {e}") from e
           return updated

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


    async def _pull_in_sequence(self, aliases: list[str]) -> None:
        failed: list[str] = []
        for position, alias in enumerate(aliases, start=1):
            tag = self.model_config_dict[alias].model_id
            logger.info("[%d/%d] Downloading %s (%s)...", position, len(aliases), alias, tag)
            self.pulls[alias] = PullState(state="pulling")
            await self._pull(alias)                          # never raises, except when cancelled
            state = self.pulls[alias]
            if state.state == "done":
                logger.info("[%d/%d] Downloaded %s (%s).", position, len(aliases), alias, tag)
            else:
                failed.append(alias)
                logger.error("[%d/%d] Download of %s (%s) failed: %s", position, len(aliases), alias, tag, state.detail)
        if failed:
            logger.warning("Startup downloads finished, %d failed: %s. Retry with POST /models/{alias}/pull.",
                           len(failed), ", ".join(failed))
        else:
            logger.info("Startup downloads finished.")

    async def pull_missing(self) -> list[str]:
        """Downloads, one after the other, every ollama model of models.json that Ollama does not have yet.
        Returns the aliases to download; the downloads themselves run in a background task."""
        ollama_models = {alias: config for alias, config in self.model_config_dict.items()
                         if config.provider == "ollama"}
        try:
            available = await self.list_ollama_models()
        except ConnectionError as e:
            logger.warning("Startup model check skipped, Ollama is not reachable: %s", e)
            return []

        missing = [alias for alias, config in ollama_models.items()
                   if not self.is_pulled(config.model_id, available)]
        if not missing:
            logger.info("Startup model check: all %d ollama model(s) are already available.", len(ollama_models))
            return []

        logger.info("Startup model check: %d of %d ollama model(s) missing: %s", len(missing), len(ollama_models),
                    ", ".join(f"{alias} ({ollama_models[alias].model_id})" for alias in missing))
        for alias in missing:
            self.pulls[alias] = PullState(state="queued")
        self._startup_task = asyncio.create_task(self._pull_in_sequence(missing))
        return missing

    async def delete_model(self, model_name: str, remove_from_ollama: bool = False) -> bool:
        """Unregisters the model from memory and models.json; with remove_from_ollama also deletes it from Ollama.
        Returns True if the model was deleted from Ollama.
        Raises UnknownModelError, DeleteBlockedError, ConnectionError (Ollama problem) or ConfigError."""
        config = self.model_config_dict.get(model_name)
        if config is None:
            raise UnknownModelError(f"Unknown model {model_name}.")
        state = self.pulls.get(model_name)
        if state is not None and state.state in ("queued", "pulling"):
            raise DeleteBlockedError(f"{model_name} is being downloaded: wait for it to finish before deleting it.")

        removed_from_ollama = False
        if remove_from_ollama:
            tag = config.model_id
            others = [a for a, c in self.model_config_dict.items()
                      if a != model_name and c.provider == "ollama" and c.model_id == tag]
            if others:
                raise DeleteBlockedError(f"Ollama model {tag} is also used by: {', '.join(others)}. "
                                         "Delete those first, or delete this one without remove_from_ollama.")
            # Ollama goes first: if it fails, nothing has changed yet
            if self.is_pulled(tag, await self.list_ollama_models()):
                try:
                    await self.ollama_client.delete(tag)
                except ResponseError as e:
                    raise ConnectionError(f"Ollama could not delete {tag}: {e}") from e
                removed_from_ollama = True

        if self.model_config_dict.pop(model_name, None) is None:      # another request deleted it during the awaits
            raise UnknownModelError(f"Unknown model {model_name}.")
        try:
            self.save()
        except OSError as e:
            self.model_config_dict[model_name] = config               # keep memory and file consistent
            raise ConfigError(f"Could not write {self.model_config_path}: {e}") from e
        self.pulls.pop(model_name, None)                              # so a re-added alias doesn't inherit an old status
        self._pull_tasks.pop(model_name, None)
        self.model_history[model_name] = HistoryEntry(config=config, deleted_at=datetime.now(timezone.utc),
                                                    removed_from_ollama=removed_from_ollama)
        try:
            self.save_history()
        except OSError:
            # The model is already deleted: don't fail the request. It stays restorable until the app restarts.
            logger.exception("Could not write %s", self.model_history_path)
        return removed_from_ollama

    async def restore_model(self, model_name: str) -> bool:
        """Puts a deleted model back (memory + models.json) with its saved parameters, and pulls it if Ollama
        doesn't have it. Returns True if a download was started.
        Raises UnknownModelError, DuplicateModelError, ConnectionError or ConfigError."""
        entry = self.model_history.get(model_name)
        if entry is None:
            raise UnknownModelError(f"{model_name} is not in the deleted models history.")
        pulling = await self.add_model(model_name, entry.config)   # duplicate check, Ollama check, save, pull
        self.model_history.pop(model_name, None)
        try:
            self.save_history()
        except OSError:
            logger.exception("Could not write %s", self.model_history_path)
        return pulling

    async def close(self) -> None:
        """Cancels downloads still running when the app shuts down."""
        tasks = list(self._pull_tasks.values())
        if self._startup_task is not None:
            tasks.append(self._startup_task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.ollama_client.close()