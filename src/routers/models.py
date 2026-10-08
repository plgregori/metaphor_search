from typing import Annotated
from pydantic import BaseModel, ConfigDict, ValidationError
from fastapi import APIRouter, Depends, Form, HTTPException, Path, Query, Request, Response
from src.dependencies.model_config import (ConfigError, ModelConfig, ModelManager, PullState, HistoryEntry,
                                           DuplicateModelError, UnknownModelError, DeleteBlockedError)

router = APIRouter(prefix="/models", tags=["Models"])

def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.res.model_config_manager

class _AliasField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alias: str      # the name you choose for this model in the app (the key in models.json)

class AddModelForm(ModelConfig, _AliasField):
    """One input per field in Swagger: alias, then the ModelConfig fields."""

class ModelEntry(BaseModel):
    config: ModelConfig
    available: bool | None = None   # None = not checked (not an ollama model, or Ollama unreachable)
    pull: PullState | None = None

class DeleteResult(BaseModel):
    alias: str
    removed_from_models_json: bool
    removed_from_ollama: bool

@router.get("", response_model=dict[str, ModelEntry])
async def list_models(manager: ModelManager = Depends(get_model_manager)) -> dict[str, ModelEntry]:
    availability = await manager.availability()
    return {alias: ModelEntry(config=config, available=availability.get(alias), pull=manager.pulls.get(alias))
            for alias, config in manager.model_config_dict.items()}

@router.post("", response_model=ModelEntry, status_code=201)
async def add_model(form: Annotated[AddModelForm, Form()], response: Response,
                    manager: ModelManager = Depends(get_model_manager)) -> ModelEntry:
    alias = form.alias
    config = ModelConfig(**form.model_dump(exclude={"alias"}))
    try:
        pulling = await manager.add_model(alias, config)
    except ConfigError as e:                       # models.json could not be written
        raise HTTPException(status_code=500, detail=str(e))
    except DuplicateModelError as e:                        # alias already registered
        raise HTTPException(status_code=409, detail=str(e))
    except ConnectionError as e:                   # Ollama unreachable
        raise HTTPException(status_code=503, detail=str(e))
    if pulling:
        response.status_code = 202                 # accepted: download continues in the background
    return ModelEntry(config=config,
                      available=(not pulling) if config.provider == "ollama" else None,
                      pull=manager.pulls.get(alias))

@router.post("/{alias}/pull", response_model=ModelEntry, status_code=202)
async def pull_model(alias: str, manager: ModelManager = Depends(get_model_manager)) -> ModelEntry:
    config = manager.model_config_dict.get(alias)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Unknown model {alias}.")
    if config.provider != "ollama":
        raise HTTPException(status_code=422, detail="Only ollama models can be pulled.")
    manager.start_pull(alias)
    availability = await manager.availability()
    return ModelEntry(config=config, available=availability.get(alias), pull=manager.pulls.get(alias))

@router.patch("/{alias}", response_model=ModelEntry)
async def edit_model(alias: Annotated[str, Path(description="Name of a model already registered in models.json")],
                    fine_tuning: Annotated[bool | None, Form()] = None,
                    rag: Annotated[bool | None, Form()] = None,
                    prompt_engineering: Annotated[bool | None, Form()] = None,
                    hf_id: Annotated[str | None, Form(description="Hugging Face id. Leave empty to keep the current one.")] = None,
                    clear_hf_id: Annotated[bool, Form(description="Set to true to remove the current hf_id.")] = False,
                    manager: ModelManager = Depends(get_model_manager)) -> ModelEntry:
    changes: dict[str, bool | str | None] = {}
    if fine_tuning is not None:
        changes["fine_tuning"] = fine_tuning
    if rag is not None:
        changes["rag"] = rag
    if prompt_engineering is not None:
        changes["prompt_engineering"] = prompt_engineering
    if hf_id is not None and hf_id.strip():
        changes["hf_id"] = hf_id.strip()
    if clear_hf_id:
        if "hf_id" in changes:
            raise HTTPException(status_code=422, detail="Give either hf_id or clear_hf_id, not both.")
        changes["hf_id"] = None
    if not changes:
        raise HTTPException(status_code=422, detail="Nothing to change: set at least one field.")

    try:
        updated = manager.edit_model(alias, changes)
    except UnknownModelError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:                   # e.g. fine_tuning=true on an ollama model without hf_id
        raise HTTPException(status_code=422, detail=e.errors(include_url=False, include_context=False))
    except ConfigError as e:                       # models.json could not be written
        raise HTTPException(status_code=500, detail=str(e))
    return ModelEntry(config=updated, pull=manager.pulls.get(alias))

@router.delete("/{alias}", response_model=DeleteResult)
async def delete_model(alias: Annotated[str, Path(description="Name of a model already registered in models.json")],
                       remove_from_ollama: Annotated[bool, Query(description="Also delete the model from Ollama (ollama models only). If false, it is only removed from models.json.")] = False,
                       manager: ModelManager = Depends(get_model_manager)) -> DeleteResult:
    config = manager.model_config_dict.get(alias)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Unknown model {alias}.")
    if remove_from_ollama and config.provider != "ollama":
        raise HTTPException(status_code=422, detail="Only ollama models can be removed from Ollama.")
    try:
        removed_from_ollama = await manager.delete_model(alias, remove_from_ollama)
    except UnknownModelError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DeleteBlockedError as e:                # downloading, or its Ollama tag is shared with another alias
        raise HTTPException(status_code=409, detail=str(e))
    except ConnectionError as e:                   # Ollama unreachable or refused
        raise HTTPException(status_code=503, detail=str(e))
    except ConfigError as e:                       # models.json could not be written
        raise HTTPException(status_code=500, detail=str(e))
    return DeleteResult(alias=alias, removed_from_models_json=True, removed_from_ollama=removed_from_ollama)

@router.get("/history", response_model=dict[str, HistoryEntry])
async def deleted_models(manager: ModelManager = Depends(get_model_manager)) -> dict[str, HistoryEntry]:
    return manager.model_history

@router.post("/{deleted_alias}/restore", response_model=ModelEntry)
async def restore_model(deleted_alias: Annotated[str, Path(description="Name of a deleted model (see GET /models/history)")],
                        response: Response,
                        manager: ModelManager = Depends(get_model_manager)) -> ModelEntry:
    try:
        pulling = await manager.restore_model(deleted_alias)
    except UnknownModelError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DuplicateModelError as e:               # a model with this alias was registered after the deletion
        raise HTTPException(status_code=409, detail=str(e))
    except ConnectionError as e:                   # Ollama unreachable
        raise HTTPException(status_code=503, detail=str(e))
    except ConfigError as e:                       # models.json could not be written
        raise HTTPException(status_code=500, detail=str(e))
    if pulling:
        response.status_code = 202                 # download continues in the background
    config = manager.model_config_dict[deleted_alias]
    return ModelEntry(config=config,
                      available=(not pulling) if config.provider == "ollama" else None,
                      pull=manager.pulls.get(deleted_alias))