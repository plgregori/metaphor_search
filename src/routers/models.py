from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from src.dependencies.model_config import ConfigError, ModelConfig, ModelManager, PullState, DuplicateModelError

router = APIRouter(prefix="/models", tags=["models"])

def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.res.model_config_manager

class ModelEntry(BaseModel):
    config: ModelConfig
    available: bool | None = None   # None = not checked (not an ollama model, or Ollama unreachable)
    pull: PullState | None = None

@router.get("", response_model=dict[str, ModelEntry])
async def list_models(manager: ModelManager = Depends(get_model_manager)) -> dict[str, ModelEntry]:
    availability = await manager.availability()
    return {alias: ModelEntry(config=config, available=availability.get(alias), pull=manager.pulls.get(alias))
            for alias, config in manager.model_config_dict.items()}

@router.post("/{alias}", response_model=ModelEntry, status_code=201)
async def add_model(alias: str, config: ModelConfig, response: Response,
                    manager: ModelManager = Depends(get_model_manager)) -> ModelEntry:
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