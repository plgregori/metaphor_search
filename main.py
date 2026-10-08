from dotenv import load_dotenv
from fastapi import FastAPI
from typing import Any, AsyncGenerator
from src.dependencies.resources import Resources
from src.routers import models
from contextlib import asynccontextmanager
import uvicorn


load_dotenv()

@asynccontextmanager
async def lifespan(api: FastAPI) -> AsyncGenerator[None, None]:
    resources = Resources()
    try:
        await resources.initialize()
        api.state.res = resources
        yield
    finally:
        await resources.close()

app = FastAPI(lifespan=lifespan)
app.include_router(models.router)

_default_openapi = app.openapi

def custom_openapi() -> dict[str, Any]:
    app.openapi_schema = None
    schema = _default_openapi()
    res = getattr(app.state, "res", None)
    dropdowns: dict[str, list[str]] = {}
    if res:
        manager = res.model_config_manager
        dropdowns = {"alias": list(manager.model_config_dict),
                     "deleted_alias": list(manager.model_history)}
    for path_item in schema["paths"].values():
        for operation in path_item.values():
            for param in operation.get("parameters", []):
                values = dropdowns.get(param["name"])
                if param["in"] == "path" and values:
                    param["schema"]["enum"] = values
    return schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
