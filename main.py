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
    app.openapi_schema = None                       # drop FastAPI's cache so the alias list is always current
    schema = _default_openapi()
    res = getattr(app.state, "res", None)
    aliases = list(res.model_config_manager.model_config_dict) if res else []
    if aliases:
        for path_item in schema["paths"].values():
            for operation in path_item.values():
                for param in operation.get("parameters", []):
                    if param["in"] == "path" and param["name"] == "alias":
                        param["schema"]["enum"] = aliases   # Swagger renders an enum as a dropdown
    return schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
