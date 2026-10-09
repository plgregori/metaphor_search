from dotenv import load_dotenv
from fastapi import FastAPI
from typing import AsyncGenerator
from src.app_resources import AppResources
from src.api.routers import models
from src.api.error_handlers import register_error_handlers
from src.api.openapi import install_dynamic_dropdowns
from contextlib import asynccontextmanager
import uvicorn

load_dotenv()

@asynccontextmanager
async def lifespan(api: FastAPI) -> AsyncGenerator[None, None]:
    resources = AppResources()
    try:
        await resources.initialize()
        api.state.res = resources
        yield
    finally:
        await resources.close()

app = FastAPI(lifespan=lifespan)
app.include_router(models.router)

install_dynamic_dropdowns(app)
register_error_handlers(app)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
