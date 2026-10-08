from dotenv import load_dotenv
from fastapi import FastAPI
from typing import AsyncGenerator
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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
