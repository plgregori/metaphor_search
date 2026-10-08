from openai import AsyncOpenAI
from src.dependencies.model_config import ModelManager
import os

class Resources():
    model_config_manager: ModelManager

    openai_client: AsyncOpenAI | None
    

    def __init__(self) -> None:
        model_config_path = os.getenv("MODEL_CONFIG_PATH", "configuration/models.json")
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model_config_manager = ModelManager.load_from_json(model_config_path, ollama_url)
        self.openai_client = None

    async def initialize(self) -> None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        await self.model_config_manager.pull_missing()

    async def close(self) -> None:
        try:
            await self.model_config_manager.close()
        finally:
            if self.openai_client is not None:
                await self.openai_client.close()