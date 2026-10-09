from fastapi import Request
from src.model_management.model_manager import ModelManager

def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.res.model_config_manager