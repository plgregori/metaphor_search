from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.errors import (ConfigError, DeleteBlockedError, DuplicateModelError, PromptFileError,
                                           OllamaUnavailableError, UnknownModelError)

_STATUS_BY_ERROR: dict[type[Exception], int] = {
    UnknownModelError: 404,
    DuplicateModelError: 409,
    DeleteBlockedError: 409,
    PromptFileError: 422,
    OllamaUnavailableError: 503,
    ConfigError: 500,
}

def register_error_handlers(app: FastAPI) -> None:
    for error_type, status_code in _STATUS_BY_ERROR.items():
        async def handler(request: Request, exc: Exception, status_code: int = status_code) -> JSONResponse:
            return JSONResponse(status_code=status_code, content={"detail": str(exc)})
        app.add_exception_handler(error_type, handler)