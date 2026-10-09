from typing import Any
from fastapi import FastAPI


def install_dynamic_dropdowns(app: FastAPI) -> None:
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
