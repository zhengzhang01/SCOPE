import importlib
from typing import Any, Type


_MODEL_REGISTRY = {
    "moge": ("moge", "MoGeModel"),
    "scope": ("scope", "ScopeModel"),
}

def normalize_model_name(model_name: str) -> str:
    return model_name


def import_model_class(model_name: str) -> Type[Any]:
    model_name = normalize_model_name(model_name)
    if model_name not in _MODEL_REGISTRY:
        supported = ", ".join(_MODEL_REGISTRY)
        raise ValueError(f'Unsupported model name: {model_name}. Supported models: {supported}')

    module_name, class_name = _MODEL_REGISTRY[model_name]
    module = importlib.import_module(f".{module_name}", __package__)
    return getattr(module, class_name)
