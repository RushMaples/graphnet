"""Transformer-specific modules with lazily loaded optional architectures."""

from importlib import import_module
from typing import Any

__all__ = ["ISeeCube"]


def __getattr__(name: str) -> Any:
    """Load ISeeCube only when requested, keeping torchscale optional."""
    if name == "ISeeCube":
        model = import_module(".iseecube", __name__).ISeeCube
        globals()[name] = model
        return model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    """Include lazy public exports in module introspection."""
    return sorted(set(globals()) | set(__all__))
