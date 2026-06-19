from __future__ import annotations

"""NeuralGuard — LLM Guard / AI Application Firewall."""

from typing import TYPE_CHECKING

try:
    from importlib.metadata import version as _version

    __version__ = _version("neuralguard")
except Exception:
    __version__ = "0.1.0"  # fallback for editable installs


if TYPE_CHECKING:
    from fastapi import FastAPI

    from neuralguard.config.settings import NeuralGuardConfig


def create_app(config: NeuralGuardConfig | None = None) -> FastAPI:
    """Lazy import to avoid circular dependency."""
    from neuralguard.main import create_app as _create

    return _create(config)


def main() -> None:
    """Lazy import to avoid circular dependency."""
    from neuralguard.main import main as _main

    _main()


__all__: list[str] = ["__version__", "create_app", "main"]
