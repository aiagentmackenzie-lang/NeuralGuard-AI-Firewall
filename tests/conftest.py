"""Shared pytest fixtures.

F17: the suite must be hermetic. ``NeuralGuardConfig`` and every sub-settings
class are pydantic-settings models with ``env_file=".env"`` resolved against
the current working directory, so a developer's repo-root ``.env`` (auth keys,
canary secret, judge model) silently reconfigures the app under test — with
the dev .env present, 69 endpoint/tenant tests failed with 401s while the same
suite passed in CI. Exported ``NEURALGUARD_*`` shell variables would pollute
the same way.

The autouse fixture below severs BOTH channels for every test:

1. purges ``NEURALGUARD_*`` from the real environment, and
2. disables the ``env_file`` (.env) lookup on every settings class.

Tests that deliberately exercise environment handling call
``monkeypatch.setenv(...)`` inside the test body (after this fixture runs);
real environment variables still take priority over defaults — only the FILE
channel is severed, not the env-var channel.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from pydantic_settings import BaseSettings

from neuralguard.config import settings as settings_module

if TYPE_CHECKING:
    from collections.abc import Iterator

# All BaseSettings subclasses defined in the settings module (each carries its
# own model_config, so every class must be severed individually).
_SETTINGS_CLASSES: tuple[type[BaseSettings], ...] = tuple(
    obj
    for obj in vars(settings_module).values()
    if isinstance(obj, type) and issubclass(obj, BaseSettings) and obj is not BaseSettings
)


@pytest.fixture(autouse=True)
def _hermetic_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate every test from the operator's .env and exported env vars (F17)."""
    for name in [key for key in os.environ if key.startswith("NEURALGUARD_")]:
        monkeypatch.delenv(name, raising=False)
    for cls in _SETTINGS_CLASSES:
        monkeypatch.setitem(cls.model_config, "env_file", None)
    yield
