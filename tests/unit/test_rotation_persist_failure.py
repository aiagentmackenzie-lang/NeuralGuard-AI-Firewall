"""P2-4: a durable rotation that cannot be written must not be applied.

Found in the two-pod rotation drill (2026-11-17). The keys-file volume was not
writable by the container user, so ``_persist`` raised **after** ``add()`` had
already put the key in the process's live map:

- pod A answered the caller with ``500 internal_error`` — so the caller could not
  tell whether the rotation happened;
- pod A then **accepted** the new key (``key_count`` 2 → 3, a new ``key_set_id``)
  while pod B still refused it, and the key existed in no file anywhere;
- a restart of pod A would have silently dropped a credential that had been
  handed to a client.

That is a split-brain credential: the one outcome a fail-closed control must
never produce is "accepted here, persisted nowhere, reported as a failure".

The contracts below fix the order and the vocabulary:

1. **Persist first, then apply.** A write that fails leaves the live key set
   exactly as it was, so the caller's 500 is true.
2. **Name the failure.** ``500 {"error": "key_store_write_failed"}`` says the
   rotation was NOT applied; ``internal_error`` says nothing.
3. The healthy path is unchanged: 200, ``persisted: true``, the key works, and a
   fresh store instance (the other pod's boot) reads it back from the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.auth.jwtauth import RuntimeKeyStore
from neuralguard.config.settings import AuthSettings, NeuralGuardConfig
from neuralguard.main import create_app

_ADMIN = "admin-key-0123456789abcdef"
_TENANT = "tenant-key-0123456789abcdef"
_NEW = "rotated-key-0123456789abcdef"


def _unwritable_store_path(tmp_path: Path) -> Path:
    """A path whose write is guaranteed to fail: it is a directory.

    Deterministic on every POSIX filesystem and needs no permission games (a
    chmod-based fixture is unreliable when tests run as root).
    """
    path = tmp_path / "keys.json"
    path.mkdir()
    return path


class TestTheStoreDoesNotMutateWithoutDurability:
    def test_add_raises_and_leaves_the_live_set_untouched(self, tmp_path: Path) -> None:
        store = RuntimeKeyStore(_unwritable_store_path(tmp_path))
        assert store.all_keys() == {}
        with pytest.raises(OSError):
            store.add(_NEW, "acme")
        assert store.all_keys() == {}, "a key that could not be written must not be live"

    def test_remove_raises_and_keeps_the_key(self, tmp_path: Path) -> None:
        path = tmp_path / "keys.json"
        store = RuntimeKeyStore(path)
        store.add(_NEW, "acme")
        assert store.all_keys() == {_NEW: "acme"}
        # Replace the file with a directory so the next write fails.
        path.unlink()
        path.mkdir()
        with pytest.raises(OSError):
            store.remove(_NEW)
        assert store.all_keys() == {_NEW: "acme"}, "an unpersisted removal must not be live"


def _app(keys_file: Path) -> Any:
    return create_app(
        NeuralGuardConfig(
            environment="development",
            auth=AuthSettings(
                enabled=True,
                api_keys=[f"{_ADMIN}|admin", f"{_TENANT}|acme"],
                public_endpoints=[],
                keys_file=keys_file,
            ),
        )
    )


async def _rotate(app: Any) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        return await client.post(
            "/v1/auth/keys/rotate",
            json={"new_key": _NEW, "tenant": "acme"},
            headers={"Authorization": f"Bearer {_ADMIN}"},
        )


async def _accepted(app: Any, key: str) -> bool:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        response = await client.get("/v1/info", headers={"Authorization": f"Bearer {key}"})
    return response.status_code == 200


class TestTheRouteDoesNotReportAFailureItDidNotHave:
    @pytest.mark.asyncio
    async def test_a_failed_write_is_named_and_the_key_is_not_accepted(
        self, tmp_path: Path
    ) -> None:
        app = _app(_unwritable_store_path(tmp_path))
        response = await _rotate(app)
        assert response.status_code == 500, response.text
        body = response.json()
        assert body["error"] == "key_store_write_failed", body
        # The whole point: the 500 must be TRUE. The key must not work.
        assert await _accepted(app, _NEW) is False, (
            "the rotation was reported as failed and took effect"
        )

    @pytest.mark.asyncio
    async def test_a_writable_store_still_rotates_and_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "keys.json"
        app = _app(path)
        response = await _rotate(app)
        assert response.status_code == 200, response.text
        assert response.json()["persisted"] is True
        assert await _accepted(app, _NEW) is True
        # A second process (another pod booting) loads the same file.
        assert RuntimeKeyStore(path).all_keys() == {_NEW: "acme"}

    @pytest.mark.asyncio
    async def test_the_posture_reports_the_set_that_is_actually_live(self, tmp_path: Path) -> None:
        app = _app(_unwritable_store_path(tmp_path))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            before = (
                await client.get("/v1/info", headers={"Authorization": f"Bearer {_ADMIN}"})
            ).json()
            await client.post(
                "/v1/auth/keys/rotate",
                json={"new_key": _NEW, "tenant": "acme"},
                headers={"Authorization": f"Bearer {_ADMIN}"},
            )
            after = (
                await client.get("/v1/info", headers={"Authorization": f"Bearer {_ADMIN}"})
            ).json()
        assert before["auth"]["key_count"] == 2
        assert after["auth"]["key_count"] == 2, "a failed rotation must not change the reported set"
        assert after["auth"]["key_set_id"] == before["auth"]["key_set_id"]
