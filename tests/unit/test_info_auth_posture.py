"""P2-4: the live API-key set, readable per pod — the rotation observability gap.

Rotating a static key in a multi-replica deployment is documented as
"env/redeploy", and the reason that path is the recommended one is that the
runtime rotation API is **per process** (the runbook says so). Neither path was
observable from outside: after a Secret change you could watch pods roll, but
you could not ask a pod *which key set am I actually serving*, so "has the old
key been revoked everywhere?" was a question answered by waiting and hoping.

`/v1/info` → `auth` now answers it, per pod:

- ``enabled`` — whether auth is enforced;
- ``key_count`` and ``key_set_id`` — a 12-hex fingerprint over the **live** key
  set (env keys merged with the durable runtime store), computed from
  ``sha256(key)`` plus each key's tenant and workload-agent binding, sorted, so
  it is order-insensitive and cannot be reversed. Two pods reporting the same
  ``key_set_id`` are serving the same credentials; during a rollout they differ,
  and that difference is the revocation window made visible.
- ``runtime_rotation_durable`` — whether a runtime rotation would survive a
  restart (``NEURALGUARD_AUTH_KEYS_FILE`` configured). Without it, production
  refuses runtime rotation, which is why env/redeploy is the documented path.
- ``jwt_enabled`` — whether short-lived HS256 tokens are accepted.

The fingerprint is computed over digests, never over key material: the response
must not carry a key, a prefix of one, or anything that shortens the search for
one. Confirming a candidate key from this field requires already holding it, at
which point you can simply authenticate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import AuthSettings, NeuralGuardConfig
from neuralguard.main import create_app

_HEX12 = re.compile(r"^[0-9a-f]{12}$")
_ADMIN = "admin-key-0123456789abcdef"
_TENANT = "tenant-key-0123456789abcdef"
_THIRD = "third-key-0123456789abcdef"


def _config(
    keys: list[str] | None = None,
    *,
    keys_file: Path | None = None,
    jwt_enabled: bool = False,
) -> NeuralGuardConfig:
    config = NeuralGuardConfig(
        environment="development",
        auth=AuthSettings(
            enabled=True,
            api_keys=[f"{_ADMIN}|admin", f"{_TENANT}|acme"] if keys is None else keys,
            public_endpoints=[],
            keys_file=keys_file,
            jwt_enabled=jwt_enabled,
            jwt_secret="x" * 40 if jwt_enabled else None,
        ),
    )
    return config


async def _auth_block(config: NeuralGuardConfig, key: str = _ADMIN) -> dict[str, Any]:
    app = create_app(config)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        response = await client.get("/v1/info", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200, response.text
    block: dict[str, Any] = response.json()["auth"]
    return block


class TestTheKeySetIsReported:
    async def test_the_live_key_set_has_a_fingerprint_and_a_count(self) -> None:
        block = await _auth_block(_config())
        assert block["enabled"] is True
        assert block["key_count"] == 2
        assert _HEX12.match(block["key_set_id"]), block["key_set_id"]

    async def test_the_fingerprint_is_order_insensitive(self) -> None:
        # Env order changes whenever an operator edits the list; a fingerprint
        # that moved with it would report a rotation that did not happen.
        forward = await _auth_block(_config([f"{_ADMIN}|admin", f"{_TENANT}|acme"]))
        reverse = await _auth_block(_config([f"{_TENANT}|acme", f"{_ADMIN}|admin"]))
        assert forward["key_set_id"] == reverse["key_set_id"]

    async def test_rebinding_a_key_to_another_tenant_is_a_different_posture(self) -> None:
        first = await _auth_block(_config())
        rebound = await _auth_block(_config([f"{_ADMIN}|admin", f"{_TENANT}|other"]))
        assert first["key_set_id"] != rebound["key_set_id"]

    async def test_adding_a_key_changes_the_id_and_removing_it_restores_it(self) -> None:
        before = await _auth_block(_config())
        after = await _auth_block(_config([f"{_ADMIN}|admin", f"{_TENANT}|acme", f"{_THIRD}|acme"]))
        assert before["key_set_id"] != after["key_set_id"]
        assert after["key_count"] == 3
        restored = await _auth_block(_config())
        assert restored["key_set_id"] == before["key_set_id"], (
            "the id must be a pure function of the set"
        )

    async def test_no_keys_reports_zero_and_no_id_rather_than_a_fake_one(self) -> None:
        # Auth enforced with an empty key set is a fail-closed misconfiguration:
        # every call is 401, so the field is only observable with auth off.
        config = _config([])
        config.auth.enabled = False
        block = await _auth_block(config, key="")
        assert block["enabled"] is False
        assert block["key_count"] == 0
        assert block["key_set_id"] is None, "an empty set has no identity; do not invent one"

    async def test_an_empty_key_set_with_auth_enforced_is_unreachable_by_design(self) -> None:
        # Fail-closed, and the reason the field above is only readable when auth
        # is off: there is no credential to present.
        app = create_app(_config([]))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            response = await client.get("/v1/info")
        assert response.status_code == 401


class TestTheFingerprintNeverCarriesKeyMaterial:
    async def test_the_response_contains_no_key_and_no_prefix_of_one(self) -> None:
        app = create_app(_config())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            response = await client.get("/v1/info", headers={"Authorization": f"Bearer {_ADMIN}"})
        serialised = response.text
        for key in (_ADMIN, _TENANT):
            assert key not in serialised
            assert key[:8] not in serialised, "a prefix shortens the search for the key"
            assert key[-8:] not in serialised

    async def test_the_fingerprint_is_not_the_key_digest_itself(self) -> None:
        import hashlib

        block = await _auth_block(_config())
        raw = hashlib.sha256(_ADMIN.encode()).hexdigest()
        assert block["key_set_id"] not in raw


class TestTheRotationPostureIsHonest:
    async def test_runtime_rotation_is_reported_as_not_durable_without_a_keys_file(self) -> None:
        block = await _auth_block(_config())
        # Production refuses runtime-only rotation, which is why env/redeploy is
        # the documented path. The field says which mode this pod is in.
        assert block["runtime_rotation_durable"] is False

    async def test_a_keys_file_makes_runtime_rotation_durable(self, tmp_path: Path) -> None:
        block = await _auth_block(_config(keys_file=tmp_path / "keys.json"))
        assert block["runtime_rotation_durable"] is True

    async def test_jwt_support_is_reported(self) -> None:
        assert (await _auth_block(_config()))["jwt_enabled"] is False
        assert (await _auth_block(_config(jwt_enabled=True)))["jwt_enabled"] is True


class TestTheLiveSetNotTheConfiguredSet:
    async def test_a_runtime_rotation_changes_the_reported_id(self, tmp_path: Path) -> None:
        # The point of the field: after a rotation, the pod that rotated says so.
        # This is the per-process property the runbook warns about, made visible.
        config = _config(keys_file=tmp_path / "keys.json")
        app = create_app(config)
        headers = {"Authorization": f"Bearer {_ADMIN}"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            before = (await client.get("/v1/info", headers=headers)).json()["auth"]
            rotated = await client.post(
                "/v1/auth/keys/rotate",
                json={"new_key": "rotated-key-0123456789abcdef", "tenant": "acme"},
                headers=headers,
            )
            assert rotated.status_code == 200, rotated.text
            after = (await client.get("/v1/info", headers=headers)).json()["auth"]
        assert rotated.json()["persisted"] is True
        assert before["key_count"] == 2
        assert after["key_count"] == 3
        assert before["key_set_id"] != after["key_set_id"]
        # And the durable store really holds it, so a restart reports the same set.
        stored = json.loads((tmp_path / "keys.json").read_text(encoding="utf-8"))
        assert [entry["key"] for entry in stored["keys"]] == ["rotated-key-0123456789abcdef"]
