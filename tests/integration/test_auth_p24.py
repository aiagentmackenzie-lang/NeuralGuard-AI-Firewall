"""P2-4: JWT bearer auth + runtime key rotation — end-to-end through the app.

Covers: token issuance/verification roundtrip, alg-allowlist rejection,
expiry enforcement, tenant binding via JWT claim, admin-gated rotation,
durable persistence via keys_file, production refusal of non-durable
rotation, and the F5 known-key surface for the new NEURALGUARD_AUTH_* knobs.
"""

from __future__ import annotations

import json
from typing import Any

import jwt as pyjwt
import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.auth.jwtauth import JwtManager, RuntimeKeyStore
from neuralguard.config.settings import AuthSettings, NeuralGuardConfig, known_env_keys
from neuralguard.main import create_app

SECRET = "unit-test-jwt-secret-0123456789abcdef-0123456789abcdef"  # ≥32 chars


def _config(
    api_keys: list[str],
    jwt_enabled: bool = True,
    keys_file: Any = None,
    environment: str = "development",
) -> NeuralGuardConfig:
    return NeuralGuardConfig(
        environment=environment,
        auth=AuthSettings(
            enabled=True,
            api_keys=api_keys,
            jwt_enabled=jwt_enabled,
            jwt_secret=SECRET if jwt_enabled else None,
            keys_file=keys_file,
        ),
    )


@pytest.fixture
async def jwt_client():
    app = create_app(_config(["admin-key|admin", "user-key|acme"]))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def rotation_client(tmp_path: Any):
    app = create_app(
        _config(
            ["admin-key|admin", "user-key|acme"],
            keys_file=tmp_path / "keys" / "runtime_keys.json",
        )
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestJwtAuth:
    @pytest.mark.asyncio
    async def test_issue_and_use_token(self, jwt_client) -> None:
        r = await jwt_client.post(
            "/v1/auth/token",
            json={},
            headers={"Authorization": "Bearer user-key"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["token_type"] == "bearer"
        assert body["tenant"] == "acme"

        # The token authenticates an evaluate call with the JWT tenant.
        r2 = await jwt_client.post(
            "/v1/evaluate",
            json={"prompt": "What is the weather?", "tenant_id": "acme"},
            headers={"Authorization": f"Bearer {body['access_token']}"},
        )
        assert r2.status_code == 200

    @pytest.mark.asyncio
    async def test_static_keys_still_work(self, jwt_client) -> None:
        r = await jwt_client.post(
            "/v1/evaluate",
            json={"prompt": "hello", "tenant_id": "acme"},
            headers={"Authorization": "Bearer user-key"},
        )
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_expired_token_rejected(self, jwt_client) -> None:
        manager = JwtManager(AuthSettings(enabled=True, jwt_enabled=True, jwt_secret=SECRET))
        expired = manager.issue("acme", ttl_minutes=-1)
        r = await jwt_client.post(
            "/v1/evaluate",
            json={"prompt": "hello"},
            headers={"Authorization": f"Bearer {expired}"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_signature_rejected(self, jwt_client) -> None:
        forged = pyjwt.encode(
            {"iss": "neuralguard", "tenant": "admin", "exp": 99999999999},
            "an-entirely-different-secret-value-aaaaaaaaaaaa",
            algorithm="HS256",
        )
        r = await jwt_client.post(
            "/v1/evaluate",
            json={"prompt": "hello"},
            headers={"Authorization": f"Bearer {forged}"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_alg_none_rejected(self, jwt_client) -> None:
        token = pyjwt.encode(
            {"iss": "neuralguard", "tenant": "admin", "exp": 99999999999},
            key="",
            algorithm="none",
            headers={"alg": "none"},
        )
        r = await jwt_client.post(
            "/v1/evaluate",
            json={"prompt": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_hs512_rejected_by_allowlist(self, jwt_client) -> None:
        token = pyjwt.encode(
            {"iss": "neuralguard", "tenant": "admin", "exp": 99999999999},
            SECRET,
            algorithm="HS512",
        )
        r = await jwt_client.post(
            "/v1/evaluate",
            json={"prompt": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_token_without_tenant_claim_rejected(self, jwt_client) -> None:
        token = pyjwt.encode({"iss": "neuralguard", "exp": 99999999999}, SECRET, algorithm="HS256")
        r = await jwt_client.post(
            "/v1/evaluate",
            json={"prompt": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_jwt_tenant_binding_enforced(self, jwt_client) -> None:
        """A token bound to acme cannot act as another tenant (403 on mismatch)."""
        r = await jwt_client.post(
            "/v1/auth/token", json={}, headers={"Authorization": "Bearer user-key"}
        )
        token = r.json()["access_token"]
        r2 = await jwt_client.post(
            "/v1/evaluate",
            json={"prompt": "hello", "tenant_id": "other-tenant"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r2.status_code == 403  # tenant-spoofing blocked for JWTs too

    @pytest.mark.asyncio
    async def test_jwt_disabled_404_on_token_route(self) -> None:
        app = create_app(_config(["k|default"], jwt_enabled=False))
        # JWT route not mounted without jwt_manager or key_store.
        assert not any(r.path == "/v1/auth/token" for r in app.routes)


class TestKeyRotation:
    @pytest.mark.asyncio
    async def test_admin_only(self, rotation_client) -> None:
        r = await rotation_client.post(
            "/v1/auth/keys/rotate",
            json={"tenant": "acme"},
            headers={"Authorization": "Bearer user-key"},  # non-admin tenant
        )
        assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_rotate_issue_and_use(self, rotation_client, tmp_path: Any) -> None:
        r = await rotation_client.post(
            "/v1/auth/keys/rotate",
            json={"tenant": "acme"},
            headers={"Authorization": "Bearer admin-key"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["persisted"] is True
        assert body["key"].startswith("ng_")

        # Old key still valid (env keys are the bootstrap path), new key works.
        r2 = await rotation_client.post(
            "/v1/evaluate",
            json={"prompt": "hello", "tenant_id": "acme"},
            headers={"Authorization": f"Bearer {body['key']}"},
        )
        assert r2.status_code == 200

        # Durable: file written with 0600 and reloadable.
        keys_file = tmp_path / "keys" / "runtime_keys.json"
        assert keys_file.exists()
        assert (keys_file.stat().st_mode & 0o777) == 0o600
        data = json.loads(keys_file.read_text())
        assert any(k["key"] == body["key"] for k in data["keys"])

        store = RuntimeKeyStore(keys_file)
        assert store.all_keys().get(body["key"]) == "acme"

    @pytest.mark.asyncio
    async def test_revoke_caller_key(self, rotation_client) -> None:
        r = await rotation_client.post(
            "/v1/auth/keys/rotate",
            json={"tenant": "admin", "revoke_caller_key": True},
            headers={"Authorization": "Bearer admin-key"},
        )
        assert r.status_code == 200
        assert r.json()["revoked_caller_key"] is True
        r2 = await rotation_client.post(
            "/v1/evaluate",
            json={"prompt": "hello"},
            headers={"Authorization": "Bearer admin-key"},
        )
        assert r2.status_code == 401  # revoked

    @pytest.mark.asyncio
    async def test_runtime_only_refused_in_production(self) -> None:
        app = create_app(_config(["admin-key|admin"], keys_file=None, environment="production"))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/v1/auth/keys/rotate",
                json={"tenant": "acme"},
                headers={"Authorization": "Bearer admin-key"},
            )
            assert r.status_code == 409

    @pytest.mark.asyncio
    async def test_supplied_key_accepted(self, rotation_client) -> None:
        r = await rotation_client.post(
            "/v1/auth/keys/rotate",
            json={"tenant": "acme", "new_key": "my-rotated-key-2026"},
            headers={"Authorization": "Bearer admin-key"},
        )
        assert r.status_code == 200
        assert r.json()["key"] == "my-rotated-key-2026"


class TestConfigSurface:
    def test_jwt_requires_secret(self) -> None:
        with pytest.raises(Exception, match="jwt_enabled=true requires"):
            AuthSettings(enabled=True, jwt_enabled=True)

    def test_jwt_secret_min_length(self) -> None:
        with pytest.raises(Exception, match="≥32"):
            AuthSettings(enabled=True, jwt_enabled=True, jwt_secret="short")

    def test_new_auth_keys_known_to_f5_gate(self) -> None:
        known = known_env_keys()
        for key in (
            "NEURALGUARD_AUTH_JWT_ENABLED",
            "NEURALGUARD_AUTH_JWT_SECRET",
            "NEURALGUARD_AUTH_JWT_TTL_MINUTES",
            "NEURALGUARD_AUTH_ADMIN_TENANT",
            "NEURALGUARD_AUTH_KEYS_FILE",
        ):
            assert key in known, f"{key} missing from known_env_keys()"

    def test_jwt_manager_rejects_garbage(self) -> None:
        manager = JwtManager(AuthSettings(enabled=True, jwt_enabled=True, jwt_secret=SECRET))
        assert manager.verify("not-a-token") is None
        assert manager.verify("") is None
