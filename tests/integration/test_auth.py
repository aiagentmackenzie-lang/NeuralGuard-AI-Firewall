"""Integration tests for API-key authentication and tenant binding (Phase A)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import AuthSettings, NeuralGuardConfig
from neuralguard.main import create_app


def _auth_config(api_keys: list[str], enforce: bool = True) -> NeuralGuardConfig:
    return NeuralGuardConfig(
        environment="development",
        auth=AuthSettings(enabled=True, api_keys=api_keys, enforce_tenant_from_key=enforce),
    )


@pytest.fixture
async def auth_client():
    config = _auth_config(["secret-key-1|acme", "plain-key"])
    app = create_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def noauth_client():
    config = NeuralGuardConfig(environment="development")  # auth disabled by default
    app = create_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_missing_key_rejected_401(self, auth_client):
        r = await auth_client.post("/v1/evaluate", json={"prompt": "hello", "tenant_id": "acme"})
        assert r.status_code == 401
        assert r.json()["error"] == "unauthorized"

    @pytest.mark.asyncio
    async def test_invalid_key_rejected_401(self, auth_client):
        r = await auth_client.post(
            "/v1/evaluate",
            json={"prompt": "hello", "tenant_id": "acme"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_bearer_key_accepted(self, auth_client):
        r = await auth_client.post(
            "/v1/evaluate",
            json={"prompt": "What is the weather?", "tenant_id": "acme"},
            headers={"Authorization": "Bearer secret-key-1"},
        )
        assert r.status_code == 200
        assert r.json()["verdict"] == "allow"

    @pytest.mark.asyncio
    async def test_x_api_key_header_accepted(self, auth_client):
        r = await auth_client.post(
            "/v1/evaluate",
            json={"prompt": "What is the weather?", "tenant_id": "default"},
            headers={"X-API-Key": "plain-key"},
        )
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_tenant_mismatch_rejected_403(self, auth_client):
        # key 'secret-key-1' is bound to tenant 'acme'; requesting 'other' must 403
        r = await auth_client.post(
            "/v1/evaluate",
            json={"prompt": "hello", "tenant_id": "other"},
            headers={"Authorization": "Bearer secret-key-1"},
        )
        assert r.status_code == 403
        assert r.json()["detail"]["error"] == "tenant_mismatch"

    @pytest.mark.asyncio
    async def test_health_public_without_key(self, auth_client):
        r = await auth_client.get("/v1/health")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_disabled_allows_all(self, noauth_client):
        r = await noauth_client.post(
            "/v1/evaluate", json={"prompt": "hello", "tenant_id": "default"}
        )
        assert r.status_code == 200


class TestRateLimitKeying:
    """Rate limiter must key on the authenticated tenant, not a spoofed header."""

    @pytest.mark.asyncio
    async def test_spoofed_tenant_header_does_not_bypass(self):
        # Enable a very low rate limit and auth. A client supplying a different
        # X-Tenant-ID header on every request must still be limited by the
        # authenticated tenant ('acme').
        from neuralguard.config.settings import RateLimitSettings

        config = NeuralGuardConfig(
            environment="development",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            rate_limit=RateLimitSettings(enabled=True, requests_per_minute=2, burst_size=0),
        )
        app = create_app(config)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            headers = {"Authorization": "Bearer k"}
            # First two requests: allowed (limit=2)
            for i in range(2):
                r = await c.post(
                    "/v1/evaluate",
                    json={"prompt": "hi", "tenant_id": "acme"},
                    headers={**headers, "X-Tenant-ID": f"spoof-{i}"},
                )
                assert r.status_code == 200, f"req {i}: {r.status_code} {r.text}"
            # Third request from same auth tenant (different spoofed header) -> 429
            r = await c.post(
                "/v1/evaluate",
                json={"prompt": "hi", "tenant_id": "acme"},
                headers={**headers, "X-Tenant-ID": "spoof-999"},
            )
            assert r.status_code == 429
