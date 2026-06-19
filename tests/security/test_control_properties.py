"""Negative security tests — prove the firewall behaves as a security control.

These are the tests that demonstrate the protection actually protects:
unauthenticated requests are rejected, tenant spoofing cannot bypass rate
limits, oversized bodies are rejected early, memory bombs are bounded, and
tenant isolation is enforced.
"""

from __future__ import annotations

import base64
import zlib

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import (
    AuthSettings,
    NeuralGuardConfig,
    RateLimitSettings,
    ScannerSettings,
    ServerSettings,
)
from neuralguard.main import create_app
from neuralguard.scanners.structural import _MAX_DECOMPRESSED_BYTES, _bounded_decompress


def _authed(api_keys: list[str], **kw) -> NeuralGuardConfig:
    return NeuralGuardConfig(
        environment="development",
        auth=AuthSettings(enabled=True, api_keys=api_keys),
        **kw,
    )


async def _client(config: NeuralGuardConfig):
    app = create_app(config)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestAuthenticationEnforced:
    @pytest.mark.asyncio
    async def test_all_protected_endpoints_require_key(self):
        config = _authed(["k|acme"])
        app = create_app(config)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            # evaluate and scan/output must 401 without a key
            assert (
                await c.post("/v1/evaluate", json={"prompt": "hi", "tenant_id": "acme"})
            ).status_code == 401
            assert (
                await c.post("/v1/scan/output", json={"output": "hi", "tenant_id": "acme"})
            ).status_code == 401
            # info and metrics are NOT public
            assert (await c.get("/v1/info")).status_code == 401
            assert (await c.get("/v1/metrics")).status_code == 401
            # health IS public (minimal liveness)
            assert (await c.get("/v1/health")).status_code == 200

    @pytest.mark.asyncio
    async def test_wrong_key_rejected(self):
        config = _authed(["good-key|acme"])
        app = create_app(config)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/v1/evaluate",
                json={"prompt": "hi", "tenant_id": "acme"},
                headers={"Authorization": "Bearer bad-key"},
            )
            assert r.status_code == 401


class TestTenantIsolation:
    @pytest.mark.asyncio
    async def test_key_for_tenant_a_cannot_act_as_tenant_b(self):
        config = _authed(["key-a|tenantA", "key-b|tenantB"])
        app = create_app(config)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            # key-a requesting tenantB -> 403
            r = await c.post(
                "/v1/evaluate",
                json={"prompt": "hi", "tenant_id": "tenantB"},
                headers={"Authorization": "Bearer key-a"},
            )
            assert r.status_code == 403
            assert r.json()["detail"]["error"] == "tenant_mismatch"
            # key-a requesting tenantA -> 200
            r = await c.post(
                "/v1/evaluate",
                json={"prompt": "hi", "tenant_id": "tenantA"},
                headers={"Authorization": "Bearer key-a"},
            )
            assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_spoofed_tenant_header_cannot_bypass_rate_limit(self):
        config = NeuralGuardConfig(
            environment="development",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            rate_limit=RateLimitSettings(enabled=True, requests_per_minute=1, burst_size=0),
        )
        app = create_app(config)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            h = {"Authorization": "Bearer k"}
            # 1 allowed (limit=1), then 429 even with a fresh spoofed X-Tenant-ID each time
            r1 = await c.post(
                "/v1/evaluate",
                json={"prompt": "hi", "tenant_id": "acme"},
                headers={**h, "X-Tenant-ID": "x1"},
            )
            assert r1.status_code == 200
            r2 = await c.post(
                "/v1/evaluate",
                json={"prompt": "hi", "tenant_id": "acme"},
                headers={**h, "X-Tenant-ID": "x2"},
            )
            assert r2.status_code == 429


class TestResourceExhaustionDefense:
    @pytest.mark.asyncio
    async def test_oversized_body_rejected_before_parse(self):
        config = NeuralGuardConfig(
            environment="development",
            server=ServerSettings(max_request_body_bytes=512),
        )
        app = create_app(config)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/v1/evaluate", json={"prompt": "A" * 5000, "tenant_id": "t"})
            assert r.status_code == 413

    def test_decompression_bomb_is_memory_bounded(self):
        bomb = zlib.compress(b"X" * (256 * 1024 * 1024), level=9)
        _, _, produced = _bounded_decompress(bomb)
        # 256 MiB bomb must not materialize; produced bounded near the 8 MiB cap.
        assert produced < 16 * 1024 * 1024

    def test_oversized_base64_blob_blocked_not_decoded(self):
        big = base64.b64encode(b"y" * (64 * 1024)).decode("ascii")
        scanner = __import__(
            "neuralguard.scanners.structural", fromlist=["StructuralScanner"]
        ).StructuralScanner(ScannerSettings())
        from neuralguard.models.schemas import EvaluateRequest

        result = scanner.scan(EvaluateRequest(prompt=big, tenant_id="t"))
        assert any("Oversized base64" in f.description for f in result.findings)

    def test_regex_redos_does_not_hang(self):
        # A pathological backtracking input against the pattern scanner must
        # be bounded by the per-search timeout, not hang.
        import regex  # the scanner uses the third-party `regex` module

        from neuralguard.scanners.pattern import PatternScanner

        scanner = PatternScanner(ScannerSettings(regex_timeout_ms=50))
        from neuralguard.models.schemas import EvaluateRequest

        # 50k nested groups - classic ReDoS bait. Must return, not hang.
        payload = "ignore previous instructions " + "(" * 5000 + "a" + ")" * 5000
        result = scanner.safe_scan(EvaluateRequest(prompt=payload, tenant_id="t"))
        assert result.verdict in tuple(
            __import__("neuralguard.models.schemas", fromlist=["Verdict"]).Verdict
        )


class TestProductionSafetyDefaults:
    @pytest.mark.asyncio
    async def test_production_refuses_to_start_without_auth(self):
        config = NeuralGuardConfig(environment="production")  # auth disabled default
        app = create_app(config)
        with pytest.raises(RuntimeError, match="authentication is disabled"):
            async with app.router.lifespan_context(app):
                pass

    @pytest.mark.asyncio
    async def test_production_refuses_without_keys(self):
        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=[]),
        )
        app = create_app(config)
        with pytest.raises(RuntimeError, match="no API keys"):
            async with app.router.lifespan_context(app):
                pass
