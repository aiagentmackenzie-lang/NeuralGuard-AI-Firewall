"""Tests for the /v1/ready readiness probe (P1-3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.api.readiness import check_readiness
from neuralguard.config.settings import (
    AuditSettings,
    NeuralGuardConfig,
    RateLimitSettings,
    ScannerSettings,
)
from neuralguard.main import create_app
from neuralguard.models.schemas import ScanLayer
from neuralguard.scanners.pattern import PatternScanner
from neuralguard.scanners.pipeline import ScannerPipeline
from neuralguard.scanners.structural import StructuralScanner


def _pipeline_with_core() -> ScannerPipeline:
    cfg = NeuralGuardConfig()
    p = ScannerPipeline(cfg)
    p.register_scanner(StructuralScanner(cfg.scanner))
    p.register_scanner(PatternScanner(cfg.scanner))
    return p


def _request(config: NeuralGuardConfig, pipeline: ScannerPipeline, app_state_extra=None):
    state = SimpleNamespace(
        config=config,
        pipeline=pipeline,
        start_time=0.0,
    )
    if app_state_extra:
        for k, v in app_state_extra.items():
            setattr(state, k, v)
    return SimpleNamespace(app=SimpleNamespace(state=state))


class TestReadinessAggregation:
    async def test_healthy_dev_config(self):
        cfg = NeuralGuardConfig()  # jsonl audit, memory ratelimit, no semantic/judge
        req = _request(cfg, _pipeline_with_core())
        r = await check_readiness(req)
        assert r.ready is True
        assert r.status == "healthy"
        assert r.components["structural"] == "ok"
        assert r.components["pattern"] == "ok"
        assert r.components["semantic"] == "skip"
        assert r.components["judge"] == "skip"
        assert r.components["audit_db"] == "skip"
        assert r.components["redis"] == "skip"

    async def test_degraded_when_semantic_uninitialized(self):
        cfg = NeuralGuardConfig(
            scanner=ScannerSettings(semantic_enabled=True),
        )
        p = ScannerPipeline(cfg)
        p.register_scanner(StructuralScanner(cfg.scanner))
        p.register_scanner(PatternScanner(cfg.scanner))
        # semantic scanner NOT registered -> "fail" (optional) => degraded
        req = _request(cfg, p)
        r = await check_readiness(req)
        assert r.ready is True  # core ok -> still serves
        assert r.status == "degraded"
        assert r.components["semantic"] == "fail"

    async def test_unhealthy_when_core_scanners_missing(self):
        cfg = NeuralGuardConfig()
        p = ScannerPipeline(cfg)  # no scanners registered
        req = _request(cfg, p)
        r = await check_readiness(req)
        assert r.ready is False
        assert r.status == "unhealthy"
        assert r.components["structural"] == "fail"

    async def test_unhealthy_when_required_redis_down(self):
        cfg = NeuralGuardConfig(
            rate_limit=RateLimitSettings(backend="redis", redis_url="redis://localhost:6379/0"),
        )

        class _BoomClient:
            async def ping(self) -> None:
                raise ConnectionError("redis down")

        class BoomLimiter:
            client = _BoomClient()

        req = _request(cfg, _pipeline_with_core(), {"redis_limiter": BoomLimiter()})
        r = await check_readiness(req)
        assert r.ready is False
        assert r.status == "unhealthy"
        assert r.components["redis"] == "fail"

    async def test_degraded_when_redis_configured_but_not_initialized(self):
        cfg = NeuralGuardConfig(
            rate_limit=RateLimitSettings(backend="redis", redis_url="redis://localhost:6379/0"),
        )
        # No redis_limiter on app.state -> configured but not wired.
        req = _request(cfg, _pipeline_with_core())
        r = await check_readiness(req)
        assert r.ready is False
        assert r.components["redis"] == "fail"

    async def test_audit_db_degraded_when_postgres_engine_not_ready(self):
        cfg = NeuralGuardConfig(
            audit=AuditSettings(backend="postgres", postgres_url="postgresql+asyncpg://x"),
        )
        req = _request(cfg, _pipeline_with_core())
        r = await check_readiness(req)
        # engine not created in lifespan -> degraded (JSONL fallback), still ready
        assert r.ready is True
        assert r.status == "degraded"
        assert r.components["audit_db"] == "degraded"


class TestReadinessEndpoint:
    async def test_ready_endpoint_returns_200_dev(self):
        app = create_app(NeuralGuardConfig(environment="development"))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/v1/ready")
            assert r.status_code == 200
            body = r.json()
            assert body["ready"] is True
            assert body["status"] in ("healthy", "degraded")
            assert "components" in body

    async def test_ready_endpoint_auth_gated_in_production(self):
        from neuralguard.config.settings import AuthSettings, ServerSettings

        cfg = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True),
        )
        app = create_app(cfg)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            # No key -> 401 (not in public_endpoints by default).
            r = await c.get("/v1/ready")
            assert r.status_code == 401
            # With key -> 200.
            r2 = await c.get("/v1/ready", headers={"X-API-Key": "k"})
            assert r2.status_code == 200

    async def test_ready_endpoint_public_when_listed_in_public_endpoints(self):
        from neuralguard.config.settings import AuthSettings, ServerSettings

        cfg = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(
                enabled=True, api_keys=["k|acme"], public_endpoints=["/v1/health", "/v1/ready"]
            ),
            server=ServerSettings(allow_insecure_http=True),
        )
        app = create_app(cfg)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/v1/ready")
            assert r.status_code == 200
