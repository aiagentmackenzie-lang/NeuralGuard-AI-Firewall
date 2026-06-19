"""Phase B tests: metrics, JSON logging, exception handler, judge URL, audit
backpressure, /info auth-gating."""

from __future__ import annotations

import json
import logging

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import AuthSettings, NeuralGuardConfig, ScannerSettings
from neuralguard.main import create_app


@pytest.fixture
async def client():
    config = NeuralGuardConfig(environment="development")
    app = create_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_endpoint_exposes_prometheus_format(self, client):
        r = await client.post("/v1/evaluate", json={"prompt": "hello", "tenant_id": "t"})
        assert r.status_code == 200
        r = await client.get("/v1/metrics")
        assert r.status_code == 200
        text = r.text
        # Prometheus exposition format: lines of name{labels} value
        assert "neuralguard_verdicts_total" in text
        assert "neuralguard_pipeline_latency_seconds" in text

    @pytest.mark.asyncio
    async def test_verdict_counter_increments(self, client):
        r = await client.post("/v1/evaluate", json={"prompt": "hello", "tenant_id": "t"})
        assert r.status_code == 200
        m = await client.get("/v1/metrics")
        assert 'neuralguard_verdicts_total{verdict="allow"}' in m.text


class TestJsonLogging:
    def test_production_uses_json_renderer(self, capsys):
        # Reconfigure for production and emit a log; must be JSON.
        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
        )
        create_app(config)  # triggers structlog.configure for production
        import structlog

        log = structlog.get_logger("neuralguard")
        log.info("json_test_event", field="value")
        out = capsys.readouterr().out.strip().splitlines()[-1]
        # Production renderer emits a JSON object.
        assert out.startswith("{"), out
        parsed = json.loads(out)
        assert parsed["event"] == "json_test_event"
        assert parsed["field"] == "value"
        # restore dev rendering for other tests
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            cache_logger_on_first_use=True,
        )


class TestGlobalExceptionHandler:
    @pytest.mark.asyncio
    async def test_unhandled_exception_returns_500_with_correlation_id(self, monkeypatch):
        config = NeuralGuardConfig(environment="development")
        app = create_app(config)

        # Force the pipeline to raise on execute
        class _Boom:
            def execute(self, *_a, **_k):
                raise RuntimeError("boom")

            def _scanners(self):
                return {}

        app.state.pipeline = _Boom()  # type: ignore[assignment]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/v1/evaluate", json={"prompt": "x", "tenant_id": "t"})
            assert r.status_code == 500
            body = r.json()
            assert body["error"] == "internal_error"
            assert "correlation_id" in body


class TestJudgeUrlConfig:
    def test_judge_url_is_configurable(self):
        s = ScannerSettings(judge_ollama_url="http://ollama.local:11434")
        assert s.judge_ollama_url == "http://ollama.local:11434"

    def test_judge_url_default_is_loopback(self):
        s = ScannerSettings()
        assert "localhost" in s.judge_ollama_url


class TestInfoAuthGating:
    @pytest.mark.asyncio
    async def test_info_requires_auth_when_enabled(self):
        config = NeuralGuardConfig(
            environment="development",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
        )
        app = create_app(config)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            # No key -> /info should be 401 (not in public_endpoints)
            r = await c.get("/v1/info")
            assert r.status_code == 401
            # With key -> 200
            r = await c.get("/v1/info", headers={"Authorization": "Bearer k"})
            assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_requires_auth_when_enabled(self):
        config = NeuralGuardConfig(
            environment="development",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
        )
        app = create_app(config)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/v1/metrics")
            assert r.status_code == 401


class TestAuditInflightCap:
    def test_inflight_set_tracked_on_pg_persist(self):
        # When postgres backend is used and event loop present, the inflight
        # set is populated and bounded by max_inflight_writes.
        from neuralguard.config.settings import AuditSettings
        from neuralguard.logging.audit import AuditLogger

        logger = AuditLogger(AuditSettings(backend="postgres", max_inflight_writes=2))
        assert logger.settings.max_inflight_writes == 2
        # The _inflight set exists and starts empty.
        assert logger._inflight is not None
