"""Tests for FastAPI app factory and lifespan (covers main.py branches)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import (
    AuthSettings,
    CanarySettings,
    NeuralGuardConfig,
    ServerSettings,
)
from neuralguard.main import create_app


class TestCreateApp:
    """Test app factory with various configs."""

    def test_create_app_default_config(self):
        app = create_app()
        assert app.title == "NeuralGuard"
        assert app.state.config is not None
        assert app.state.pipeline is not None
        assert app.state.audit_logger is not None

    def test_create_app_custom_config(self):
        config = NeuralGuardConfig(
            app_name="TestGuard",
            version="0.2.1",
            environment="development",
        )
        app = create_app(config)
        assert app.title == "TestGuard"
        assert app.state.config.version == "0.2.1"

    def test_create_app_production_hides_docs(self):
        config = NeuralGuardConfig(environment="production")
        app = create_app(config)
        assert app.docs_url is None
        assert app.redoc_url is None

    def test_create_app_development_shows_docs(self):
        config = NeuralGuardConfig(environment="development")
        app = create_app(config)
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    def test_create_app_has_routes(self):
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/v1/evaluate" in routes
        assert "/v1/scan/output" in routes
        assert "/v1/health" in routes
        assert "/v1/info" in routes

    def test_create_app_has_middleware(self):
        app = create_app()
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "RateLimitMiddleware" in middleware_classes


class TestLifespan:
    """Test FastAPI lifespan startup/shutdown logic."""

    @pytest.mark.asyncio
    async def test_lifespan_jsonl_backend(self):
        """App with jsonl backend should start fine (no postgres init)."""
        config = NeuralGuardConfig(audit={"backend": "jsonl"})
        app = create_app(config)
        # Directly invoke the lifespan context manager
        async with app.router.lifespan_context(app):
            pass

    @pytest.mark.asyncio
    async def test_lifespan_postgres_no_url(self):
        """App with postgres backend but no URL should start fine (JSONL fallback)."""
        config = NeuralGuardConfig(audit={"backend": "postgres", "postgres_url": None})
        app = create_app(config)
        async with app.router.lifespan_context(app):
            pass

    @pytest.mark.asyncio
    async def test_lifespan_postgres_with_url(self):
        """App with postgres backend and URL should initialize DB engine."""
        config = NeuralGuardConfig(
            audit={
                "backend": "postgres",
                "postgres_url": "postgresql+asyncpg://user:pass@localhost:5432/testdb",
            }
        )
        app = create_app(config)
        async with app.router.lifespan_context(app):
            pass
        # Cleanup
        from neuralguard.db.engine import dispose_engine

        await dispose_engine()

    @pytest.mark.asyncio
    async def test_lifespan_postgres_init_error_fallback(self):
        """App with postgres backend where init fails should fall back gracefully."""
        config = NeuralGuardConfig(
            audit={
                "backend": "postgres",
                "postgres_url": "postgresql+asyncpg://invalid:invalid@nonexistent:5432/nodb",
            }
        )
        app = create_app(config)
        # Should not crash — lifespan catches the connection exception
        async with app.router.lifespan_context(app):
            pass

    @pytest.mark.asyncio
    async def test_health_endpoint_via_client(self):
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert "version" in data

    @pytest.mark.asyncio
    async def test_info_endpoint_via_client(self):
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/info")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "NeuralGuard"
            assert "owasp_coverage" in data


# ── Canary production fail-fast (Phase 3, B3) ──────────────────────────────


class TestCanaryProductionGates:
    """Production must refuse to start when canary is enabled but the secret is
    missing or too short (a weak/empty secret makes the canary guessable)."""

    async def test_canary_enabled_no_secret_refused(self):
        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True, workers=1),
            canary=CanarySettings(enabled=True, secret=""),
        )
        app = create_app(config)
        with pytest.raises(RuntimeError, match=r"canary\.enabled=true"):
            async with app.router.lifespan_context(app):
                pass

    async def test_canary_short_secret_refused(self):
        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True, workers=1),
            canary=CanarySettings(enabled=True, secret="short"),
        )
        app = create_app(config)
        with pytest.raises(RuntimeError, match=r"shorter than 32"):
            async with app.router.lifespan_context(app):
                pass

    async def test_canary_strong_secret_allowed(self):
        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True, workers=1),
            canary=CanarySettings(enabled=True, secret="x" * 40),
        )
        app = create_app(config)
        async with app.router.lifespan_context(app):
            pass  # should not raise
        # The manager is constructed on app state when enabled.
        assert app.state.canary_manager is not None
        assert app.state.canary_manager.enabled is True

    def test_canary_disabled_no_manager_on_state(self):
        config = NeuralGuardConfig(environment="development")
        app = create_app(config)
        assert app.state.canary_manager is None


# ── SIEM production fail-fast (P2-7 / F23) ─────────────────────────────


class TestSiemProductionGates:
    """F23: the enabled-without-sink gate must know EVERY sink the router
    supports. A scarletai-only deployment (the local-SIEM posture) was
    previously refused in production and silently unrouted in dev because the
    gate checked only splunk/webhook."""

    @staticmethod
    def _base_config(**siem_kwargs) -> NeuralGuardConfig:
        from neuralguard.config.settings import SiemSettings

        return NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True, workers=1),
            siem=SiemSettings(enabled=True, **siem_kwargs),
        )

    async def test_scarletai_only_production_boots_and_routes(self):
        """The F23 regression: scarletai-only config is a VALID sink config."""
        from neuralguard.siem import SiemRouter

        config = self._base_config(
            scarletai_url="http://127.0.0.1:8000/api/v1/ingest",
            scarletai_token="tok",
        )
        app = create_app(config)
        async with app.router.lifespan_context(app):
            siem = app.state.audit_logger._siem
        assert isinstance(siem, SiemRouter)
        assert siem._sinks == ["scarletai"]

    def test_siem_enabled_no_sink_production_refused(self):
        # The gate fires in create_app (SIEM router construction), not lifespan.
        config = self._base_config()
        with pytest.raises(RuntimeError, match="no sink is configured"):
            create_app(config)

    async def test_siem_enabled_no_sink_dev_boots_unrouted(self):
        """Dev warns and boots; the audit logger must have NO router (the
        silent-unroute trap applied when scarletai-only was misread as empty)."""
        from neuralguard.config.settings import SiemSettings

        config = NeuralGuardConfig(
            environment="development",
            siem=SiemSettings(enabled=True),
        )
        app = create_app(config)
        async with app.router.lifespan_context(app):
            assert app.state.audit_logger._siem is None

    async def test_all_three_sinks_production_boots(self):
        """Multi-sink production config: router sees all three sinks."""
        config = self._base_config(
            splunk_hec_url="https://splunk.test:8088",
            webhook_url="http://elk.test/ingest",
            scarletai_url="http://127.0.0.1:8000/api/v1/ingest",
        )
        app = create_app(config)
        async with app.router.lifespan_context(app):
            assert app.state.audit_logger._siem._sinks == [
                "splunk_hec",
                "webhook",
                "scarletai",
            ]
