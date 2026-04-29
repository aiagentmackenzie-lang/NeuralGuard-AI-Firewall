"""Tests for FastAPI app factory and lifespan (covers main.py branches)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import NeuralGuardConfig
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
            version="0.2.0",
            environment="development",
        )
        app = create_app(config)
        assert app.title == "TestGuard"
        assert app.state.config.version == "0.2.0"

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
