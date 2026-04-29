"""Tests for lazy imports in db/__init__.py and session module."""

from __future__ import annotations

import pytest

try:
    from neuralguard.db.engine import create_engine, dispose_engine, get_engine
    from neuralguard.db.models import AuditEventORM, Base
    from neuralguard.db.session import get_session, session_factory

    HAS_DB_DEPS = True
except ImportError:
    HAS_DB_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DB_DEPS, reason="sqlalchemy/asyncpg not installed")


class TestDbLazyImports:
    """Test lazy import behavior from db package."""

    def test_create_engine_importable(self):
        from neuralguard.db import create_engine

        assert callable(create_engine)

    def test_audit_event_orm_importable(self):
        from neuralguard.db import AuditEventORM

        assert AuditEventORM.__tablename__ == "audit_events"

    def test_session_factory_importable(self):
        from neuralguard.db import session_factory

        assert callable(session_factory)

    def test_get_session_importable(self):
        from neuralguard.db import get_session

        assert callable(get_session)

    def test_invalid_attribute_raises(self):
        import neuralguard.db

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = neuralguard.db.nonexistent_thing


class TestSessionModule:
    """Test session factory and get_session generator."""

    @pytest.mark.asyncio
    async def test_get_session_with_engine(self):
        """get_session should yield a session when engine is active."""
        create_engine("postgresql+asyncpg://user:pass@localhost:5432/testdb")
        try:
            async for session in get_session():
                assert session is not None
                # Session is usable even without a real DB
                break  # Just verify we get one
        finally:
            await dispose_engine()

    @pytest.mark.asyncio
    async def test_get_session_rollback_on_error(self):
        """get_session should rollback on exception."""
        create_engine("postgresql+asyncpg://user:pass@localhost:5432/testdb")
        try:
            with pytest.raises(ValueError):
                async for _session in get_session():
                    raise ValueError("test error")
        finally:
            await dispose_engine()

    def test_session_factory_returns_maker(self):
        """session_factory returns an async_sessionmaker."""
        create_engine("postgresql+asyncpg://user:pass@localhost:5432/testdb")
        try:
            maker = session_factory()
            from sqlalchemy.ext.asyncio import async_sessionmaker

            assert isinstance(maker, async_sessionmaker)
        finally:
            import asyncio

            asyncio.get_event_loop().run_until_complete(dispose_engine())
