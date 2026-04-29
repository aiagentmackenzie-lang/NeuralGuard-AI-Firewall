"""Integration tests for database engine lifecycle and session management.

These tests verify the async engine creation, table creation, and session
handling that can't be tested with pure unit mocks.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

try:
    from neuralguard.db.engine import create_engine, dispose_engine, get_engine
    from neuralguard.db.models import AuditEventORM, Base
    from neuralguard.db.session import session_factory

    HAS_DB_DEPS = True
except ImportError:
    HAS_DB_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DB_DEPS, reason="sqlalchemy/asyncpg not installed")


class TestEngineLifecycle:
    """Test engine creation, retrieval, and disposal."""

    def test_create_engine_stores_globally(self):
        engine = create_engine("postgresql+asyncpg://user:pass@localhost:5432/testdb")
        assert get_engine() is engine

    def test_create_engine_second_time_replaces(self):
        create_engine("postgresql+asyncpg://user:pass@localhost:5432/db1")
        engine2 = create_engine("postgresql+asyncpg://user:pass@localhost:5432/db2")
        assert get_engine() is engine2

    @pytest.mark.asyncio
    async def test_dispose_engine_clears_global(self):
        create_engine("postgresql+asyncpg://user:pass@localhost:5432/testdb")
        await dispose_engine()
        assert get_engine() is None

    @pytest.mark.asyncio
    async def test_dispose_when_none_is_safe(self):
        import neuralguard.db.engine as eng_module

        eng_module._engine = None
        await dispose_engine()  # Should not raise
        assert get_engine() is None


class TestAuditEventORMMapping:
    """Test ORM field types and constraints."""

    def test_all_required_fields_present(self):
        """Construct ORM object with all required fields."""
        orm = AuditEventORM(
            event_id=uuid.uuid4(),
            request_id=uuid.uuid4(),
            tenant_id="acme-corp",
            timestamp=datetime.now(UTC),
            verdict="block",
            findings_count=5,
            confidence=0.95,
            total_latency_ms=12.3,
            threat_categories=["T-PI-D", "T-JB", "T-EXT"],
            scanner_details=[{"layer": "pattern", "verdict": "block", "latency_ms": 3.2}],
            metadata_={"source": "test", "ip": "10.0.0.1"},
        )
        assert orm.tenant_id == "acme-corp"
        assert orm.findings_count == 5
        assert len(orm.threat_categories) == 3
        assert orm.metadata_["source"] == "test"

    def test_nullable_jsonb_fields(self):
        """JSONB fields should accept None."""
        orm = AuditEventORM(
            event_id=uuid.uuid4(),
            request_id=uuid.uuid4(),
            tenant_id="minimal",
            timestamp=datetime.now(UTC),
            verdict="allow",
            threat_categories=None,
            scanner_details=None,
            metadata_=None,
        )
        assert orm.threat_categories is None
        assert orm.scanner_details is None
        assert orm.metadata_ is None

    def test_table_has_composite_index(self):
        """Verify the tenant+timestamp composite index for query performance."""
        idx_names = {idx.name for idx in AuditEventORM.__table__.indexes}
        assert "ix_audit_tenant_timestamp" in idx_names

    def test_table_has_all_expected_columns(self):
        expected = {
            "event_id",
            "request_id",
            "tenant_id",
            "timestamp",
            "verdict",
            "findings_count",
            "confidence",
            "total_latency_ms",
            "threat_categories",
            "scanner_details",
            "metadata",
        }
        actual = {c.name for c in AuditEventORM.__table__.columns}
        assert actual == expected

    def test_metadata_column_named_correctly(self):
        """metadata (not metadata_) is the DB column name."""
        col_names = {c.name for c in AuditEventORM.__table__.columns}
        assert "metadata" in col_names
        # Python attribute is metadata_ to avoid conflict
        assert hasattr(AuditEventORM, "metadata_")
