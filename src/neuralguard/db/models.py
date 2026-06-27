"""NeuralGuard ORM models — SQLAlchemy declarative mapping for audit events.

Maps the Pydantic AuditEvent to a PostgreSQL table with proper indexing
for common query patterns (tenant_id, timestamp, verdict).
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import TIMESTAMP, Float, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all NeuralGuard ORM models."""

    pass


class AuditEventORM(Base):
    """Persistent audit event record — one row per evaluation/output scan.

    Indexed for common SOC queries:
    - By tenant (who triggered the event?)
    - By time range (what happened in the last hour?)
    - By verdict (how many blocks today?)
    """

    __tablename__ = "audit_events"

    event_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), index=True, default=lambda: datetime.now(UTC)
    )
    verdict: Mapped[str] = mapped_column(String(32), index=True)
    findings_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    threat_categories: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    scanner_details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, nullable=True)
    # Tamper-evidence chain (P1-4). Per-worker chain; see logging.chain.
    worker_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    prev_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    __table_args__ = (Index("ix_audit_tenant_timestamp", "tenant_id", "timestamp"),)

    def __repr__(self) -> str:
        return f"<AuditEventORM id={self.event_id} tenant={self.tenant_id} verdict={self.verdict}>"
