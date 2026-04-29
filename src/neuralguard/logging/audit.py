"""NeuralGuard audit logger — structured logging and JSONL/PostgreSQL persistence.

Every evaluation request produces an AuditEvent that is:
1. Written to structured logs (structlog)
2. Persisted to JSONL files (default) or PostgreSQL (optional)
3. Retained per configurable retention policy

PII tokenization: When audit.tokenize_pii is enabled, user input
is replaced with a SHA-256 hash prefix in the audit trail.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from neuralguard.models.schemas import (
    AuditEvent,
    EvaluateRequest,
    EvaluateResponse,
    LayerArbitrationResult,
    ScanOutputRequest,
    ScanOutputResponse,
)

if TYPE_CHECKING:
    from neuralguard.config.settings import AuditSettings

logger = structlog.get_logger(__name__)

# Maximum JSONL file size before rotation (100 MB)
_MAX_JSONL_BYTES = 100 * 1024 * 1024


def tokenize_value(value: str) -> str:
    """Replace a value with its SHA-256 hash prefix for PII-safe logging."""
    return f"TOK:{hashlib.sha256(value.encode()).hexdigest()[:16]}"


class AuditLogger:
    """Structured audit logger for NeuralGuard.

    Supports:
    - JSONL file output (default) with daily + size-based rotation
    - PostgreSQL backend (async insert via SQLAlchemy)
    - PII tokenization
    - Configurable retention with automatic cleanup
    """

    def __init__(self, settings: AuditSettings) -> None:
        self.settings = settings
        self._jsonl_path = Path(settings.jsonl_path)
        self._jsonl_path.mkdir(parents=True, exist_ok=True)
        self._pg_available: bool = False
        self._pg_checked: bool = False

    # ── Public API ────────────────────────────────────────────────────────

    def log_evaluation(
        self,
        request: EvaluateRequest,
        response: EvaluateResponse,
        arbitration: LayerArbitrationResult,
    ) -> AuditEvent:
        """Create and persist an audit event for an evaluation."""
        event = AuditEvent(
            request_id=response.request_id,
            tenant_id=response.tenant_id,
            verdict=response.verdict,
            findings_count=len(response.findings),
            threat_categories=[f.category for f in response.findings],
            confidence=response.confidence,
            total_latency_ms=response.total_latency_ms,
            scanner_details=[
                {
                    "layer": r.layer.value,
                    "verdict": r.verdict.value,
                    "findings": len(r.findings),
                    "latency_ms": round(r.latency_ms, 2),
                    "error": r.error,
                }
                for r in arbitration.scanner_results
            ],
            metadata=self._tokenize_metadata(request.metadata)
            if self.settings.tokenize_pii
            else request.metadata,
        )

        self._persist(event)
        return event

    def log_output_scan(
        self,
        request: ScanOutputRequest,
        response: ScanOutputResponse,
    ) -> AuditEvent:
        """Create and persist an audit event for an output scan."""
        event = AuditEvent(
            request_id=response.request_id,
            tenant_id=response.tenant_id,
            verdict=response.verdict,
            findings_count=len(response.findings),
            threat_categories=[f.category for f in response.findings],
            confidence=max((f.confidence for f in response.findings), default=0.0),
            total_latency_ms=response.total_latency_ms,
            metadata={"canary_leaked": response.canary_leaked},
        )

        self._persist(event)
        return event

    # ── Persistence ───────────────────────────────────────────────────────

    def _persist(self, event: AuditEvent) -> None:
        """Write audit event to configured backend."""
        if not self.settings.enabled:
            return

        try:
            if self.settings.backend == "postgres":
                if self.settings.postgres_url:
                    self._persist_postgres(event)
                else:
                    logger.warning(
                        "postgres_url_missing",
                        msg="No postgres_url configured; falling back to JSONL",
                        event_id=event.event_id,
                    )
                    self._write_jsonl(event)
            elif self.settings.backend == "jsonl":
                self._write_jsonl(event)
        except Exception as exc:
            logger.error("audit_persist_failed", error=str(exc), event_id=event.event_id)

    def _persist_postgres(self, event: AuditEvent) -> None:
        """Insert audit event into PostgreSQL via SQLAlchemy async session.

        If the database engine is not initialized (db extras not installed,
        or engine not yet started), falls back to JSONL with a warning.
        """
        try:
            from neuralguard.db.engine import get_engine
            from neuralguard.db.models import AuditEventORM

            engine = get_engine()
            if engine is None:
                if not self._pg_checked:
                    logger.warning(
                        "postgres_engine_not_initialized",
                        msg="DB engine not ready; falling back to JSONL",
                        event_id=event.event_id,
                    )
                    self._pg_checked = True
                self._write_jsonl(event)
                return

            orm_obj = AuditEventORM(
                event_id=uuid.UUID(event.event_id)
                if isinstance(event.event_id, str)
                else event.event_id,
                request_id=uuid.UUID(event.request_id)
                if isinstance(event.request_id, str)
                else event.request_id,
                tenant_id=event.tenant_id,
                timestamp=event.timestamp,
                verdict=event.verdict.value,
                findings_count=event.findings_count,
                confidence=event.confidence,
                total_latency_ms=event.total_latency_ms,
                threat_categories=[
                    c.value if hasattr(c, "value") else c for c in event.threat_categories
                ],
                scanner_details=event.scanner_details or None,
                metadata_=event.metadata or None,
            )

            # Schedule the async insert — fire-and-forget with error logging
            asyncio.ensure_future(self._async_insert(orm_obj))
            self._pg_available = True
            self._pg_checked = True

        except ImportError:
            if not self._pg_checked:
                logger.warning(
                    "postgres_deps_missing",
                    msg="asyncpg/sqlalchemy not installed; install with [db] extra. Falling back to JSONL",
                    event_id=event.event_id,
                )
                self._pg_checked = True
            self._write_jsonl(event)
        except Exception as exc:
            logger.error(
                "postgres_insert_failed",
                error=str(exc),
                event_id=event.event_id,
            )
            self._write_jsonl(event)

    @staticmethod
    async def _async_insert(orm_obj: Any) -> None:
        """Perform the actual async database insert.

        Separate method for testability and clean error handling.
        """
        try:
            from neuralguard.db.session import session_factory

            factory = session_factory()
            async with factory() as session:
                session.add(orm_obj)
                await session.commit()
        except Exception as exc:
            import structlog as _slog

            _slog.get_logger(__name__).error(
                "postgres_async_insert_error",
                error=str(exc),
                event_id=str(orm_obj.event_id),
            )

    # ── JSONL Backend ──────────────────────────────────────────────────────

    def _write_jsonl(self, event: AuditEvent) -> None:
        """Append audit event to today's JSONL file with rotation.

        Rotation rules:
        - Daily: new file per UTC date (audit-YYYY-MM-DD.jsonl)
        - Size-based: if file exceeds 100MB, rotate to audit-YYYY-MM-DD-N.jsonl
        - Retention: delete files older than retention_days on each write
        """
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        filepath = self._jsonl_path / f"audit-{today}.jsonl"

        # Size-based rotation: if current file exceeds limit, rotate to numbered file
        if filepath.exists() and filepath.stat().st_size >= _MAX_JSONL_BYTES:
            rotation_num = 1
            rotated = self._jsonl_path / f"audit-{today}-{rotation_num}.jsonl"
            while rotated.exists():
                rotation_num += 1
                rotated = self._jsonl_path / f"audit-{today}-{rotation_num}.jsonl"
            filepath.rename(rotated)
            logger.info(
                "jsonl_rotated",
                old=filepath.name,
                new=rotated.name,
                size_bytes=rotated.stat().st_size,
            )
            # Fresh file for today
            filepath = self._jsonl_path / f"audit-{today}.jsonl"

        # Append the event
        line = event.model_dump_json() + "\n"
        with filepath.open("a", encoding="utf-8") as f:
            f.write(line)

        # Periodic retention cleanup (every 100 writes to avoid overhead)
        self._cleanup_retention()

    def _cleanup_retention(self) -> None:
        """Delete JSONL files older than retention_days.

        Uses file modification time for age calculation.
        Called periodically from _write_jsonl to avoid per-write overhead.
        """
        if self.settings.retention_days <= 0:
            return  # 0 = keep forever

        cutoff = datetime.now(UTC).timestamp() - (self.settings.retention_days * 86400)

        for f in self._jsonl_path.glob("audit-*.jsonl"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    logger.info("jsonl_retention_cleanup", deleted=f.name)
            except OSError:
                pass  # File may have been rotated/deleted concurrently

    # ── PII Tokenization ──────────────────────────────────────────────────

    def _tokenize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Tokenize string values in metadata for PII protection."""
        tokenized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, str) and len(value) > 10:
                tokenized[key] = tokenize_value(value)
            else:
                tokenized[key] = value
        return tokenized
