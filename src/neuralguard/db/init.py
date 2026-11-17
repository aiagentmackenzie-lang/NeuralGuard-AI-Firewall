"""Postgres audit-backend initialisation with a bounded retry.

The audit store is created in the app lifespan, and in a container orchestrator
the database is frequently *not* up when the first app replica is. The original
implementation was a one-shot ``create_all`` wrapped in try/except: when the
``postgres`` Service had no endpoints yet, asyncpg raised
``[Errno -2] Name or service not known``, the error was logged once, and the
process fell back to JSONL **for its entire lifetime** — every later audit
INSERT then failed with ``relation "audit_events" does not exist`` while
``/v1/ready`` still reported ``audit_db: ok`` (found by the 2026-11-17 cluster
drill against the published v0.3.0 image; only a ``rollout restart`` recovered).

So: retry a bounded number of times with backoff, because a late dependency is
expected, and report the outcome instead of swallowing it. The JSONL fallback
still exists — it is a designed capability — but it is now a *reported* state
rather than an invisible one.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import text

from neuralguard.db.engine import create_engine as db_create_engine
from neuralguard.db.models import Base

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

logger = structlog.get_logger(__name__)

#: Total attempts (1 try + retries). Six attempts with the backoff below covers
#: roughly 1+2+4+8+8 ≈ 23 s, which is longer than a StatefulSet PVC attach and
#: shorter than any sane readiness timeout.
DEFAULT_ATTEMPTS = 6
#: Backoff ceiling — never sleep longer than this between attempts.
MAX_BACKOFF_SECONDS = 8.0


@dataclass
class InitResult:
    """Outcome of the audit-store initialisation."""

    ok: bool
    attempts: int
    engine: AsyncEngine | None = None
    error: str | None = None


async def _ensure_schema(engine: AsyncEngine) -> None:
    """Create the audit schema, idempotently, including post-release columns.

    ``create_all`` never alters an existing table, so the columns added after
    the first release (``event_sig`` from P2-10, ``global_seq`` /
    ``global_prev_hash`` / ``global_hash`` from cross-worker ordering) are added
    with idempotent DDL. Nullable by design: legacy rows stay readable and are
    reported as outside the chain.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text("ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS event_sig VARCHAR(128)")
        )
        await conn.execute(text("CREATE SEQUENCE IF NOT EXISTS audit_global_seq"))
        await conn.execute(
            text("ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS global_seq BIGINT")
        )
        await conn.execute(
            text("ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS global_prev_hash VARCHAR(64)")
        )
        await conn.execute(
            text("ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS global_hash VARCHAR(64)")
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_audit_events_global_seq ON audit_events (global_seq)"
            )
        )


async def initialise_postgres(
    postgres_url: str,
    *,
    engine_factory: Callable[[str], Any] = db_create_engine,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = 1.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> InitResult:
    """Create the engine and the audit schema, retrying a late dependency.

    Args:
        postgres_url: SQLAlchemy async URL for the audit database.
        engine_factory: Injectable engine factory (tests, or a custom pool).
        attempts: Total tries, including the first.
        base_delay: First backoff, doubled each retry up to
            :data:`MAX_BACKOFF_SECONDS`.
        sleep: Injectable sleep (tests run without waiting).

    Returns:
        :class:`InitResult` — never raises. A failure keeps the JSONL fallback
        working and carries the underlying error so readiness can report it.
    """
    delay = base_delay
    last_error: str | None = None
    engine: AsyncEngine | None = None

    for attempt in range(1, attempts + 1):
        try:
            engine = engine_factory(postgres_url)
            await _ensure_schema(engine)
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "db_init_attempt_failed",
                attempt=attempt,
                attempts=attempts,
                error=last_error,
            )
            if attempt < attempts:
                await sleep(delay)
                delay = min(delay * 2, MAX_BACKOFF_SECONDS)
            continue
        logger.info("db_tables_created", backend="postgres", attempts=attempt)
        return InitResult(ok=True, attempts=attempt, engine=engine)

    logger.error(
        "db_init_failed",
        attempts=attempts,
        error=last_error,
        msg="PostgreSQL init failed after every attempt; JSONL fallback active",
    )
    return InitResult(ok=False, attempts=attempts, engine=engine, error=last_error)
