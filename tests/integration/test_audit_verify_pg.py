"""P2-10 close-out: postgres audit-chain verification, LIVE-FIRE.

Requires a reachable Postgres. Skipped unless NEURALGUARD_TEST_PG_URL is set,
e.g.:

    docker run -d --name ng-verify-drill -e POSTGRES_PASSWORD=x \
      -p 127.0.0.1:55432:5432 postgres:17-alpine
    NEURALGUARD_TEST_PG_URL=postgresql+asyncpg://postgres:drillpg@127.0.0.1:55432/postgres \
      uv run pytest tests/integration/test_audit_verify_pg.py -v

These tests are NOT mock-based by design: an operator's tamper-evidence
guarantee must be proven against a real Postgres — insert via the REAL
AuditLogger (chain stamping + Ed25519 signing), verify with the REAL
verify_audit_postgres, then tamper rows and watch the verdict flip. Each
test gets a freshly seeded table (drop + create) so tampering never leaks
across tests.
"""

from __future__ import annotations

import asyncio
import os

import pytest
import sqlalchemy

from neuralguard.config.settings import AuditSettings
from neuralguard.db.engine import get_engine
from neuralguard.logging.audit import AuditLogger
from neuralguard.logging.verify import verify_audit_postgres
from neuralguard.models.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    LayerArbitrationResult,
    Verdict,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("NEURALGUARD_TEST_PG_URL"),
    reason="NEURALGUARD_TEST_PG_URL not set (live postgres required)",
)

PG_URL = os.environ.get("NEURALGUARD_TEST_PG_URL", "")
SIGNING_SEED = "ab" * 32  # 32-byte Ed25519 seed, hex


async def _wait_for_rows(expected: int, timeout: float = 5.0) -> int:
    """The AuditLogger inserts fire-and-forget; poll until the rows land."""
    from sqlalchemy import text

    from neuralguard.db.engine import get_engine

    engine = get_engine()
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        async with engine.connect() as conn:
            n = (await conn.execute(text("SELECT count(*) FROM audit_events"))).scalar()
        if n >= expected:
            return int(n)
        await asyncio.sleep(0.05)
    raise TimeoutError(f"expected {expected} audit rows, saw fewer")


async def _seed_table() -> None:
    """Fresh table + REAL signed events through the REAL writer path.

    The engine singleton is created here and kept ALIVE for the writer's
    fire-and-forget inserts; disposal happens in the autouse teardown AFTER
    the rows have flushed.
    """
    from sqlalchemy import text

    from neuralguard.db.engine import create_engine
    from neuralguard.db.models import Base

    create_engine(PG_URL)  # installs the engine singleton the writer uses
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
            # Same idempotent migration the lifespan runs (pre-existing tables).
            await conn.execute(
                text("ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS event_sig VARCHAR(128)")
            )
    finally:
        # Engine singleton stays alive — the fire-and-forget writer inserts
        # use it; disposal happens in the autouse teardown fixture.
        pass

    # TWO loggers = TWO worker chains, exactly like two processes.
    for worker, tenant in (("workerA", "acme"), ("workerB", "globex")):
        settings = AuditSettings(
            enabled=True,
            backend="postgres",
            postgres_url=PG_URL,
            signing_key=SIGNING_SEED,
        )
        audit = AuditLogger(settings)
        audit._worker_id = worker
        for i in range(3):
            audit.log_evaluation(
                EvaluateRequest(prompt=f"prompt {tenant} {i}", tenant_id=tenant),
                EvaluateResponse(
                    tenant_id=tenant,
                    verdict=Verdict.ALLOW,
                    confidence=0.0,
                    scan_layers_used=[],
                    total_latency_ms=1.0,
                ),
                LayerArbitrationResult(
                    verdict=Verdict.ALLOW,
                    findings=[],
                    scanner_results=[],
                    total_latency_ms=1.0,
                    arbitration_reason="clean",
                ),
            )

    await _wait_for_rows(6)
    # Give the fire-and-forget inserts a beat to fully settle.
    await asyncio.sleep(0.2)


@pytest.fixture()
async def _seeded_db():
    await _seed_table()


@pytest.fixture(autouse=True)
async def _teardown_engine():
    """Dispose the engine singleton AFTER the test (the seeded writes use it)."""
    yield
    from neuralguard.db.engine import dispose_engine

    await dispose_engine()


@pytest.fixture()
async def _seeded_db():
    await _seed_table()


async def _tamper(worker_id: str, *, event_hash: str | None = None, delete: bool = False) -> None:
    """Mutate the seeded table the way an on-disk attacker would."""
    from sqlalchemy import text

    from neuralguard.db.engine import create_engine

    engine = create_engine(PG_URL)
    try:
        async with engine.begin() as conn:
            if delete:
                # Delete a MIDDLE row (not the head): the remaining latest row
                # still references the deleted event's hash → orphan → BROKEN.
                # (Deleting non-head rows wholesale would leave a coherent
                # head-only chain — a valid shorter chain, not tamper evidence.)
                await conn.execute(
                    text(
                        "DELETE FROM audit_events WHERE event_id IN ("
                        "SELECT event_id FROM audit_events WHERE worker_id = :w "
                        "ORDER BY timestamp ASC OFFSET 1 LIMIT 1)"
                    ),
                    {"w": worker_id},
                )
            else:
                await conn.execute(
                    text(
                        "UPDATE audit_events SET event_hash = :h WHERE worker_id = :w "
                        "AND prev_hash IS NOT NULL"
                    ),
                    {"h": event_hash, "w": worker_id},
                )
    finally:
        await engine.dispose()


class TestVerifyAuditPostgresLive:
    async def test_honest_signed_table_is_valid_with_pubkey(self, _seeded_db) -> None:
        from neuralguard.logging.signing import public_key_from_seed

        report = await verify_audit_postgres(PG_URL, pubkey_hex=public_key_from_seed(SIGNING_SEED))
        assert report.parse_errors == 0
        assert report.events_parsed == 6
        assert {c.worker_id for c in report.chains} == {"workerA", "workerB"}
        assert report.all_valid, f"chains: {report.chains}"

    async def test_unsigned_table_rejected_with_pubkey(self, _seeded_db) -> None:
        """Signing-mode verification: unsigned rows are BROKEN even when the
        hash chain is internally consistent — that is the forgery signing
        exists to catch."""
        # Strip signatures, keep the honest chain.
        from sqlalchemy import text

        from neuralguard.db.engine import create_engine

        engine = create_engine(PG_URL)
        async with engine.begin() as conn:
            await conn.execute(text("UPDATE audit_events SET event_sig = NULL"))
        await engine.dispose()
        report = await verify_audit_postgres(PG_URL, pubkey_hex="cd" * 32)
        assert not report.all_valid
        assert all(not c.valid for c in report.chains)

    async def test_hash_tampering_is_broken(self, _seeded_db) -> None:
        """The core tamper-evidence promise: a modified row breaks its own
        hash AND the next event's prev_hash link."""
        await _tamper("workerA", event_hash="0" * 64)
        report = await verify_audit_postgres(PG_URL)
        by_worker = {c.worker_id: c for c in report.chains}
        assert not by_worker["workerA"].valid
        assert by_worker["workerB"].valid  # other chain unaffected

    async def test_deleted_row_orphan_is_broken(self, _seeded_db) -> None:
        """A row deleted from the middle of a chain: link-walk finds the
        orphan — a partial/suppressed table is not verifiable."""
        await _tamper("workerB", delete=True)
        report = await verify_audit_postgres(PG_URL)
        by_worker = {c.worker_id: c for c in report.chains}
        assert not by_worker["workerB"].valid

    async def test_unreachable_postgres_raises(self) -> None:
        from neuralguard.logging.verify import verify_audit_postgres as v

        with pytest.raises(OSError):
            await v("postgresql+asyncpg://postgres:drillpg@127.0.0.1:59999/nodb")
