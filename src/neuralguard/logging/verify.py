"""Operator audit-chain verification (F14).

``verify_audit_files`` loads JSONL audit files, groups events per worker
chain (a naive single-chain verify over an interleaved multi-worker file
FAILS BY DESIGN — hash chains are per-process, P2-10 tracks cross-worker
ordering + signing), and verifies each chain with
``neuralguard.logging.chain.verify_chain``.

``verify_audit_postgres`` (P2-10 close-out) does the same for the postgres
audit backend: rows are read from the ``audit_events`` table, grouped per
worker, and each chain is reconstructed by LINK-WALK (prev_hash links —
SQL row order is not trusted as write order) before hash + signature
verification.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

from neuralguard.logging.chain import verify_chain
from neuralguard.models.schemas import AuditEvent

logger = structlog.get_logger(__name__)


@dataclass
class ChainReport:
    worker_id: str
    event_count: int
    valid: bool


@dataclass
class AuditVerifyReport:
    files_read: int
    events_parsed: int
    parse_errors: int
    chains: list[ChainReport]

    @property
    def all_valid(self) -> bool:
        return self.parse_errors == 0 and all(c.valid for c in self.chains)

    def to_dict(self) -> dict[str, object]:
        return {
            "files_read": self.files_read,
            "events_parsed": self.events_parsed,
            "parse_errors": self.parse_errors,
            "all_valid": self.all_valid,
            "chains": [
                {
                    "worker_id": c.worker_id,
                    "events": c.event_count,
                    "valid": c.valid,
                }
                for c in self.chains
            ],
        }


def _audit_files(target: Path) -> list[Path]:
    """Expand a file or directory into the audit JSONL files to verify."""
    if target.is_dir():
        return sorted(target.rglob("*.jsonl"))
    if target.suffix != ".jsonl":
        raise ValueError(f"not a .jsonl audit file: {target}")
    return [target]


def verify_audit_files(target: Path, pubkey_hex: str | None = None) -> AuditVerifyReport:
    """Load + group + verify every per-worker chain under ``target``.

    Files are read in sorted order (daily rotation names sort
    chronologically); events group by ``worker_id`` across files so a chain
    spanning a rotation boundary is still verified end-to-end.

    P2-10: when ``pubkey_hex`` is provided, EVERY event must also carry a
    valid Ed25519 signature (``event_sig``) over its ``event_hash`` — a
    chain that is hash-consistent but unsigned/signed by another key is
    reported BROKEN (a forged file would be hash-consistent too; that is
    exactly what signing exists to catch).
    """
    files = _audit_files(target)
    chains: dict[str, list[AuditEvent]] = {}
    events_parsed = 0
    parse_errors = 0

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = AuditEvent.model_validate_json(line)
                except Exception:
                    parse_errors += 1
                    logger.warning("audit_verify_parse_error", file=str(path), line=line_no)
                    continue
                events_parsed += 1
                worker = event.worker_id or "<unknown-worker>"
                chains.setdefault(worker, []).append(event)

    report = AuditVerifyReport(
        files_read=len(files),
        events_parsed=events_parsed,
        parse_errors=parse_errors,
        chains=[],
    )
    for worker_id in sorted(chains):
        events = chains[worker_id]
        valid = verify_chain(events)
        if valid and pubkey_hex is not None:
            from neuralguard.logging.signing import verify_event_signature

            valid = all(
                e.event_sig is not None
                and e.event_hash is not None
                and verify_event_signature(e.event_hash, e.event_sig, pubkey_hex)
                for e in events
            )
        report.chains.append(
            ChainReport(
                worker_id=worker_id,
                event_count=len(events),
                valid=valid,
            )
        )
    return report


# ── Postgres audit source (P2-10 close-out) ───────────────────────────────
# The postgres audit backend stores the tamper-evidence chain (worker_id /
# prev_hash / event_hash / event_sig) in the audit_events table, but until
# now only JSONL files had a verification tool — a postgres-audit operator
# could not verify their own chains. This path reads the table and verifies
# the SAME semantics (per-worker chains + optional Ed25519 signatures).


def reconstruct_chain_order(events: list[AuditEvent]) -> list[AuditEvent] | None:
    """Order-independent chain reconstruction from prev_hash links.

    SQL row order (timestamp, event_id) is only an approximation of write
    order — microsecond ties under a write burst can reorder rows, and a
    false-BROKEN report on an honest table is a real failure mode for an
    operator. So the postgres path reconstructs the chain by WALKING the
    prev_hash links: start at the chain head(s) (prev_hash None) and follow
    event_hash -> prev_hash until the walk ends.

    Returns the events in chain order, or None when the worker's events do
    not form one coherent chain (orphan rows — a gap or fork — including a
    partial backup whose head lives outside this table). None is reported
    BROKEN by the caller: an unverifiable chain is not a valid one.
    """
    by_prev: dict[str, AuditEvent] = {}
    for event in events:
        if event.prev_hash is not None:
            # A fork (two events claiming the same parent) leaves the second
            # link unvisited by the walk — the orphan check below catches it.
            by_prev.setdefault(event.prev_hash, event)
    heads = [e for e in events if e.prev_hash is None]

    ordered: list[AuditEvent] = []
    seen_ids: set[str] = set()
    for head in heads:
        current = head
        ordered.append(current)
        seen_ids.add(current.event_id)
        while True:
            if current.event_hash is None:
                break  # a chain event always carries a stamped hash
            nxt = by_prev.get(current.event_hash)
            if nxt is None or nxt.event_id in seen_ids:  # chain end / cycle guard
                break
            ordered.append(nxt)
            seen_ids.add(nxt.event_id)
            current = nxt

    if len(ordered) != len(events):
        return None  # orphan / fork / gap — the table is not one coherent chain
    return ordered


async def verify_audit_postgres(pg_url: str, pubkey_hex: str | None = None) -> AuditVerifyReport:
    """Verify per-worker chains + optional signatures in the audit_events table.

    Requires the ``[db]`` extra (sqlalchemy[asyncio] + asyncpg). Rows are read
    via the ORM, grouped per ``worker_id``, and each worker's chain is
    reconstructed by link-walk (see :func:`reconstruct_chain_order`) before
    hash + signature verification — the same verdicts as the JSONL path.

    Verification is read-only and connects with a THROWAWAY engine (it does
    not touch the engine singleton the app lifespan manages).
    """
    from sqlalchemy import select

    from neuralguard.db.engine import create_engine as db_create_engine
    from neuralguard.db.models import AuditEventORM
    from neuralguard.db.session import session_factory
    from neuralguard.models.schemas import ThreatCategory, Verdict

    engine = db_create_engine(pg_url)
    events: list[AuditEvent] = []
    parse_errors = 0
    try:
        # ORM entity results REQUIRE a session context — a bare
        # connection.execute(select(ORM)) returns raw Row tuples (live-fire
        # caught: .scalars() then yielded the event_id UUID column instead of
        # entities).
        factory = session_factory()
        async with factory() as session:
            rows = (await session.execute(select(AuditEventORM))).scalars().all()
        for row in rows:
            try:
                events.append(
                    AuditEvent(
                        event_id=str(row.event_id),
                        request_id=row.request_id or str(uuid.uuid4()),
                        tenant_id=row.tenant_id or "unknown",
                        timestamp=row.timestamp,
                        verdict=Verdict(row.verdict),
                        findings_count=row.findings_count or 0,
                        threat_categories=[
                            ThreatCategory(tc) for tc in (row.threat_categories or [])
                        ],
                        confidence=row.confidence if row.confidence is not None else 0.0,
                        total_latency_ms=row.total_latency_ms or 0.0,
                        scanner_details=[dict(d) for d in (row.scanner_details or [])],
                        metadata=dict(row.metadata_ or {}),
                        worker_id=row.worker_id,
                        prev_hash=row.prev_hash,
                        event_hash=row.event_hash,
                        event_sig=row.event_sig,
                    )
                )
            except Exception:
                parse_errors += 1
                logger.warning("audit_verify_pg_row_rejected", event_id=str(row.event_id))
    finally:
        await engine.dispose()

    chains: dict[str, list[AuditEvent]] = {}
    for event in events:
        chains.setdefault(event.worker_id or "<unknown-worker>", []).append(event)

    report = AuditVerifyReport(
        files_read=1,  # one table
        events_parsed=len(events),
        parse_errors=parse_errors,
        chains=[],
    )
    for worker_id in sorted(chains):
        worker_events = chains[worker_id]
        ordered = reconstruct_chain_order(worker_events)
        valid = ordered is not None and verify_chain(ordered)
        if valid and pubkey_hex is not None:
            from neuralguard.logging.signing import verify_event_signature

            valid = all(
                e.event_sig is not None
                and e.event_hash is not None
                and verify_event_signature(e.event_hash, e.event_sig, pubkey_hex)
                for e in worker_events
            )
        report.chains.append(
            ChainReport(worker_id=worker_id, event_count=len(worker_events), valid=valid)
        )
    return report
