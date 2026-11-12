"""Hash-chained audit events for tamper-evidence (P1-4).

Each audit event carries ``prev_hash`` (the previous event's ``event_hash``
in this worker's chain, or ``None`` for the chain head) and ``event_hash``
(SHA-256 over a canonical encoding of the event plus ``prev_hash``).

Why per-worker: the audit logger is per-process, and a single shared JSONL
file receives concurrent appends from multiple uvicorn workers with no
defined inter-worker ordering. A single global chain would therefore be
incoherent across workers. Instead each worker (identified by a fresh
``worker_id`` per process) keeps its own chain. Tampering with a written
event on disk breaks that event's recorded ``event_hash`` *and* the next
event's ``prev_hash`` — so post-hoc tampering is detectable per chain.

What this is NOT (documented honestly):
What this is NOT (documented honestly):
- It does not prevent a privileged operator from deleting a whole chain.
- It does not order events across workers BY ITSELF. The postgres backend
  closes that with a DB sequence + global hash link (see the global-chain
  section at the bottom of this module); JSONL keeps per-worker chains only.
- It does not sign events (no key material in this module) — Ed25519 signing
  closes the forged-chain gap (P2-10, `logging/signing.py`; verify with
  `neuralguard audit-verify --pubkey <hex>`).

The canonical encoding is deterministic: fields are concatenated with a
delimiter that cannot appear in the hashed content's boundaries, and dict
fields are JSON-encoded with sorted keys + tight separators so re-serializing
the same logical content yields the same bytes.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralguard.models.schemas import AuditEvent


def _canonical_json(value: object) -> str:
    """Deterministic JSON encoding for hashing (sorted keys, tight separators)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def compute_event_hash(event: AuditEvent, prev_hash: str | None) -> str:
    """Compute the SHA-256 chain hash for one audit event."""
    h = hashlib.sha256()
    parts = [
        event.event_id,
        event.request_id,
        event.tenant_id,
        event.timestamp.isoformat(),
        event.verdict.value,
        str(event.findings_count),
        repr(event.confidence),
        repr(event.total_latency_ms),
        _canonical_json(event.scanner_details),
        _canonical_json(event.metadata),
        prev_hash or "",
    ]
    h.update("\x1f".join(parts).encode("utf-8"))
    return h.hexdigest()


def verify_chain(events: list[AuditEvent]) -> bool:
    """Return True if every event's hash and prev_hash are consistent.

    ``events`` must be in chain order (the order they were written). A single
    tampered event makes both its own ``event_hash`` and the next event's
    ``prev_hash`` fail to recompute, so verification returns False.
    """
    prev: str | None = None
    for event in events:
        expected = compute_event_hash(event, event.prev_hash)
        if event.event_hash != expected:
            return False
        if event.prev_hash != prev:
            return False
        prev = event.event_hash
    return True


# ── Global (cross-worker) ordering chain ────────────────────────────────────
#
# Per-worker chains (above) prove integrity WITHIN a process but say nothing
# about the order events from different workers hit durable storage. That gap
# is closed for the postgres backend by a DB-level global sequence plus a
# second hash link over it (P2-10 residual, closed 2026-11-12):
#
#   global_seq        BIGINT from ``audit_global_seq`` — total order of insert
#   global_prev_hash  the previous COMMITTED event's global_hash
#   global_hash       SHA-256(domain || global_prev_hash || event_hash)
#
# The writer serializes audit inserts with a Postgres transaction-scoped
# advisory lock, reads the last committed global hash, and stamps the link in
# the same transaction — so every chained event points at its true
# predecessor. Deleting or reordering a middle row breaks the link and is
# reported BROKEN by ``audit-verify --pg-url``.
#
# Honest limits (unchanged in kind from the per-worker chain):
# - Trimming the HEAD (retention) is not detectable from the table alone; the
#   chain is verified from the first observed row, exactly like a per-worker
#   chain whose head file was removed.
# - A DB writer who can rewrite every row can recompute this hash chain. The
#   per-event Ed25519 signatures (P2-10) are what make a rewritten chain fail
#   when verification is given a pubkey.
# - JSONL fallback events are outside the global chain (documented: the DB
#   was unavailable; the JSONL per-worker chain still covers them).

_GLOBAL_CHAIN_DOMAIN = "neuralguard.audit.global.v1"


def compute_global_hash(prev_global_hash: str | None, event_hash: str) -> str:
    """Hash-link one event into the cross-worker global chain."""
    h = hashlib.sha256()
    h.update(_GLOBAL_CHAIN_DOMAIN.encode("utf-8"))
    h.update(b"\x00")
    h.update((prev_global_hash or "").encode("utf-8"))
    h.update(b"\x00")
    h.update(event_hash.encode("utf-8"))
    return h.hexdigest()
