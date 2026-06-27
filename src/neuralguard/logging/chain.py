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
- It does not prevent a privileged operator from deleting a whole chain.
- It does not order events across workers. Cross-worker tamper-evidence
  requires a WORM sink or a DB-level sequence (tracked as P2).
- It does not sign events (no key material); it detects modification, not
  forgery of an entirely new file. Add Ed25519 signing (P2) for that.

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
