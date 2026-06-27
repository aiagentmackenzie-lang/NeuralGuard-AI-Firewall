"""Tests for audit hash-chaining / tamper-evidence (P1-4)."""

from __future__ import annotations

import json
from pathlib import Path

from neuralguard.config.settings import AuditSettings
from neuralguard.logging.audit import AuditLogger
from neuralguard.logging.chain import compute_event_hash, verify_chain
from neuralguard.models.schemas import (
    AuditEvent,
    EvaluateRequest,
    EvaluateResponse,
    LayerArbitrationResult,
    Verdict,
)


def _evt(
    eid: str = "e1",
    tenant: str = "t",
    verdict: Verdict = Verdict.ALLOW,
    confidence: float = 0.0,
) -> AuditEvent:
    return AuditEvent(
        event_id=eid,
        request_id="r1",
        tenant_id=tenant,
        verdict=verdict,
        findings_count=0,
        threat_categories=[],
        confidence=confidence,
        total_latency_ms=1.0,
    )


class TestChainHash:
    def test_hash_is_deterministic(self):
        e = _evt()
        h1 = compute_event_hash(e, None)
        h2 = compute_event_hash(e, None)
        assert h1 == h2
        assert len(h1) == 64

    def test_prev_hash_changes_the_hash(self):
        e = _evt()
        assert compute_event_hash(e, None) != compute_event_hash(e, "abc")

    def test_content_change_changes_the_hash(self):
        e1 = _evt(verdict=Verdict.ALLOW)
        e2 = _evt(verdict=Verdict.BLOCK)
        assert compute_event_hash(e1, None) != compute_event_hash(e2, None)

    def test_verify_chain_valid(self):
        # Build a 3-event chain manually.
        events = [_evt("e1"), _evt("e2"), _evt("e3")]
        prev = None
        for e in events:
            e.prev_hash = prev
            e.event_hash = compute_event_hash(e, prev)
            prev = e.event_hash
        assert verify_chain(events) is True

    def test_verify_chain_detects_tampered_content(self):
        events = [_evt("e1"), _evt("e2")]
        prev = None
        for e in events:
            e.prev_hash = prev
            e.event_hash = compute_event_hash(e, prev)
            prev = e.event_hash
        # Tamper with the first event's verdict AFTER hashing.
        events[0].verdict = Verdict.BLOCK
        assert verify_chain(events) is False

    def test_verify_chain_detects_tampered_hash(self):
        events = [_evt("e1"), _evt("e2")]
        prev = None
        for e in events:
            e.prev_hash = prev
            e.event_hash = compute_event_hash(e, prev)
            prev = e.event_hash
        events[1].event_hash = "deadbeef" + "0" * 56
        assert verify_chain(events) is False

    def test_verify_chain_detects_reorder(self):
        events = [_evt("e1"), _evt("e2"), _evt("e3")]
        prev = None
        for e in events:
            e.prev_hash = prev
            e.event_hash = compute_event_hash(e, prev)
            prev = e.event_hash
        # Swap two events — prev_hash links break.
        reordered = [events[0], events[2], events[1]]
        assert verify_chain(reordered) is False

    def test_empty_chain_is_valid(self):
        assert verify_chain([]) is True


class TestAuditLoggerChain:
    """End-to-end: the logger stamps hashes and the written JSONL verifies."""

    def _log_one(self, audit: AuditLogger, tenant: str = "t") -> AuditEvent:
        return audit.log_evaluation(
            EvaluateRequest(prompt="x", tenant_id=tenant),
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

    def _read_events(self, audit_dir: Path) -> list[AuditEvent]:
        events: list[AuditEvent] = []
        for f in sorted(audit_dir.glob("audit-*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    events.append(AuditEvent.model_validate_json(line))
        return events

    def test_events_stamped_and_chain_verifies(self, tmp_path: Path):
        audit = AuditLogger(AuditSettings(backend="jsonl", jsonl_path=tmp_path, tokenize_pii=False))
        self._log_one(audit, "t1")
        self._log_one(audit, "t2")
        self._log_one(audit, "t3")

        events = self._read_events(tmp_path)
        assert len(events) == 3
        # All share one worker_id (one process).
        assert len({e.worker_id for e in events}) == 1
        # First event is the chain head.
        assert events[0].prev_hash is None
        # Every event has a hash.
        assert all(e.event_hash and len(e.event_hash) == 64 for e in events)
        assert verify_chain(events) is True

    def test_disk_tamper_detected(self, tmp_path: Path):
        audit = AuditLogger(AuditSettings(backend="jsonl", jsonl_path=tmp_path, tokenize_pii=False))
        self._log_one(audit, "t1")
        self._log_one(audit, "t2")

        f = next(tmp_path.glob("audit-*.jsonl"))
        lines = f.read_text().splitlines()
        obj = json.loads(lines[0])
        obj["verdict"] = "block"  # tamper with the first event's verdict
        lines[0] = json.dumps(obj)
        f.write_text("\n".join(lines) + "\n")

        events = self._read_events(tmp_path)
        assert verify_chain(events) is False

    def test_chain_resumes_within_process(self, tmp_path: Path):
        audit = AuditLogger(AuditSettings(backend="jsonl", jsonl_path=tmp_path, tokenize_pii=False))
        self._log_one(audit, "t1")
        first_hash = audit._last_hash
        self._log_one(audit, "t2")
        # The second event's prev_hash must equal the first event's event_hash.
        events = self._read_events(tmp_path)
        assert events[1].prev_hash == first_hash
        assert events[1].prev_hash == events[0].event_hash
