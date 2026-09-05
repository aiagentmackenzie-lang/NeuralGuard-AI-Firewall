"""P2-7: SIEM routing + BLOCK-rate spike alerting.

Hermetic: every network interaction goes through an injected httpx
MockTransport. Covers payload shapes, auth headers, failure tolerance,
bounded concurrency, spike edge-triggering, and the wiring seam
(AuditLogger routes when a router is present; unknown-key gate knows the
NEURALGUARD_SIEM_* surface).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from neuralguard.config.settings import SiemSettings, known_env_keys
from neuralguard.logging.audit import AuditLogger
from neuralguard.models.schemas import (
    AuditEvent,
    EvaluateRequest,
    EvaluateResponse,
    LayerArbitrationResult,
    Verdict,
)
from neuralguard.siem import SiemRouter


def _settings(**over: Any) -> SiemSettings:
    base: dict[str, Any] = {
        "enabled": True,
        "webhook_url": "http://siem.test/ingest",
        "webhook_token": "whsec_test_token_123",
        "splunk_hec_url": "https://splunk.test:8088",
        "splunk_hec_token": "splunk_test_token_456",
        "spike_window": 10,
        "spike_block_threshold": 0.5,
        "spike_cooldown_seconds": 300,
    }
    base.update(over)
    return SiemSettings(**base)


def _event(verdict: Verdict = Verdict.BLOCK) -> AuditEvent:
    return AuditEvent(
        request_id="req-1",
        tenant_id="default",
        verdict=verdict,
        findings_count=1,
        threat_categories=[],
        confidence=0.9,
        total_latency_ms=1.0,
    )


class _Capture:
    """Collects requests seen by a MockTransport (sync + async variants)."""

    def __init__(self, status: int = 200, delay: float = 0.0) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.status = status
        self.delay = delay

    def _record(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content.decode()))
        return httpx.Response(self.status, json={})

    def transport(self) -> httpx.MockTransport:
        capture = self

        def handler(request: httpx.Request) -> httpx.Response:
            return capture._record(request)

        return httpx.MockTransport(handler)

    def async_transport(self) -> httpx.MockTransport:
        """Async handler so delays actually yield the loop (overlap tests)."""
        capture = self

        async def handler(request: httpx.Request) -> httpx.Response:
            if capture.delay:
                await asyncio.sleep(capture.delay)
            return capture._record(request)

        return httpx.MockTransport(handler)

    def webhook_bodies(self) -> list[dict[str, Any]]:
        """Bodies delivered to the generic-webhook sink (one per event)."""
        return [json.loads(r.content.decode()) for r in self.requests if "siem.test" in str(r.url)]


def _make_router(
    settings: SiemSettings, capture: _Capture, async_client: bool = False
) -> SiemRouter:
    transport = capture.async_transport() if async_client else capture.transport()
    return SiemRouter(settings, transport=transport)


async def _drain() -> None:
    """Let scheduled background tasks run to completion."""
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    if pending:
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=5)


@pytest.mark.asyncio
async def test_both_sinks_receive_verdict_event() -> None:
    capture = _Capture()
    router = _make_router(_settings(), capture)
    router.route(_event(Verdict.BLOCK))
    await _drain()

    assert len(capture.requests) == 2
    urls = [str(r.url) for r in capture.requests]
    assert any("splunk.test:8088/services/collector/event" in u for u in urls)
    assert any("siem.test/ingest" in u for u in urls)

    splunk_req = next(r for r in capture.requests if "splunk" in str(r.url))
    assert splunk_req.headers["Authorization"] == "Splunk splunk_test_token_456"
    splunk_body = json.loads(splunk_req.content.decode())
    assert splunk_body["sourcetype"] == "neuralguard:verdict"
    assert splunk_body["event"]["event_type"] == "neuralguard.verdict"
    assert splunk_body["event"]["event"]["verdict"] == "block"

    webhook_req = next(r for r in capture.requests if "siem.test" in str(r.url))
    assert webhook_req.headers["Authorization"] == "Bearer whsec_test_token_123"
    webhook_body = json.loads(webhook_req.content.decode())
    assert webhook_body["event_type"] == "neuralguard.verdict"
    assert webhook_body["event"]["verdict"] == "block"


@pytest.mark.asyncio
async def test_event_carries_chain_hash_for_tamper_evidence() -> None:
    """SIEM consumers receive the tamper-evident event (chain hash stamped)."""
    capture = _Capture()
    logger = AuditLogger.__new__(AuditLogger)  # bypass FS init; test routing only
    logger.settings = type("S", (), {"enabled": False})()
    router = _make_router(_settings(), capture)
    event = _event(Verdict.BLOCK)
    event.worker_id = "w1"
    event.event_hash = "abc123"
    event.prev_hash = None
    router.route(event)
    await _drain()

    webhook_body = capture.webhook_bodies()[0]
    assert webhook_body["event"]["worker_id"] == "w1"
    assert webhook_body["event"]["event_hash"] == "abc123"


@pytest.mark.asyncio
async def test_sink_failure_is_swallowed_and_counted() -> None:
    capture = _Capture(status=500)
    router = _make_router(_settings(), capture)
    router.route(_event(Verdict.ALLOW))  # must not raise
    await _drain()
    assert router._drops >= 1


@pytest.mark.asyncio
async def test_inflight_cap_drops_without_unbounded_queue() -> None:
    capture = _Capture(delay=0.3)  # slow sink holds the single permit
    settings = _settings(max_inflight=1)
    router = _make_router(settings, capture, async_client=True)
    for _ in range(5):
        router.route(_event(Verdict.ALLOW))
    await asyncio.sleep(0.05)  # first delivery holds the semaphore now
    assert router._semaphore.locked()
    await _drain()
    assert router._drops >= 1  # the excess were dropped, not queued


def test_no_running_loop_uses_sync_delivery() -> None:
    capture = _Capture()
    router = _make_router(_settings(), capture)
    router.route(_event(Verdict.ALLOW))  # sync context → _deliver_sync fallback
    assert len(capture.requests) == 2


def test_spike_edge_trigger_fires_once_per_episode() -> None:
    capture = _Capture()
    router = _make_router(_settings(), capture)  # window 10, threshold 0.5
    for _ in range(5):  # window not yet full — no alert
        router.route(_event(Verdict.BLOCK))
    for _ in range(6):  # window fills with ≥50% blocks → ONE alert
        router.route(_event(Verdict.BLOCK))
    alerts = sum(
        1 for b in capture.webhook_bodies() if b.get("event_type") == "neuralguard.block_spike"
    )
    assert alerts == 1
    assert router._in_spike is True


def test_spike_recovery_rearms_and_cooldown_suppresses() -> None:
    capture = _Capture()
    router = _make_router(_settings(), capture)

    def push(verdict: Verdict, n: int) -> None:
        for _ in range(n):
            router.route(_event(verdict))

    push(Verdict.BLOCK, 10)  # spike fires (window 10/10 blocks)
    push(Verdict.ALLOW, 10)  # ratio recovers → re-armed
    push(Verdict.BLOCK, 10)  # re-cross WITHIN cooldown → suppressed
    spike_events = [b for b in capture.bodies if b.get("event_type") == "neuralguard.block_spike"]
    assert len(spike_events) == 1  # cooldown held the second alert


def test_spike_alert_payload_shape() -> None:
    capture = _Capture()
    router = _make_router(_settings(), capture)
    for _ in range(10):
        router.route(_event(Verdict.BLOCK))
    spike = next(b for b in capture.bodies if b.get("event_type") == "neuralguard.block_spike")
    assert spike["event"]["alert"] == "block_rate_spike"
    assert spike["event"]["window_events"] == 10
    assert spike["event"]["block_count"] == 10
    assert spike["event"]["block_ratio"] == 1.0
    assert spike["event"]["threshold"] == 0.5


def test_disabled_or_sinkless_router_is_noop() -> None:
    capture = _Capture()
    router = _make_router(_settings(), capture)
    router._sinks = []  # sinkless posture
    router.route(_event(Verdict.BLOCK))
    assert capture.requests == []


def test_siem_env_keys_are_known_to_f5_gate() -> None:
    known = known_env_keys()
    for key in (
        "NEURALGUARD_SIEM_ENABLED",
        "NEURALGUARD_SIEM_SPLUNK_HEC_URL",
        "NEURALGUARD_SIEM_SPLUNK_HEC_TOKEN",
        "NEURALGUARD_SIEM_SPLUNK_SOURCE_TYPE",
        "NEURALGUARD_SIEM_WEBHOOK_URL",
        "NEURALGUARD_SIEM_WEBHOOK_TOKEN",
        "NEURALGUARD_SIEM_TIMEOUT_SECONDS",
        "NEURALGUARD_SIEM_MAX_INFLIGHT",
        "NEURALGUARD_SIEM_SPIKE_WINDOW",
        "NEURALGUARD_SIEM_SPIKE_BLOCK_THRESHOLD",
        "NEURALGUARD_SIEM_SPIKE_COOLDOWN_SECONDS",
    ):
        assert key in known, f"{key} missing from known_env_keys()"


def test_auditlogger_routes_when_router_present(tmp_path: Any) -> None:
    """The wiring seam: _persist taps the router with the stamped event."""
    from neuralguard.config.settings import AuditSettings

    audit_settings = AuditSettings(jsonl_path=tmp_path / "audit")
    audit_logger = AuditLogger(audit_settings)

    routed: list[AuditEvent] = []

    class _StubRouter:
        def route(self, event: AuditEvent) -> None:
            routed.append(event)

    audit_logger._siem = _StubRouter()  # type: ignore[assignment]

    request = EvaluateRequest(prompt="hello")
    response = EvaluateResponse(
        request_id="r1",
        tenant_id="default",
        verdict=Verdict.ALLOW,
        findings=[],
        confidence=0.0,
        total_latency_ms=1.0,
        scan_layers_used=[],
    )
    arbitration = LayerArbitrationResult(
        verdict=Verdict.ALLOW,
        findings=[],
        scanner_results=[],
        total_latency_ms=1.0,
        arbitration_reason="test",
    )
    audit_logger.log_evaluation(request, response, arbitration)
    assert len(routed) == 1
    assert routed[0].event_hash is not None  # chain hash stamped BEFORE routing


def test_auditlogger_no_router_by_default(tmp_path: Any) -> None:
    from neuralguard.config.settings import AuditSettings

    audit_settings = AuditSettings(jsonl_path=tmp_path / "audit")
    audit_logger = AuditLogger(audit_settings)
    assert audit_logger._siem is None
