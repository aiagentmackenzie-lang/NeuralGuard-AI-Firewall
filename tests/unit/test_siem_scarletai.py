"""P2-7 extension: SecurityScarletAI sink — NeuralGuard feeds the local SIEM.

ScarletAI ingests ECS-normalized events at POST /api/v1/ingest (batch of
1-1000 IngestEvent dicts, bearer auth via the scoped INGEST_BEARER_TOKEN)
and fires its correlation chains on arrival. This suite pins the mapping
(AuditEvent → IngestEvent), the severity ladder, ALLOW filtering, spike
delivery, and auth headers.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from neuralguard.config.settings import SiemSettings
from neuralguard.models.schemas import Verdict
from neuralguard.siem import SiemRouter, map_to_scarletai


def _settings(**over: Any) -> SiemSettings:
    base: dict[str, Any] = {
        "enabled": True,
        "splunk_hec_url": None,
        "webhook_url": None,
        "scarletai_url": "http://127.0.0.1:8000/api/v1/ingest",
        "scarletai_token": "ingest_scoped_token_123",
        "scarletai_host": "ng-appliance-01",
        "scarletai_route_allow": False,
        "spike_window": 10,
        "spike_block_threshold": 0.5,
    }
    base.update(over)
    return SiemSettings(**base)


class _Capture:
    def __init__(self, status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.status = status

    def transport(self) -> httpx.MockTransport:
        capture = self

        def handler(request: httpx.Request) -> httpx.Response:
            capture.requests.append(request)
            return httpx.Response(capture.status, json={})

        return httpx.MockTransport(handler)

    def bodies(self) -> list[Any]:
        return [json.loads(r.content.decode()) for r in self.requests]


def _make(settings: SiemSettings, capture: _Capture) -> SiemRouter:
    return SiemRouter(settings, transport=capture.transport())


async def _drain() -> None:
    import asyncio as _aio

    pending = [t for t in _aio.all_tasks() if t is not _aio.current_task() and not t.done()]
    if pending:
        await _aio.wait_for(_aio.gather(*pending, return_exceptions=True), timeout=5)


@pytest.mark.asyncio
async def test_ingestevent_shape_and_mapping() -> None:
    capture = _Capture()
    router = _make(_settings(), capture)
    from neuralguard.models.schemas import AuditEvent

    event = AuditEvent(
        request_id="r1",
        tenant_id="acme",
        verdict=Verdict.BLOCK,
        findings_count=2,
        threat_categories=[],
        confidence=0.95,
        total_latency_ms=1.0,
    )
    event.event_hash = "ab" * 32
    router.route(event)
    await _drain()

    assert len(capture.requests) == 1
    body = capture.bodies()[0]
    assert isinstance(body, list) and len(body) == 1  # ScarletAI batch of 1
    mapped = body[0]
    assert mapped["source"] == "neuralguard"
    assert mapped["host_name"] == "ng-appliance-01"
    assert mapped["event_category"] == "intrusion_detection"
    assert mapped["event_type"] == "info"
    assert mapped["event_action"] == "verdict_block"
    assert mapped["severity"] == "critical"  # block @ confidence 0.95 ≥ 0.9
    ng = mapped["raw_data"]["neuralguard"]
    assert ng["event_hash"] == "ab" * 32  # tamper-evidence carried into the SIEM
    assert ng["tenant_id"] == "acme"
    assert mapped["@timestamp"].endswith(("Z", "+00:00"))
    req = capture.requests[0]
    assert req.headers["Authorization"] == "Bearer ingest_scoped_token_123"


@pytest.mark.asyncio
async def test_severity_ladder() -> None:
    from neuralguard.models.schemas import AuditEvent

    cases = {
        Verdict.BLOCK: "high",
        Verdict.ESCALATE: "medium",
        Verdict.SANITIZE: "medium",
        Verdict.QUARANTINE: "critical",
        Verdict.RATE_LIMIT: "low",
        Verdict.ALLOW: "info",
    }
    for verdict, expected in cases.items():
        event = AuditEvent(
            request_id="r",
            tenant_id="t",
            verdict=verdict,
            findings_count=0,
            threat_categories=[],
            confidence=0.5,
            total_latency_ms=1.0,
        )
        mapped = map_to_scarletai(
            {
                "event_type": "neuralguard.verdict",
                "time": 0.0,
                "event": event.model_dump(mode="json"),
            },
            _settings(),
        )
        assert mapped["severity"] == expected, verdict


@pytest.mark.asyncio
async def test_allow_filtered_by_default_routed_when_enabled() -> None:
    from neuralguard.models.schemas import AuditEvent

    capture = _Capture()
    router = _make(_settings(), capture)  # route_allow=False
    router.route(
        AuditEvent(
            request_id="r",
            tenant_id="t",
            verdict=Verdict.ALLOW,
            findings_count=0,
            threat_categories=[],
            confidence=0.0,
            total_latency_ms=1.0,
        )
    )
    await _drain()
    assert capture.requests == []  # filtered

    capture2 = _Capture()
    router2 = _make(_settings(scarletai_route_allow=True), capture2)
    router2.route(
        AuditEvent(
            request_id="r",
            tenant_id="t",
            verdict=Verdict.ALLOW,
            findings_count=0,
            threat_categories=[],
            confidence=0.0,
            total_latency_ms=1.0,
        )
    )
    await _drain()
    assert len(capture2.requests) == 1
    assert capture2.bodies()[0][0]["severity"] == "info"


def test_allow_still_counts_for_spike_ratio() -> None:
    """Filtering removes the DELIVERY, not the spike detector's denominator."""
    capture = _Capture()
    router = _make(_settings(), capture)
    for _ in range(10):  # all ALLOW → 0% block ratio, no spike
        router.route(
            __import__("neuralguard.models.schemas", fromlist=["AuditEvent"]).AuditEvent(
                request_id="r",
                tenant_id="t",
                verdict=Verdict.ALLOW,
                findings_count=0,
                threat_categories=[],
                confidence=0.0,
                total_latency_ms=1.0,
            )
        )
    assert router._block_count == 0
    assert len(router._recent_blocks) == 10  # denominator preserved


@pytest.mark.asyncio
async def test_spike_alerts_reach_scarletai_as_critical() -> None:
    capture = _Capture()
    router = _make(_settings(), capture)
    from neuralguard.models.schemas import AuditEvent

    for _ in range(10):  # window fills with blocks → spike fires
        router.route(
            AuditEvent(
                request_id="r",
                tenant_id="t",
                verdict=Verdict.BLOCK,
                findings_count=1,
                threat_categories=[],
                confidence=0.9,
                total_latency_ms=1.0,
            )
        )
    await _drain()
    spike_events = [
        b[0]
        for b in capture.bodies()
        if isinstance(b, list) and b and b[0]["event_action"] == "block_rate_spike"
    ]
    assert len(spike_events) >= 1
    assert spike_events[0]["severity"] == "critical"
    assert spike_events[0]["event_category"] == "intrusion_detection"
