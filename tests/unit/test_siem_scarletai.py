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
from neuralguard.models.schemas import AuditEvent, Verdict
from neuralguard.siem import SiemRouter, map_to_scarletai, map_to_scarletai_batch


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


# ── Companion events (fleet producer contract) ────────────────────────────
# One verdict audit event maps to a LIST of Scarlet IngestEvents posted in
# the SAME HTTP POST: the intrusion_detection parent + ai-category
# companions (ai_prompt_injection for injection-shaped verdicts,
# mcp_tool_denied for MCP-gate denials). Pinned here per Scarlet's closed
# vocabulary + the ai_usage producer doctrine (user_name = ACTOR slot).


def _audit_event(**over: Any) -> AuditEvent:
    base: dict[str, Any] = {
        "request_id": "r",
        "tenant_id": "acme",
        "verdict": Verdict.BLOCK,
        "findings_count": 1,
        "threat_categories": [],
        "confidence": 0.95,
        "total_latency_ms": 1.0,
    }
    base.update(over)
    return AuditEvent(**base)


def _envelope_for(event: AuditEvent) -> dict[str, Any]:
    return {
        "event_type": "neuralguard.verdict",
        "time": 0.0,
        "event": event.model_dump(mode="json"),
    }


def test_actor_slot_mapping_pins_tenant_in_user_name() -> None:
    """user_name = tenant_id on the parent (ECS-borrowed actor slot, capped).

    tenant_id ALSO stays inside raw_data.neuralguard — Scarlet's
    ai_verdict_block_sustained detector groups on it (load-bearing).
    """
    event = _audit_event(tenant_id="tenant-with-a-long-name" * 20)  # > 256 chars
    mapped = map_to_scarletai_batch(_envelope_for(event), _settings())[0]
    assert mapped["user_name"] == ("tenant-with-a-long-name" * 20)[:256]
    assert len(mapped["user_name"]) <= 256  # IngestEvent field cap
    assert mapped["raw_data"]["neuralguard"]["tenant_id"] == "tenant-with-a-long-name" * 20


@pytest.mark.asyncio
async def test_injection_companion_fires_same_post() -> None:
    capture = _Capture()
    router = _make(_settings(), capture)
    router.route(_audit_event(threat_categories=["T-PI-D"], confidence=0.95, verdict=Verdict.BLOCK))
    await _drain()

    assert len(capture.requests) == 1  # ONE POST for the whole family
    body = capture.bodies()[0]
    assert isinstance(body, list) and len(body) == 2
    parent, companion = body
    assert parent["event_action"] == "verdict_block"
    assert parent["user_name"] == "acme"
    assert companion["event_category"] == "ai"
    assert companion["event_type"] == "info"
    assert companion["event_action"] == "ai_prompt_injection"
    assert companion["source"] == "neuralguard"
    assert companion["user_name"] == "acme"  # actor slot on companions too
    assert companion["severity"] == parent["severity"] == "critical"  # ≥0.9 conf ladder
    assert companion["@timestamp"] == parent["@timestamp"]
    assert companion["host_name"] == parent["host_name"]
    assert companion["raw_data"]["neuralguard_companion"] == "prompt_injection"
    assert companion["raw_data"]["neuralguard"]["tenant_id"] == "acme"  # detector contract
    assert "process_name" not in companion  # no tool in an input scan


@pytest.mark.asyncio
async def test_injection_companion_fires_for_each_injection_category() -> None:
    for category in ("T-PI-D", "T-PI-I", "T-JB"):
        mapped = map_to_scarletai_batch(
            _envelope_for(_audit_event(threat_categories=[category])), _settings()
        )
        assert [e["event_action"] for e in mapped] == [
            "verdict_block",
            "ai_prompt_injection",
        ], category


@pytest.mark.asyncio
async def test_non_injection_categories_get_no_companion() -> None:
    for categories in (["T-EXF"], ["T-OUT"], ["T-EXF", "T-OUT"], []):
        mapped = map_to_scarletai_batch(
            _envelope_for(_audit_event(threat_categories=categories)), _settings()
        )
        assert len(mapped) == 1, categories


@pytest.mark.asyncio
async def test_mcp_denial_companion_fires() -> None:
    capture = _Capture()
    router = _make(_settings(), capture)
    router.route(
        _audit_event(
            confidence=0.5,  # plain block → high, not critical
            metadata={
                "mcp": True,
                "rule_id": "MCP-INTENT-001",
                "method": "tools/call",
                "tool": "investigate",
            },
        )
    )
    await _drain()

    body = capture.bodies()[0]
    assert len(body) == 2
    parent, companion = body
    assert companion["event_action"] == "mcp_tool_denied"
    assert companion["event_category"] == "ai"
    assert companion["process_name"] == "investigate"  # the denied tool
    assert companion["user_name"] == "acme"
    assert companion["severity"] == parent["severity"] == "high"
    assert companion["raw_data"]["neuralguard_companion"] == "mcp_denial"
    assert companion["@timestamp"] == parent["@timestamp"]


@pytest.mark.asyncio
async def test_mcp_denial_companion_not_on_allow_or_escalate_free_verdicts() -> None:
    for verdict in (Verdict.ALLOW, Verdict.SANITIZE, Verdict.RATE_LIMIT):
        mapped = map_to_scarletai_batch(
            _envelope_for(
                _audit_event(
                    verdict=verdict, confidence=0.5, metadata={"mcp": True, "tool": "investigate"}
                )
            ),
            _settings(),
        )
        assert len(mapped) == 1, verdict
    # escalate IS in the companion set
    mapped = map_to_scarletai_batch(
        _envelope_for(
            _audit_event(
                verdict=Verdict.ESCALATE,
                confidence=0.5,
                metadata={"mcp": True, "tool": "investigate"},
            )
        ),
        _settings(),
    )
    assert [e["event_action"] for e in mapped] == ["verdict_escalate", "mcp_tool_denied"]


@pytest.mark.asyncio
async def test_both_companions_on_injection_shaped_mcp_block() -> None:
    """An injection-shaped MCP gate block is two domain lenses — parent+2."""
    mapped = map_to_scarletai_batch(
        _envelope_for(
            _audit_event(
                threat_categories=["T-PI-I"],
                metadata={"mcp": True, "rule_id": "MCP-INTENT-001", "tool": "investigate"},
            )
        ),
        _settings(),
    )
    assert [e["event_action"] for e in mapped] == [
        "verdict_block",
        "ai_prompt_injection",
        "mcp_tool_denied",
    ]
    assert mapped[1]["raw_data"]["neuralguard_companion"] == "prompt_injection"
    assert mapped[2]["raw_data"]["neuralguard_companion"] == "mcp_denial"


def test_mcp_tool_missing_maps_to_null_process_name() -> None:
    mapped = map_to_scarletai_batch(
        _envelope_for(_audit_event(metadata={"mcp": True})), _settings()
    )
    assert len(mapped) == 2
    assert mapped[1]["event_action"] == "mcp_tool_denied"
    assert mapped[1]["process_name"] is None


def test_companion_build_failure_degrades_to_parent_only() -> None:
    """Purity: a companion path that raises degrades to parent-only."""
    payload = _envelope_for(_audit_event(threat_categories=["T-PI-D"]))
    payload["event"]["threat_categories"] = 12345  # not iterable → build raises
    mapped = map_to_scarletai_batch(payload, _settings())
    assert len(mapped) == 1
    assert mapped[0]["event_action"] == "verdict_block"
    assert "user_name" in mapped[0]  # parent slot mapping already applied


@pytest.mark.asyncio
async def test_allow_filter_drops_parent_and_companions_together() -> None:
    """Companions inherit the parent's routing decision (A5)."""
    capture = _Capture()
    router = _make(_settings(), capture)  # route_allow=False
    router.route(_audit_event(verdict=Verdict.ALLOW, confidence=0.0, threat_categories=["T-PI-I"]))
    await _drain()
    assert capture.requests == []  # the whole family filtered with the parent

    capture2 = _Capture()
    router2 = _make(_settings(scarletai_route_allow=True), capture2)
    router2.route(_audit_event(verdict=Verdict.ALLOW, confidence=0.0, threat_categories=["T-PI-I"]))
    await _drain()
    assert len(capture2.requests) == 1
    assert len(capture2.bodies()[0]) == 2  # parent + companion, same POST


def test_splunk_envelope_unchanged_by_companions() -> None:
    """Companions are scarletai-only: Splunk keeps the single-event envelope."""
    import json as _json

    settings = _settings(
        splunk_hec_url="https://splunk.test:8088",
        splunk_hec_token="splunk_test_token_456",
        scarletai_url="http://127.0.0.1:8000/api/v1/ingest",
    )
    capture = _Capture()
    router = SiemRouter(settings, transport=capture.transport())
    router.route(
        _audit_event(threat_categories=["T-PI-D"], metadata={"mcp": True, "tool": "investigate"})
    )  # sync context → sync delivery
    scarletai_bodies = [
        _json.loads(r.content.decode()) for r in capture.requests if "127.0.0.1:8000" in str(r.url)
    ]
    splunk_bodies = [
        _json.loads(r.content.decode()) for r in capture.requests if "splunk" in str(r.url)
    ]
    assert len(scarletai_bodies) == 1 and len(scarletai_bodies[0]) == 3  # parent + 2 companions
    # Splunk envelope UNCHANGED: one event, full audit dump inside event.event.
    assert len(splunk_bodies) == 1
    assert splunk_bodies[0]["event"]["event_type"] == "neuralguard.verdict"
    assert splunk_bodies[0]["event"]["event"]["verdict"] == "block"
