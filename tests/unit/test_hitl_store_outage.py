"""NG-15: a store outage is a named, audited refusal — not an internal error.

Found in the two-pod cluster drill (2026-11-17), by accident and then on
purpose. With a HITL Redis URL the store could not authenticate, the escalation
route raised ``HitlStoreError`` out of the handler:

- the **security property held** — nothing was approved, nothing reached the
  upstream, and no path returned an approval from an error;
- but the operator saw ``500 {"error":"internal_error"}`` with **no rule id**,
  and **no audit event was written**, so a SIEM watching the audit chain could
  not see that the control was refusing every approval-gated call. The same
  mistake in production looks like "the gateway is broken", not "the store is
  down", and the wrong diagnosis is how a control gets disabled.

Worse, the decide route already caught ``HitlError`` — and ``HitlStoreError``
subclasses it — so a store outage on a decision was reported as
``409 mcp_escalation_not_pending`` ("expired between the check and the
decision"): a *wrong* diagnosis, not a vague one.

The contracts below fix the vocabulary:

| Surface | Outage behaviour |
|---|---|
| raise (a call is gated) | 503 ``mcp_escalation_store_unavailable`` + audit ``MCP-HITL-STORE`` |
| list (approver polling) | 503 same shape, **no audit event** — polling must not become an audit amplifier |
| decide | 503 same shape + audit ``MCP-HITL-STORE`` — never 409 |
| genuine expiry | still 409 ``mcp_escalation_not_pending`` |
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import AuthSettings, NeuralGuardConfig
from neuralguard.main import create_app
from neuralguard.mcp.hitl import (
    STATUS_PENDING,
    HitlBroker,
    HitlStoreError,
    PendingEscalation,
)
from neuralguard.mcp.intent_gate import McpToolPolicy, ToolAction

_CALLER_KEY = "caller-key"
_APPROVER_KEY = "approver-key"
_UPSTREAM_RESPONSE = {"jsonrpc": "2.0", "id": 1, "result": {"content": [], "isError": False}}


class _OutageStore:
    """Every operation fails the way a Redis outage does."""

    async def raise_escalation(self, **kwargs: Any) -> Any:
        raise HitlStoreError("shared HITL store unavailable: connection refused")

    async def pending_for_approval(self) -> list[Any]:
        raise HitlStoreError("shared HITL store unavailable: connection refused")

    async def get(self, escalation_id: str) -> Any:
        raise HitlStoreError("shared HITL store unavailable: connection refused")

    async def decide(self, escalation_id: str, **kwargs: Any) -> Any:
        raise HitlStoreError("shared HITL store unavailable: connection refused")

    async def wait(self, escalation_id: str, timeout: float) -> Any:
        raise HitlStoreError("shared HITL store unavailable: connection refused")

    async def consume(self, escalation_id: str) -> Any:
        raise HitlStoreError("shared HITL store unavailable: connection refused")


class _PendingButUndecidableStore(_OutageStore):
    """The store can be read but the write of a decision fails."""

    def __init__(self) -> None:
        now = time.monotonic()
        self.record = PendingEscalation(
            escalation_id="esc-decision-outage",
            tenant="acme",
            agent="deploy-bot",
            method="tools/call",
            tool="deploy_prod",
            rule_id="MCP-GATE-ESC-001",
            reason="policy requires approval",
            source="intent_gate",
            digest="0" * 64,
            argument_preview='{"env": "prod"}',
            created_at=now,
            expires_at=now + 300.0,
            status=STATUS_PENDING,
        )

    async def get(self, escalation_id: str) -> Any:
        return self.record

    async def pending_for_approval(self) -> list[Any]:
        return [self.record]


class _StubTransport:
    """Scripted upstream: answers the catalog so the rug-pull gate is satisfied,
    and records only forwarded tool calls (the thing the drill cares about)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def forward(self, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if payload.get("method") == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "result": {
                    "tools": [
                        {
                            "name": "deploy_prod",
                            "description": "Deploy a build",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                },
            }
        self.calls.append(payload)
        return _UPSTREAM_RESPONSE

    async def aclose(self) -> None:  # pragma: no cover - symmetry
        return None


def _app(store: Any) -> tuple[Any, _StubTransport, list[dict[str, Any]]]:
    config = NeuralGuardConfig(
        environment="development",
        auth=AuthSettings(
            enabled=True,
            api_keys=[
                f"{_CALLER_KEY}|acme|deploy-bot",
                f"{_APPROVER_KEY}|admin",
            ],
            public_endpoints=[],
        ),
    )
    for key, value in {
        "enabled": True,
        "upstream_url": "http://fake-mcp-server.local",
        "timeout_seconds": 5.0,
        "server_id": "test-mcp",
        "hitl_enabled": True,
        "hitl_timeout_seconds": 5.0,
        "hitl_ttl_seconds": 60.0,
        "hitl_approver_tenants": ["admin"],
    }.items():
        setattr(config.mcp, key, value)
    app = create_app(config)
    policy = McpToolPolicy(tool_rules={"deploy_prod": ToolAction.ESCALATE})

    class _FakeRegistry:
        enabled = True

        def get(self, tenant_id: str) -> Any:
            class _Cfg:
                mcp = policy

            return _Cfg()

    app.state.tenant_registry = _FakeRegistry()
    stub = _StubTransport()
    app.state.mcp_transport = stub
    app.state.mcp_hitl = store
    captured: list[dict[str, Any]] = []
    app.state.audit_logger.log_mcp_event = lambda **kw: captured.append(kw)  # type: ignore[method-assign]
    return app, stub, captured


def _call_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_CALLER_KEY}",
        "Mcp-Method": "tools/call",
        "Mcp-Name": "deploy_prod",
    }


def _approver() -> dict[str, str]:
    return {"Authorization": f"Bearer {_APPROVER_KEY}"}


async def _prime(client: AsyncClient) -> None:
    await client.post(
        "/v1/mcp",
        json={"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}},
        headers={"Mcp-Method": "tools/list", "Authorization": f"Bearer {_CALLER_KEY}"},
    )


_RPC = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "deploy_prod", "arguments": {"env": "prod"}},
}


class TestStoreOutageIsANamedRefusal:
    @pytest.mark.asyncio
    async def test_a_gated_call_is_refused_by_name_not_internal_error(self) -> None:
        app, stub, captured = _app(_OutageStore())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            response = await client.post("/v1/mcp", json=_RPC, headers=_call_headers())
        assert response.status_code == 503, response.text
        body = response.json()
        assert body["error"] == "mcp_escalation_store_unavailable"
        assert body["rule_id"] == "MCP-HITL-STORE"
        # The security property: an unverifiable store is never an approval.
        assert stub.calls == [], "no call may reach the upstream while the store is down"
        assert any(event.get("rule_id") == "MCP-HITL-STORE" for event in captured), captured

    @pytest.mark.asyncio
    async def test_the_refusal_is_audited_with_the_store_outcome(self) -> None:
        app, _stub, captured = _app(_OutageStore())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            await client.post("/v1/mcp", json=_RPC, headers=_call_headers())
        store_events = [e for e in captured if e.get("rule_id") == "MCP-HITL-STORE"]
        assert store_events, captured
        assert store_events[-1]["details"]["outcome"] == "escalation_store_unavailable"

    @pytest.mark.asyncio
    async def test_approver_polling_reports_the_outage_without_auditing_each_poll(self) -> None:
        app, _stub, captured = _app(_OutageStore())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            first = await client.get("/v1/mcp/escalations", headers=_approver())
            for _ in range(3):
                again = await client.get("/v1/mcp/escalations", headers=_approver())
                assert again.status_code == 503
        assert first.status_code == 503
        assert first.json()["error"] == "mcp_escalation_store_unavailable"
        assert first.json()["rule_id"] == "MCP-HITL-STORE"
        # Polling is not a gate decision: four polls, zero audit events.
        assert captured == [], captured

    @pytest.mark.asyncio
    async def test_a_decision_that_cannot_be_written_is_not_reported_as_expiry(self) -> None:
        # HitlStoreError subclasses HitlError, so the pre-existing 409 handler
        # would swallow an outage and tell the approver the escalation expired.
        app, _stub, captured = _app(_PendingButUndecidableStore())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            response = await client.post(
                "/v1/mcp/escalations/esc-decision-outage",
                json={"approve": True},
                headers=_approver(),
            )
        assert response.status_code == 503, response.text
        assert response.json()["error"] == "mcp_escalation_store_unavailable"
        assert response.json()["rule_id"] == "MCP-HITL-STORE"
        assert any(e.get("rule_id") == "MCP-HITL-STORE" for e in captured), captured


class TestTheHealthyPathsAreUnchanged:
    @pytest.mark.asyncio
    async def test_a_healthy_store_still_approves_and_forwards(self) -> None:
        broker = HitlBroker(ttl_seconds=60.0, max_pending=8)
        app, stub, _captured = _app(broker)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            task = asyncio.create_task(client.post("/v1/mcp", json=_RPC, headers=_call_headers()))
            record: dict[str, Any] | None = None
            for _ in range(200):
                listed = await client.get("/v1/mcp/escalations", headers=_approver())
                assert listed.status_code == 200, listed.text
                if listed.json()["escalations"]:
                    record = listed.json()["escalations"][0]
                    break
                await asyncio.sleep(0.01)
            assert record is not None
            decided = await client.post(
                f"/v1/mcp/escalations/{record['escalation_id']}",
                json={"approve": True},
                headers=_approver(),
            )
            assert decided.status_code == 200, decided.text
            gated = await task
        assert gated.status_code == 200, gated.text
        assert len(stub.calls) == 1, stub.calls
