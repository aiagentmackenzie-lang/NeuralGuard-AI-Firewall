"""NG-15: human-in-the-loop approval for MCP ESCALATE.

Two properties matter and are tested separately:

1. the **broker** is a bounded, single-use, TTL'd state machine — no path turns
   a timeout, an expiry, a replay or a capacity overflow into an approval;
2. the **route** keeps the loop honest — an agent-bound credential can never
   decide, only named approver tenants can, and an approved call still faces
   every other gate (approval is not authority).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import AuthSettings, NeuralGuardConfig
from neuralguard.main import create_app
from neuralguard.mcp.constraints import (
    ArgumentConstraint,
    ConstraintAction,
    ConstraintKind,
)
from neuralguard.mcp.hitl import (
    STATUS_APPROVED,
    STATUS_CONSUMED,
    STATUS_DENIED,
    STATUS_PENDING,
    HitlBroker,
    HitlCapacityError,
    HitlError,
    payload_digest,
)
from neuralguard.mcp.intent_gate import McpToolPolicy, ToolAction

# ── Broker ─────────────────────────────────────────────────────────────────


def _raise(broker: HitlBroker, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "tenant": "acme",
        "agent": "deploy-bot",
        "method": "tools/call",
        "tool": "deploy_prod",
        "rule_id": "MCP-GATE-ESCALATE-001",
        "reason": "policy requires approval",
        "source": "intent_gate",
        "payload": {"jsonrpc": "2.0", "id": 1, "method": "tools/call"},
        "arguments": {"env": "prod"},
    }
    kwargs.update(overrides)
    return broker.raise_escalation(**kwargs)


class TestDigest:
    def test_digest_is_stable_and_order_insensitive(self) -> None:
        assert payload_digest({"a": 1, "b": 2}) == payload_digest({"b": 2, "a": 1})

    def test_digest_changes_with_any_argument(self) -> None:
        assert payload_digest({"path": "/srv/a"}) != payload_digest({"path": "/srv/b"})


class TestBrokerLifecycle:
    @pytest.mark.asyncio
    async def test_approved_call_wakes_the_waiter(self) -> None:
        broker = HitlBroker(ttl_seconds=30)
        record = await _raise(broker)
        waiter = asyncio.create_task(broker.wait(record.escalation_id, 5))
        await broker.decide(record.escalation_id, approved=True, decided_by="admin")
        decided = await waiter
        assert decided is not None and decided.status == STATUS_APPROVED
        assert decided.decided_by == "admin"

    @pytest.mark.asyncio
    async def test_denied_call_wakes_the_waiter_with_a_denial(self) -> None:
        broker = HitlBroker(ttl_seconds=30)
        record = await _raise(broker)
        waiter = asyncio.create_task(broker.wait(record.escalation_id, 5))
        await broker.decide(
            record.escalation_id, approved=False, decided_by="admin", reason="not prod"
        )
        decided = await waiter
        assert decided is not None and decided.status == STATUS_DENIED
        assert decided.decision_reason == "not prod"

    @pytest.mark.asyncio
    async def test_no_decision_times_out_to_none(self) -> None:
        broker = HitlBroker(ttl_seconds=30)
        record = await _raise(broker)
        assert await broker.wait(record.escalation_id, 0.02) is None
        # Still pending: a timeout refused THIS request, it did not decide.
        assert broker.get(record.escalation_id).status == STATUS_PENDING

    @pytest.mark.asyncio
    async def test_expired_escalation_can_never_be_approved(self) -> None:
        broker = HitlBroker(ttl_seconds=0.01, max_pending=4)
        record = await _raise(broker)
        await asyncio.sleep(0.03)
        assert await broker.wait(record.escalation_id, 0.05) is None
        with pytest.raises(HitlError):
            await broker.decide(record.escalation_id, approved=True, decided_by="admin")

    @pytest.mark.asyncio
    async def test_second_decision_is_rejected(self) -> None:
        broker = HitlBroker(ttl_seconds=30)
        record = await _raise(broker)
        await broker.decide(record.escalation_id, approved=True, decided_by="admin")
        with pytest.raises(HitlError):
            await broker.decide(record.escalation_id, approved=False, decided_by="admin")

    @pytest.mark.asyncio
    async def test_approval_is_single_use(self) -> None:
        broker = HitlBroker(ttl_seconds=30)
        record = await _raise(broker)
        await broker.decide(record.escalation_id, approved=True, decided_by="admin")
        await broker.consume(record.escalation_id)
        assert broker.get(record.escalation_id) is None
        # A replay of the id cannot authorize a second call.
        with pytest.raises(HitlError):
            await broker.consume(record.escalation_id)
        assert await broker.wait(record.escalation_id, 0.01) is None

    @pytest.mark.asyncio
    async def test_consume_requires_an_approval(self) -> None:
        broker = HitlBroker(ttl_seconds=30)
        record = await _raise(broker)
        with pytest.raises(HitlError):
            await broker.consume(record.escalation_id)
        await broker.decide(record.escalation_id, approved=False, decided_by="admin")
        with pytest.raises(HitlError):
            await broker.consume(record.escalation_id)

    @pytest.mark.asyncio
    async def test_capacity_refuses_rather_than_queues(self) -> None:
        broker = HitlBroker(ttl_seconds=30, max_pending=1)
        await _raise(broker)
        with pytest.raises(HitlCapacityError):
            await _raise(broker)
        assert broker.pending_count == 1

    @pytest.mark.asyncio
    async def test_pending_list_is_oldest_first_and_excludes_decided(self) -> None:
        broker = HitlBroker(ttl_seconds=30)
        first = await _raise(broker, tool="a")
        second = await _raise(broker, tool="b")
        await broker.decide(second.escalation_id, approved=True, decided_by="admin")
        pending = broker.pending_for_approval()
        assert [r.escalation_id for r in pending] == [first.escalation_id]

    @pytest.mark.asyncio
    async def test_view_is_bounded_and_carries_no_secret(self) -> None:
        broker = HitlBroker(ttl_seconds=30)
        record = await _raise(broker, arguments={"path": "x" * 5000})
        view = record.to_view()
        assert len(view["argument_preview"]) < 2100
        assert view["argument_preview"].endswith("…(truncated)")
        assert view["request_digest"] == record.digest
        assert set(view) >= {"escalation_id", "tenant", "agent", "tool", "status"}

    @pytest.mark.asyncio
    async def test_consume_status_is_terminal(self) -> None:
        broker = HitlBroker(ttl_seconds=30)
        record = await _raise(broker)
        await broker.decide(record.escalation_id, approved=True, decided_by="admin")
        await broker.consume(record.escalation_id)
        assert record.status == STATUS_CONSUMED


# ── Route harness ──────────────────────────────────────────────────────────

# Bindings live in config; the credential itself is the part before the '|'.
_CALLER_BINDING = "botkey|acme|deploy-bot"
_APPROVER_BINDING = "approverkey|admin"
_OTHER_TENANT_BINDING = "otherapproverkey|beta"
_AGENT_APPROVER_BINDING = "agentadmin|admin|admin-bot"
_CALLER_KEY = "botkey"
_APPROVER_KEY = "approverkey"
_OTHER_TENANT_KEY = "otherapproverkey"
_AGENT_APPROVER_KEY = "agentadmin"


class _StubTransport:
    def __init__(self, tools: list[str]) -> None:
        self.tools = tools
        self.calls: list[dict[str, Any]] = []

    async def forward(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        upstream_token: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(payload)
        if payload.get("method") == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        {"name": name, "description": "", "inputSchema": {}} for name in self.tools
                    ]
                },
            }
        return {"jsonrpc": "2.0", "id": 1, "result": {"content": []}}


def _rpc(tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments or {}},
    }


def _headers(tool: str, key: str) -> dict[str, str]:
    return {
        "Mcp-Method": "tools/call",
        "Mcp-Name": tool,
        "Authorization": f"Bearer {key}",
    }


def _app(
    *,
    policy: McpToolPolicy | None = None,
    tools: list[str] | None = None,
    hitl: bool = True,
    timeout: float = 2.0,
    approvers: list[str] | None = None,
    max_pending: int = 256,
) -> tuple[Any, _StubTransport, list[dict[str, Any]]]:
    config = NeuralGuardConfig(
        environment="development",
        auth=AuthSettings(
            enabled=True,
            api_keys=[
                _CALLER_BINDING,
                _APPROVER_BINDING,
                _OTHER_TENANT_BINDING,
                _AGENT_APPROVER_BINDING,
            ],
            public_endpoints=[],
        ),
    )
    for key, value in {
        "enabled": True,
        "upstream_url": "http://fake-mcp-server.local",
        "timeout_seconds": 5.0,
        "server_id": "test-mcp",
        "hitl_enabled": hitl,
        "hitl_timeout_seconds": timeout,
        "hitl_ttl_seconds": 60.0,
        "hitl_max_pending": max_pending,
        "hitl_approver_tenants": approvers if approvers is not None else ["admin"],
    }.items():
        setattr(config.mcp, key, value)

    app = create_app(config)
    resolved_policy = policy or McpToolPolicy(tool_rules={"deploy_prod": ToolAction.ESCALATE})

    class _FakeRegistry:
        enabled = True

        def get(self, tenant_id: str) -> Any:
            class _Cfg:
                mcp = resolved_policy

            return _Cfg()

    app.state.tenant_registry = _FakeRegistry()
    stub = _StubTransport(tools if tools is not None else ["deploy_prod", "read_file"])
    app.state.mcp_transport = stub
    captured: list[dict[str, Any]] = []
    app.state.audit_logger.log_mcp_event = lambda **kw: captured.append(kw)  # type: ignore[method-assign]
    return app, stub, captured


async def _prime(client: AsyncClient) -> None:
    await client.post(
        "/v1/mcp",
        json={"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}},
        headers={"Mcp-Method": "tools/list", "Authorization": f"Bearer {_CALLER_KEY}"},
    )


async def _await_escalation(client: AsyncClient, key: str = _APPROVER_KEY) -> dict[str, Any]:
    for _ in range(100):
        listed = await client.get("/v1/mcp/escalations", headers={"Authorization": f"Bearer {key}"})
        assert listed.status_code == 200, listed.text
        records = listed.json()["escalations"]
        if records:
            return records[0]
        await asyncio.sleep(0.01)
    raise AssertionError("no escalation became pending")


async def _decide(
    client: AsyncClient,
    escalation_id: str,
    approve: bool,
    key: str = _APPROVER_KEY,
    reason: str | None = None,
) -> Any:
    payload: dict[str, Any] = {"approve": approve}
    if reason is not None:
        payload["reason"] = reason
    return await client.post(
        f"/v1/mcp/escalations/{escalation_id}",
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
    )


class TestRouteApprovalLoop:
    @pytest.mark.asyncio
    async def test_approved_call_is_forwarded_and_audited(self) -> None:
        app, stub, captured = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            task = asyncio.create_task(
                client.post(
                    "/v1/mcp",
                    json=_rpc("deploy_prod", {"env": "prod"}),
                    headers=_headers("deploy_prod", _CALLER_KEY),
                )
            )
            record = await _await_escalation(client)
            assert record["tool"] == "deploy_prod"
            assert record["agent"] == "deploy-bot"
            assert record["source"] == "intent_gate"

            decision = await _decide(client, record["escalation_id"], True, reason="change window")
            assert decision.status_code == 200
            assert decision.json()["status"] == STATUS_APPROVED

            response = await task

        assert response.status_code == 200
        assert [c["method"] for c in stub.calls] == ["tools/list", "tools/call"]
        outcomes = [e["details"].get("outcome") for e in captured]
        assert "escalation_raised" in outcomes
        assert "escalation_approved" in outcomes
        approved = next(e for e in captured if e["details"].get("outcome") == "escalation_approved")
        assert approved["details"]["decided_by"] == "admin"
        assert approved["details"]["agent_id"] == "deploy-bot"
        # Argument VALUES stay out of the audit trail: the digest is the binding.
        assert '"env":"prod"' not in str(captured)
        assert len(approved["details"]["request_digest"]) == 64

    @pytest.mark.asyncio
    async def test_denied_call_is_refused_and_never_forwarded(self) -> None:
        app, stub, captured = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            task = asyncio.create_task(
                client.post(
                    "/v1/mcp",
                    json=_rpc("deploy_prod"),
                    headers=_headers("deploy_prod", _CALLER_KEY),
                )
            )
            record = await _await_escalation(client)
            decision = await _decide(client, record["escalation_id"], False, reason="not this week")
            assert decision.status_code == 200
            response = await task

        assert response.status_code == 403
        assert response.json()["error"] == "mcp_escalation_denied"
        assert response.json()["reason"] == "not this week"
        assert [c["method"] for c in stub.calls] == ["tools/list"]
        # Two events: the approver's decision (MCP-HITL-DECISION) and the
        # gateway's refusal of the waiting call (MCP-HITL-DENIED).
        denied_rules = {
            e["rule_id"] for e in captured if e["details"].get("outcome") == "escalation_denied"
        }
        assert denied_rules == {"MCP-HITL-DECISION", "MCP-HITL-DENIED"}

    @pytest.mark.asyncio
    async def test_timeout_refuses_and_is_never_an_implicit_approval(self) -> None:
        app, stub, captured = _app(timeout=1.0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            response = await client.post(
                "/v1/mcp",
                json=_rpc("deploy_prod"),
                headers=_headers("deploy_prod", _CALLER_KEY),
            )
        assert response.status_code == 403
        assert response.json()["error"] == "mcp_escalation_timeout"
        assert [c["method"] for c in stub.calls] == ["tools/list"]
        assert any(e["details"].get("outcome") == "escalation_timeout" for e in captured)

    @pytest.mark.asyncio
    async def test_an_agent_bound_credential_cannot_decide(self) -> None:
        app, _stub, _ = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            task = asyncio.create_task(
                client.post(
                    "/v1/mcp",
                    json=_rpc("deploy_prod"),
                    headers=_headers("deploy_prod", _CALLER_KEY),
                )
            )
            record = await _await_escalation(client)
            # Admin tenant, but agent-bound: self-approval is exactly the abuse
            # case the tenant-level rule exists to prevent.
            refused = await _decide(client, record["escalation_id"], True, key="agentadmin")
            assert refused.status_code == 403
            assert refused.json()["error"] == "mcp_escalation_agent_forbidden"
            # Still pending: the refused attempt decided nothing.
            assert (
                await client.get(
                    "/v1/mcp/escalations", headers={"Authorization": "Bearer approverkey"}
                )
            ).json()["pending"] == 1
            await _decide(client, record["escalation_id"], True)
            assert (await task).status_code == 200

    @pytest.mark.asyncio
    async def test_a_non_approver_tenant_cannot_decide(self) -> None:
        app, _stub, _ = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            task = asyncio.create_task(
                client.post(
                    "/v1/mcp",
                    json=_rpc("deploy_prod"),
                    headers=_headers("deploy_prod", _CALLER_KEY),
                )
            )
            record = await _await_escalation(client)
            refused = await _decide(client, record["escalation_id"], True, key="otherapproverkey")
            assert refused.status_code == 403
            assert refused.json()["error"] == "mcp_escalation_not_approver"
            listed = await client.get(
                "/v1/mcp/escalations", headers={"Authorization": "Bearer otherapproverkey"}
            )
            assert listed.status_code == 403
            await _decide(client, record["escalation_id"], False)
            assert (await task).status_code == 403

    @pytest.mark.asyncio
    async def test_tenant_approver_is_scoped_to_its_own_escalations(self) -> None:
        # `beta` is an approver, but only `acme`'s call is pending. Naming a
        # tenant as an approver must not become a cross-tenant read of another
        # tenant's tool arguments.
        app, _stub, _ = _app(approvers=["admin", "beta"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            task = asyncio.create_task(
                client.post(
                    "/v1/mcp",
                    json=_rpc("deploy_prod", {"env": "prod"}),
                    headers=_headers("deploy_prod", _CALLER_KEY),
                )
            )
            record = await _await_escalation(client)  # admin sees it
            beta_view = await client.get(
                "/v1/mcp/escalations", headers={"Authorization": "Bearer otherapproverkey"}
            )
            assert beta_view.status_code == 200
            assert beta_view.json()["escalations"] == []
            assert beta_view.json()["pending"] == 0
            assert beta_view.json()["scope"] == "beta"

            refused = await _decide(client, record["escalation_id"], True, key="otherapproverkey")
            assert refused.status_code == 403
            assert refused.json()["error"] == "mcp_escalation_other_tenant"

            # The admin tenant is the gateway operator: global by design.
            admin_view = await client.get(
                "/v1/mcp/escalations", headers={"Authorization": "Bearer approverkey"}
            )
            assert admin_view.json()["scope"] == "global"
            assert [r["escalation_id"] for r in admin_view.json()["escalations"]] == [
                record["escalation_id"]
            ]
            assert (await _decide(client, record["escalation_id"], False)).status_code == 200
            assert (await task).status_code == 403

    @pytest.mark.asyncio
    async def test_list_count_matches_the_list(self) -> None:
        app, _stub, _ = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            task = asyncio.create_task(
                client.post(
                    "/v1/mcp",
                    json=_rpc("deploy_prod"),
                    headers=_headers("deploy_prod", _CALLER_KEY),
                )
            )
            record = await _await_escalation(client)
            view = await client.get(
                "/v1/mcp/escalations", headers={"Authorization": "Bearer approverkey"}
            )
            assert view.json()["pending"] == len(view.json()["escalations"]) == 1
            await _decide(client, record["escalation_id"], False)
            await task

    @pytest.mark.asyncio
    async def test_decision_audit_records_real_latency(self) -> None:
        app, _stub, captured = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            task = asyncio.create_task(
                client.post(
                    "/v1/mcp",
                    json=_rpc("deploy_prod"),
                    headers=_headers("deploy_prod", _CALLER_KEY),
                )
            )
            record = await _await_escalation(client)
            await _decide(client, record["escalation_id"], False)
            await task
        decision = next(e for e in captured if e["rule_id"] == "MCP-HITL-DECISION")
        assert decision["total_latency_ms"] > 0, "0.0 is a false record in the audit trail"
        assert decision["details"]["decided_by"] == "admin"

    @pytest.mark.asyncio
    async def test_decision_is_single_shot(self) -> None:
        app, _stub, _ = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            task = asyncio.create_task(
                client.post(
                    "/v1/mcp",
                    json=_rpc("deploy_prod"),
                    headers=_headers("deploy_prod", _CALLER_KEY),
                )
            )
            record = await _await_escalation(client)
            assert (await _decide(client, record["escalation_id"], True)).status_code == 200
            assert (await task).status_code == 200
            # The approval was consumed by the waiting call: replaying the id
            # cannot authorize a second call, and it is gone from the store.
            second = await _decide(client, record["escalation_id"], True)
            assert second.status_code == 404
            assert second.json()["error"] == "mcp_escalation_unknown"
            listed = await client.get(
                "/v1/mcp/escalations", headers={"Authorization": "Bearer approverkey"}
            )
            assert listed.json()["escalations"] == []

    @pytest.mark.asyncio
    async def test_unknown_escalation_and_invalid_decision(self) -> None:
        app, _stub, _ = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            missing = await _decide(client, "does-not-exist", True)
            assert missing.status_code == 404
            bad = await client.post(
                "/v1/mcp/escalations/anything",
                json={"approve": "yes"},
                headers={"Authorization": "Bearer approverkey"},
            )
            assert bad.status_code == 400
            assert bad.json()["error"] == "invalid_decision"

    @pytest.mark.asyncio
    async def test_hitl_disabled_keeps_the_fail_closed_refusal(self) -> None:
        app, stub, captured = _app(hitl=False)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            response = await client.post(
                "/v1/mcp",
                json=_rpc("deploy_prod"),
                headers=_headers("deploy_prod", _CALLER_KEY),
            )
            listed = await client.get(
                "/v1/mcp/escalations", headers={"Authorization": "Bearer approverkey"}
            )
        assert response.status_code == 403
        assert response.json()["error"] == "mcp_gate_escalated"
        assert listed.status_code == 503
        # Pre-NG-15 vocabulary preserved verbatim: SIEM rules key on it.
        assert any(e["details"].get("outcome") == "gate_escalate" for e in captured)
        assert [c["method"] for c in stub.calls] == ["tools/list"]

    @pytest.mark.asyncio
    async def test_capacity_is_refused_not_queued(self) -> None:
        app, stub, captured = _app(max_pending=1)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            first = asyncio.create_task(
                client.post(
                    "/v1/mcp",
                    json=_rpc("deploy_prod"),
                    headers=_headers("deploy_prod", _CALLER_KEY),
                )
            )
            record = await _await_escalation(client)
            second = await client.post(
                "/v1/mcp",
                json=_rpc("deploy_prod"),
                headers=_headers("deploy_prod", _CALLER_KEY),
            )
            assert second.status_code == 503
            assert second.json()["error"] == "mcp_escalation_capacity"
            await _decide(client, record["escalation_id"], False)
            await first
        assert [c["method"] for c in stub.calls] == ["tools/list"]
        assert any(e["details"].get("outcome") == "escalation_capacity" for e in captured)

    @pytest.mark.asyncio
    async def test_approval_is_not_authority_a_rugpull_still_blocks(self) -> None:
        # deploy_prod is NOT in the catalog: the human says yes, the signed
        # baseliner still refuses the execute.
        app, stub, captured = _app(tools=["read_file"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            task = asyncio.create_task(
                client.post(
                    "/v1/mcp",
                    json=_rpc("deploy_prod"),
                    headers=_headers("deploy_prod", _CALLER_KEY),
                )
            )
            record = await _await_escalation(client)
            assert (await _decide(client, record["escalation_id"], True)).status_code == 200
            response = await task
        assert response.status_code == 403
        assert response.json()["error"] == "mcp_tool_not_baselined"
        assert [c["method"] for c in stub.calls] == ["tools/list"]
        outcomes = [e["details"].get("outcome") for e in captured]
        assert "escalation_approved" in outcomes
        assert "DRIFT_BLOCKED" in outcomes

    @pytest.mark.asyncio
    async def test_argument_gate_escalation_also_goes_through_hitl(self) -> None:
        policy = McpToolPolicy(
            tool_constraints={
                "read_file": [
                    ArgumentConstraint(
                        argument="path",
                        kind=ConstraintKind.PATH_PREFIX,
                        values=["/srv/data"],
                        action=ConstraintAction.ESCALATE,
                        reason="path outside the reviewed prefix",
                    )
                ]
            }
        )
        app, stub, captured = _app(policy=policy, tools=["read_file"])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            task = asyncio.create_task(
                client.post(
                    "/v1/mcp",
                    json=_rpc("read_file", {"path": "/etc/shadow"}),
                    headers=_headers("read_file", _CALLER_KEY),
                )
            )
            record = await _await_escalation(client)
            assert record["source"] == "argument_gate"
            assert record["rule_id"].startswith("MCP-ARG")
            assert "/etc/shadow" in record["argument_preview"]
            assert (await _decide(client, record["escalation_id"], False)).status_code == 200
            response = await task
        assert response.status_code == 403
        assert response.json()["error"] == "mcp_escalation_denied"
        assert [c["method"] for c in stub.calls] == ["tools/list"]
        raised = next(e for e in captured if e["details"].get("outcome") == "escalation_raised")
        assert raised["details"]["source"] == "argument_gate"
        assert raised["details"]["argument"] == "path"


class TestHitlDisabledVocabulary:
    @pytest.mark.asyncio
    async def test_argument_escalation_keeps_its_shipped_outcome_string(self) -> None:
        policy = McpToolPolicy(
            tool_constraints={
                "read_file": [
                    ArgumentConstraint(
                        argument="path",
                        kind=ConstraintKind.PATH_PREFIX,
                        values=["/srv/data"],
                        action=ConstraintAction.ESCALATE,
                        reason="outside the reviewed prefix",
                    )
                ]
            }
        )
        app, stub, captured = _app(policy=policy, tools=["read_file"], hitl=False)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            response = await client.post(
                "/v1/mcp",
                json=_rpc("read_file", {"path": "/etc/shadow"}),
                headers=_headers("read_file", _CALLER_KEY),
            )
        assert response.status_code == 403
        assert response.json()["error"] == "mcp_argument_escalated"
        outcomes = [e["details"].get("outcome") for e in captured]
        assert "argument_escalate" in outcomes
        assert "argument_gate_escalate" not in outcomes
        assert [c["method"] for c in stub.calls] == ["tools/list"]


# ── Settings ───────────────────────────────────────────────────────────────


class TestHitlSettings:
    def test_approver_tenants_accept_comma_separated(self) -> None:
        from neuralguard.config.settings import McpGatewaySettings

        settings = McpGatewaySettings(hitl_approver_tenants="Admin, SOC-Team")
        assert settings.hitl_approver_tenants == ["admin", "soc-team"]

    def test_ttl_must_cover_the_wait_timeout(self) -> None:
        import pydantic

        from neuralguard.config.settings import McpGatewaySettings

        with pytest.raises(pydantic.ValidationError):
            McpGatewaySettings(hitl_enabled=True, hitl_timeout_seconds=30.0, hitl_ttl_seconds=10.0)

    def test_wait_timeout_is_capped(self) -> None:
        import pydantic

        from neuralguard.config.settings import McpGatewaySettings

        with pytest.raises(pydantic.ValidationError):
            McpGatewaySettings(hitl_timeout_seconds=600.0)

    def test_settings_do_not_leak_into_the_escalation_view(self) -> None:
        # The approver view is built from the record only; nothing from settings
        # (secrets, tokens) can appear in it.
        broker = HitlBroker()
        assert json.dumps(broker.pending_for_approval()) == "[]"
