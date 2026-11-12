"""NG-11: trusted workload identity and per-agent policy narrowing.

Identity comes from the credential (a ``key|tenant|agent`` binding or a JWT
``agent`` claim), never from a request header, and it can only ever narrow the
tenant's policy. These tests cover the identity plumbing, the resolution rules,
and the route wiring including audit attribution.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.auth.jwtauth import AuthRuntimeState, JwtManager
from neuralguard.auth.principal import Principal, normalize_agent
from neuralguard.config.settings import AuthSettings, NeuralGuardConfig
from neuralguard.main import create_app
from neuralguard.mcp.constraints import ArgumentConstraint, ConstraintKind
from neuralguard.mcp.intent_gate import (
    AgentToolPolicy,
    McpToolPolicy,
    ToolAction,
    evaluate_intent,
)

# ── Principal normalization ────────────────────────────────────────────────


class TestNormalizeAgent:
    def test_none_and_blank_are_no_identity(self) -> None:
        assert normalize_agent(None) is None
        assert normalize_agent("   ") is None

    def test_lowercased(self) -> None:
        assert normalize_agent("Support-Bot") == "support-bot"

    @pytest.mark.parametrize(
        "bad", ["-leading", "with space", "semi;colon", "quote'", "x" * 65, "ünïcode", "a/b"]
    )
    def test_invalid_ids_are_refused(self, bad: str) -> None:
        from neuralguard.auth.principal import InvalidPrincipalError

        with pytest.raises(InvalidPrincipalError):
            normalize_agent(bad)


# ── Credential bindings ────────────────────────────────────────────────────


class TestKeyBindings:
    def test_three_field_entry_binds_an_agent(self) -> None:
        settings = AuthSettings(api_keys="k1|acme|support-bot,k2|acme,k3")
        principals = settings.key_to_principal()
        assert principals["k1"] == Principal("acme", "support-bot")
        assert principals["k2"] == Principal("acme", None)
        assert principals["k3"] == Principal("default", None)

    def test_tenant_only_view_is_unchanged(self) -> None:
        settings = AuthSettings(api_keys="k1|acme|support-bot")
        assert settings.key_to_tenant() == {"k1": "acme"}

    def test_malformed_agent_fails_at_boot_not_silently(self) -> None:
        from neuralguard.auth.principal import InvalidPrincipalError

        settings = AuthSettings(api_keys="k1|acme|bad agent")
        with pytest.raises(InvalidPrincipalError):
            settings.key_to_principal()

    def test_runtime_state_resolves_principal(self) -> None:
        state = AuthRuntimeState(AuthSettings(api_keys="k1|acme|bot1"))
        assert state.resolve("k1") == Principal("acme", "bot1")
        assert state.lookup("k1") == "acme"  # tenant-only view preserved
        assert state.resolve("nope") is None


class TestJwtIdentity:
    def _manager(self) -> JwtManager:
        return JwtManager(
            AuthSettings(enabled=True, jwt_enabled=True, jwt_secret="s" * 40, api_keys="k|acme")
        )

    def test_agent_claim_round_trips(self) -> None:
        manager = self._manager()
        token = manager.issue("acme", agent="bot1")
        assert manager.verify_principal(token) == Principal("acme", "bot1")
        assert manager.verify(token) == "acme"

    def test_token_without_agent_is_tenant_level(self) -> None:
        manager = self._manager()
        assert manager.verify_principal(manager.issue("acme")) == Principal("acme", None)

    def test_malformed_agent_claim_invalidates_the_token(self) -> None:
        import jwt as pyjwt

        manager = self._manager()
        forged = pyjwt.encode(
            {
                "iss": "neuralguard",
                "sub": "tenant:acme",
                "tenant": "acme",
                "agent": "bad agent!",
            },
            "s" * 40,
            algorithm="HS256",
        )
        assert manager.verify_principal(forged) is None


# ── Policy resolution: narrowing only ──────────────────────────────────────


def _policy(**kw: Any) -> McpToolPolicy:
    return McpToolPolicy.model_validate(kw)


class TestAgentResolution:
    def test_without_agent_rules_identity_is_ignored(self) -> None:
        policy = _policy(tool_rules={"read_file": ToolAction.ALLOW})
        assert policy.action_for("tools/call", "read_file", "bot1") is ToolAction.ALLOW

    def test_agent_can_deny_what_the_tenant_allows(self) -> None:
        policy = _policy(
            tool_rules={"delete_file": ToolAction.ALLOW},
            agent_rules={"bot1": {"tool_rules": {"delete_file": ToolAction.DENY}}},
        )
        assert policy.action_for("tools/call", "delete_file", None) is ToolAction.ALLOW
        assert policy.action_for("tools/call", "delete_file", "bot1") is ToolAction.DENY

    def test_agent_cannot_widen_a_tenant_deny(self) -> None:
        policy = _policy(
            tool_rules={"delete_file": ToolAction.DENY},
            agent_rules={"bot1": {"tool_rules": {"delete_file": ToolAction.ALLOW}}},
        )
        assert policy.action_for("tools/call", "delete_file", "bot1") is ToolAction.DENY

    def test_agent_cannot_downgrade_escalate_to_allow(self) -> None:
        policy = _policy(
            tool_rules={"deploy": ToolAction.ESCALATE},
            agent_rules={"bot1": {"tool_rules": {"deploy": ToolAction.ALLOW}}},
        )
        assert policy.action_for("tools/call", "deploy", "bot1") is ToolAction.ESCALATE

    def test_unenumerated_caller_gets_default_agent_action(self) -> None:
        policy = _policy(
            agent_rules={"bot1": {"tool_rules": {"read_file": ToolAction.ALLOW}}},
            default_agent_action=ToolAction.DENY,
        )
        assert policy.action_for("tools/call", "read_file", "bot1") is ToolAction.ALLOW
        assert policy.action_for("tools/call", "read_file", None) is ToolAction.DENY
        assert policy.action_for("tools/call", "read_file", "unknown-bot") is ToolAction.DENY

    def test_default_agent_action_defaults_to_allow(self) -> None:
        """Adding rules for one bot must not lock out everyone else by default."""
        policy = _policy(agent_rules={"bot1": {"tool_rules": {"read_file": ToolAction.DENY}}})
        assert policy.action_for("tools/call", "write_file", None) is ToolAction.ALLOW

    def test_agent_rules_do_not_apply_to_non_tool_methods(self) -> None:
        policy = _policy(
            agent_rules={"bot1": {"tool_rules": {"tools/list": ToolAction.DENY}}},
            default_agent_action=ToolAction.DENY,
        )
        assert policy.action_for("tools/list", None, "bot1") is ToolAction.ALLOW

    def test_intent_gate_threads_the_identity(self) -> None:
        policy = _policy(
            agent_rules={"bot1": {"tool_rules": {"delete_file": ToolAction.DENY}}},
        )
        headers = {"mcp-method": "tools/call", "mcp-name": "delete_file"}
        assert evaluate_intent(headers, policy, agent="bot1").action is ToolAction.DENY
        assert evaluate_intent(headers, policy).action is ToolAction.ALLOW

    def test_effective_view_exposes_the_agent_layer(self) -> None:
        policy = _policy(agent_rules={"bot1": {"tool_rules": {"x": ToolAction.DENY}}})
        view = policy.to_effective_dict()
        assert view["agent_rules"]["bot1"]["tool_rules"] == {"x": "deny"}
        assert view["default_agent_action"] == "allow"


class TestAgentConstraints:
    def _constraints(self, *values: str) -> list[ArgumentConstraint]:
        return [
            ArgumentConstraint.model_validate(
                {"argument": "path", "kind": ConstraintKind.PATH_PREFIX, "values": list(values)}
            )
        ]

    def test_agent_adds_restrictions_on_top_of_the_tenant(self) -> None:
        from neuralguard.mcp.argument_gate import evaluate_arguments

        policy = _policy(
            tool_constraints={"read_file": self._constraints("/srv")},
            agent_rules={
                "bot1": {"tool_constraints": {"read_file": self._constraints("/srv/bot1-public")}}
            },
        )
        # Tenant-allowed but outside the agent's narrower prefix.
        assert (
            evaluate_arguments(policy, "read_file", {"path": "/srv/secret"}, "bot1").action
            is ToolAction.DENY
        )
        # Inside both.
        assert (
            evaluate_arguments(policy, "read_file", {"path": "/srv/bot1-public/x"}, "bot1").action
            is ToolAction.ALLOW
        )
        # The tenant-level caller keeps the tenant rule only.
        assert (
            evaluate_arguments(policy, "read_file", {"path": "/srv/secret"}).action
            is ToolAction.ALLOW
        )

    def test_agent_cannot_remove_a_tenant_constraint(self) -> None:
        from neuralguard.mcp.argument_gate import evaluate_arguments

        policy = _policy(
            tool_constraints={"read_file": self._constraints("/srv")},
            agent_rules={"bot1": {"tool_constraints": {}}},
        )
        assert (
            evaluate_arguments(policy, "read_file", {"path": "/etc/passwd"}, "bot1").action
            is ToolAction.DENY
        )


# ── Route wiring ───────────────────────────────────────────────────────────


class _StubTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.tokens: list[str | None] = []

    async def forward(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        upstream_token: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(payload)
        self.tokens.append(upstream_token)
        if payload.get("method") == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        {
                            "name": "read_file",
                            "description": "Read a file",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                },
            }
        return {"jsonrpc": "2.0", "id": 1, "result": {"content": []}}


def _config() -> NeuralGuardConfig:
    # Auth is constructed (not mutated) so the api_keys parser runs — assigning
    # a comma-separated string onto an existing AuthSettings would leave a str
    # where key_to_principal expects a list.
    config = NeuralGuardConfig(
        environment="development",
        auth=AuthSettings(
            enabled=True,
            api_keys=["botkey|acme|support-bot", "tenantkey|acme"],
            public_endpoints=[],
            jwt_enabled=True,
            jwt_secret="s" * 40,
        ),
    )
    for key, value in {
        "enabled": True,
        "upstream_url": "http://fake-mcp-server.local",
        "timeout_seconds": 5.0,
        "server_id": "test-mcp",
    }.items():
        setattr(config.mcp, key, value)
    return config


def _registry(policy: McpToolPolicy) -> Any:
    class _FakeRegistry:
        enabled = True

        def get(self, tenant_id: str) -> Any:
            class _Cfg:
                mcp = policy

            return _Cfg()

    return _FakeRegistry()


def _call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }


def _app(policy: McpToolPolicy) -> tuple[Any, _StubTransport, list[dict[str, Any]]]:
    app = create_app(_config())
    stub = _StubTransport()
    app.state.mcp_transport = stub
    app.state.tenant_registry = _registry(policy)
    captured: list[dict[str, Any]] = []
    app.state.audit_logger.log_mcp_event = lambda **kw: captured.append(kw)  # type: ignore[method-assign]
    return app, stub, captured


class TestAgentRoute:
    @pytest.mark.asyncio
    async def test_agent_bound_key_is_denied_by_its_own_rule(self) -> None:
        app, stub, _ = _app(
            McpToolPolicy(
                agent_rules={"support-bot": {"tool_rules": {"read_file": ToolAction.DENY}}}
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            r = await client.post(
                "/v1/mcp",
                json=_call("read_file", {"path": "/srv/x"}),
                headers={
                    "Mcp-Method": "tools/call",
                    "Mcp-Name": "read_file",
                    "Authorization": "Bearer botkey",
                },
            )
        assert r.status_code == 403
        assert r.json()["rule_id"] == "MCP-GATE-DENY-001"
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_tenant_key_is_not_denied_by_the_agent_rule(self) -> None:
        app, _stub, _ = _app(
            McpToolPolicy(
                agent_rules={"support-bot": {"tool_rules": {"read_file": ToolAction.DENY}}}
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await client.post(
                "/v1/mcp",
                json={"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}},
                headers={"Mcp-Method": "tools/list", "Authorization": "Bearer tenantkey"},
            )
            r = await client.post(
                "/v1/mcp",
                json=_call("read_file", {"path": "/srv/x"}),
                headers={
                    "Mcp-Method": "tools/call",
                    "Mcp-Name": "read_file",
                    "Authorization": "Bearer tenantkey",
                },
            )
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_agent_header_cannot_impersonate(self) -> None:
        """A caller-supplied agent header must not change enforcement."""
        app, stub, _ = _app(
            McpToolPolicy(
                agent_rules={"support-bot": {"tool_rules": {"read_file": ToolAction.ALLOW}}},
                default_agent_action=ToolAction.DENY,
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            r = await client.post(
                "/v1/mcp",
                json=_call("read_file", {"path": "/srv/x"}),
                headers={
                    "Mcp-Method": "tools/call",
                    "Mcp-Name": "read_file",
                    "Authorization": "Bearer tenantkey",
                    # The spoof attempt: no agent binding on this key.
                    "Mcp-Agent": "support-bot",
                    "X-Agent-Id": "support-bot",
                },
            )
        assert r.status_code == 403, "header-supplied identity must not be trusted"
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_audit_records_the_agent_and_null_for_tenant_keys(self) -> None:
        app, _stub, captured = _app(McpToolPolicy())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await client.post(
                "/v1/mcp",
                json=_call("read_file", {"path": "/srv/x"}),
                headers={
                    "Mcp-Method": "tools/call",
                    "Mcp-Name": "read_file",
                    "Authorization": "Bearer botkey",
                },
            )
            await client.post(
                "/v1/mcp",
                json=_call("read_file", {"path": "/srv/x"}),
                headers={
                    "Mcp-Method": "tools/call",
                    "Mcp-Name": "read_file",
                    "Authorization": "Bearer tenantkey",
                },
            )
        agents = [event["details"]["agent_id"] for event in captured]
        assert "support-bot" in agents
        assert None in agents, "tenant-level events must be explicit about having no agent"

    @pytest.mark.asyncio
    async def test_token_exchange_carries_the_binding(self) -> None:
        app = create_app(_config())
        app.state.mcp_transport = _StubTransport()
        app.state.tenant_registry = _registry(McpToolPolicy())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            r = await client.post(
                "/v1/auth/token",
                json={},
                headers={"Authorization": "Bearer botkey"},
            )
            assert r.status_code == 200
            assert r.json()["agent"] == "support-bot"

            tenant_level = await client.post(
                "/v1/auth/token", json={}, headers={"Authorization": "Bearer tenantkey"}
            )
            assert tenant_level.json()["agent"] is None
