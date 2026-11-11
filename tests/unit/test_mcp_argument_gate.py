"""NG-10: argument / resource constraints for MCP tool calls.

Two layers are tested here: the pure evaluator (no HTTP, no upstream) and the
route wiring (constraint denied => 403, nothing forwarded, no argument value in
the audit trail).
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import NeuralGuardConfig
from neuralguard.main import create_app
from neuralguard.mcp.argument_gate import evaluate_arguments
from neuralguard.mcp.constraints import ArgumentConstraint, ConstraintAction, ConstraintKind
from neuralguard.mcp.intent_gate import McpToolPolicy, ToolAction


def _rule(
    argument: str,
    kind: ConstraintKind | str,
    values: list[str] | None = None,
    action: ConstraintAction | str = ConstraintAction.DENY,
    reason: str = "",
) -> ArgumentConstraint:
    return ArgumentConstraint.model_validate(
        {
            "argument": argument,
            "kind": kind,
            "values": values or [],
            "action": action,
            "reason": reason,
        }
    )


def _policy(tool: str, *rules: ArgumentConstraint) -> McpToolPolicy:
    return McpToolPolicy(tool_constraints={tool: list(rules)})


# ── Pure evaluator ─────────────────────────────────────────────────────────


class TestNoConstraints:
    def test_unconstrained_tool_allows(self) -> None:
        decision = evaluate_arguments(McpToolPolicy(), "read_file", {"path": "/etc/passwd"})
        assert decision.action is ToolAction.ALLOW

    def test_other_tools_are_unaffected(self) -> None:
        policy = _policy("read_file", _rule("path", ConstraintKind.PATH_PREFIX, ["/srv"]))
        assert evaluate_arguments(policy, "list_files", {"path": "/etc"}).action is ToolAction.ALLOW


class TestRequiredAndAllowlist:
    def test_required_missing_denies(self) -> None:
        d = evaluate_arguments(_policy("t", _rule("path", ConstraintKind.REQUIRED)), "t", {})
        assert d.action is ToolAction.DENY
        assert d.rule_id == "MCP-ARG-001"

    def test_required_empty_string_denies(self) -> None:
        d = evaluate_arguments(
            _policy("t", _rule("path", ConstraintKind.REQUIRED)), "t", {"path": "   "}
        )
        assert d.action is ToolAction.DENY

    def test_required_with_value_passes(self) -> None:
        d = evaluate_arguments(
            _policy("t", _rule("path", ConstraintKind.REQUIRED)), "t", {"path": "/x"}
        )
        assert d.action is ToolAction.ALLOW

    def test_key_allowlist_rejects_extra_keys_and_names_them_only(self) -> None:
        policy = _policy("t", _rule("", ConstraintKind.KEY_ALLOWLIST, ["path", "encoding"]))
        d = evaluate_arguments(policy, "t", {"path": "/x", "callback_url": "https://evil.test"})
        assert d.action is ToolAction.DENY
        assert d.rule_id == "MCP-ARG-002"
        assert "callback_url" in d.reason
        assert "evil.test" not in d.reason  # key names only, never values

    def test_key_allowlist_passes_when_keys_are_declared(self) -> None:
        policy = _policy("t", _rule("", ConstraintKind.KEY_ALLOWLIST, ["path"]))
        assert evaluate_arguments(policy, "t", {"path": "/x"}).action is ToolAction.ALLOW


class TestScalarKinds:
    def test_enum_allows_listed_and_denies_other(self) -> None:
        policy = _policy("t", _rule("mode", ConstraintKind.ENUM, ["read", "list"]))
        assert evaluate_arguments(policy, "t", {"mode": "read"}).action is ToolAction.ALLOW
        denied = evaluate_arguments(policy, "t", {"mode": "write"})
        assert denied.action is ToolAction.DENY and denied.rule_id == "MCP-ARG-003"

    def test_max_length_boundary(self) -> None:
        policy = _policy("t", _rule("q", ConstraintKind.MAX_LENGTH, ["5"]))
        assert evaluate_arguments(policy, "t", {"q": "12345"}).action is ToolAction.ALLOW
        assert evaluate_arguments(policy, "t", {"q": "123456"}).action is ToolAction.DENY

    def test_max_length_misconfigured_denies(self) -> None:
        policy = _policy("t", _rule("q", ConstraintKind.MAX_LENGTH, []))
        d = evaluate_arguments(policy, "t", {"q": "x"})
        assert d.action is ToolAction.DENY

    def test_regex_deny_matches(self) -> None:
        policy = _policy("t", _rule("cmd", ConstraintKind.REGEX_DENY, [r"(?i)\brm\s+-rf\b"]))
        assert evaluate_arguments(policy, "t", {"cmd": "ls"}).action is ToolAction.ALLOW
        d = evaluate_arguments(policy, "t", {"cmd": "rm -rf /"})
        assert d.action is ToolAction.DENY and d.rule_id == "MCP-ARG-005"

    def test_invalid_deny_pattern_fails_closed(self) -> None:
        policy = _policy("t", _rule("cmd", ConstraintKind.REGEX_DENY, ["("]))
        assert evaluate_arguments(policy, "t", {"cmd": "x"}).action is ToolAction.DENY

    def test_nested_dotted_path(self) -> None:
        policy = _policy("t", _rule("options.format", ConstraintKind.ENUM, ["json"]))
        assert evaluate_arguments(policy, "t", {"options": {"format": "json"}}).action is (
            ToolAction.ALLOW
        )
        assert evaluate_arguments(policy, "t", {"options": {"format": "xml"}}).action is (
            ToolAction.DENY
        )

    def test_absent_optional_argument_is_vacuous(self) -> None:
        policy = _policy("t", _rule("path", ConstraintKind.PATH_PREFIX, ["/srv"]))
        assert evaluate_arguments(policy, "t", {}).action is ToolAction.ALLOW

    def test_non_string_where_scalar_expected_denies(self) -> None:
        """A list/dict smuggled where a path is expected must not skip the check."""
        policy = _policy("t", _rule("path", ConstraintKind.PATH_PREFIX, ["/srv"]))
        d = evaluate_arguments(policy, "t", {"path": ["/srv/ok", "/etc/passwd"]})
        assert d.action is ToolAction.DENY and d.rule_id == "MCP-ARG-009"

    def test_boolean_is_not_a_string(self) -> None:
        policy = _policy("t", _rule("path", ConstraintKind.PATH_PREFIX, ["/srv"]))
        assert evaluate_arguments(policy, "t", {"path": True}).action is ToolAction.DENY


class TestPathConstraints:
    def test_prefix_inside_allows(self) -> None:
        policy = _policy("t", _rule("path", ConstraintKind.PATH_PREFIX, ["/srv/data"]))
        assert evaluate_arguments(policy, "t", {"path": "/srv/data/a.txt"}).action is (
            ToolAction.ALLOW
        )

    def test_prefix_boundary_is_not_string_prefix(self) -> None:
        """``/srv/data-evil`` must not pass a ``/srv/data`` prefix."""
        policy = _policy("t", _rule("path", ConstraintKind.PATH_PREFIX, ["/srv/data"]))
        d = evaluate_arguments(policy, "t", {"path": "/srv/data-evil/secret"})
        assert d.action is ToolAction.DENY and d.rule_id == "MCP-ARG-006"

    def test_prefix_equality_allows(self) -> None:
        policy = _policy("t", _rule("path", ConstraintKind.PATH_PREFIX, ["/srv/data"]))
        assert evaluate_arguments(policy, "t", {"path": "/srv/data"}).action is ToolAction.ALLOW

    @pytest.mark.parametrize(
        "path",
        [
            "/srv/data/../../etc/passwd",
            "../../etc/passwd",
            "..",
            "/srv/data/../secret",
            "/srv/data\\..\\etc",
            "/srv/data/\x00/x",
        ],
    )
    def test_traversal_is_refused(self, path: str) -> None:
        policy = _policy("t", _rule("path", ConstraintKind.PATH_PREFIX, ["/srv/data"]))
        assert evaluate_arguments(policy, "t", {"path": path}).action is ToolAction.DENY

    def test_glob_allows_only_matching(self) -> None:
        policy = _policy("t", _rule("path", ConstraintKind.PATH_GLOB, ["/srv/**/*.json"]))
        assert evaluate_arguments(policy, "t", {"path": "/srv/a/b.json"}).action is ToolAction.ALLOW
        assert evaluate_arguments(policy, "t", {"path": "/srv/a/b.yaml"}).action is ToolAction.DENY


class TestUrlConstraints:
    def test_allowed_host_and_subdomain(self) -> None:
        policy = _policy("t", _rule("url", ConstraintKind.URL_HOST, ["api.internal.example"]))
        assert evaluate_arguments(
            policy, "t", {"url": "https://api.internal.example/x"}
        ).action is (ToolAction.ALLOW)
        assert evaluate_arguments(
            policy, "t", {"url": "http://v1.api.internal.example/x"}
        ).action is (ToolAction.ALLOW)

    def test_lookalike_host_denied(self) -> None:
        policy = _policy("t", _rule("url", ConstraintKind.URL_HOST, ["internal.example"]))
        d = evaluate_arguments(policy, "t", {"url": "https://internal.example.evil.test"})
        assert d.action is ToolAction.DENY

    def test_userinfo_trick_resolves_to_actual_host(self) -> None:
        policy = _policy("t", _rule("url", ConstraintKind.URL_HOST, ["allowed.example"]))
        d = evaluate_arguments(policy, "t", {"url": "https://allowed.example@evil.test/steal"})
        assert d.action is ToolAction.DENY

    @pytest.mark.parametrize(
        "url", ["file:///etc/passwd", "gopher://allowed.example/", "https://", "not-a-url"]
    )
    def test_non_http_or_hostless_denied(self, url: str) -> None:
        policy = _policy("t", _rule("url", ConstraintKind.URL_HOST, ["allowed.example"]))
        assert evaluate_arguments(policy, "t", {"url": url}).action is ToolAction.DENY


class TestResolutionAndMalformedInput:
    def test_arguments_must_be_an_object(self) -> None:
        policy = _policy("t", _rule("path", ConstraintKind.REQUIRED))
        d = evaluate_arguments(policy, "t", ["/etc/passwd"])
        assert d.action is ToolAction.DENY and d.rule_id == "MCP-ARG-010"

    def test_none_arguments_denied_when_constrained(self) -> None:
        policy = _policy("t", _rule("path", ConstraintKind.REQUIRED))
        assert evaluate_arguments(policy, "t", None).action is ToolAction.DENY

    def test_most_restrictive_wins_regardless_of_order(self) -> None:
        escalate = _rule("path", ConstraintKind.REQUIRED, action=ConstraintAction.ESCALATE)
        deny = _rule("path", ConstraintKind.MAX_LENGTH, ["1"])
        for rules in ([escalate, deny], [deny, escalate]):
            d = evaluate_arguments(_policy("t", *rules), "t", {"path": "too long"})
            assert d.action is ToolAction.DENY, "DENY must beat ESCALATE in any order"

    def test_escalate_action_is_preserved(self) -> None:
        policy = _policy(
            "t", _rule("path", ConstraintKind.PATH_PREFIX, ["/srv"], ConstraintAction.ESCALATE)
        )
        assert evaluate_arguments(policy, "t", {"path": "/etc"}).action is ToolAction.ESCALATE

    def test_custom_reason_surfaces(self) -> None:
        policy = _policy(
            "t",
            _rule("path", ConstraintKind.PATH_PREFIX, ["/srv"], reason="tenant data is /srv only"),
        )
        d = evaluate_arguments(policy, "t", {"path": "/etc"})
        assert "tenant data is /srv only" in d.reason


# ── Route wiring ───────────────────────────────────────────────────────────


class _StubTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def forward(
        self, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        self.calls.append(payload)
        if payload.get("method") == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        {
                            "name": "read_file",
                            "description": "Read a file from disk",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        }
                    ]
                },
            }
        return {"jsonrpc": "2.0", "id": 1, "result": {"content": []}}


def _config() -> NeuralGuardConfig:
    config = NeuralGuardConfig(environment="development")
    for key, value in {
        "enabled": True,
        "upstream_url": "http://fake-mcp-server.local",
        "timeout_seconds": 5.0,
        "server_id": "test-mcp",
    }.items():
        setattr(config.mcp, key, value)
    return config


def _call(tool: str, arguments: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }


def _registry(policy: McpToolPolicy) -> Any:
    class _FakeRegistry:
        enabled = True

        def get(self, tenant_id: str) -> Any:
            class _Cfg:
                mcp = policy

            return _Cfg()

    return _FakeRegistry()


class TestArgumentGateRoute:
    @pytest.mark.asyncio
    async def test_violation_blocks_before_forward(self) -> None:
        app = create_app(_config())
        stub = _StubTransport()
        app.state.mcp_transport = stub
        app.state.tenant_registry = _registry(
            _policy("read_file", _rule("path", ConstraintKind.PATH_PREFIX, ["/srv/data"]))
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            r = await client.post(
                "/v1/mcp",
                json=_call("read_file", {"path": "/etc/passwd"}),
                headers={"Mcp-Method": "tools/call", "Mcp-Name": "read_file"},
            )
        assert r.status_code == 403
        body = r.json()
        assert body["error"] == "mcp_argument_denied"
        assert body["rule_id"] == "MCP-ARG-006"
        assert body["argument"] == "path"
        assert r.headers["X-NeuralGuard-Verdict"] == "block"
        assert stub.calls == [], "a constraint violation must never reach the upstream"

    @pytest.mark.asyncio
    async def test_audit_records_rule_and_path_but_never_the_value(self) -> None:
        app = create_app(_config())
        app.state.mcp_transport = _StubTransport()
        app.state.tenant_registry = _registry(
            _policy("read_file", _rule("path", ConstraintKind.PATH_PREFIX, ["/srv/data"]))
        )
        captured: list[dict[str, Any]] = []
        app.state.audit_logger.log_mcp_event = lambda **kw: captured.append(kw)  # type: ignore[method-assign]

        secret_path = "/etc/very-secret-tenant-path"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await client.post(
                "/v1/mcp",
                json=_call("read_file", {"path": secret_path}),
                headers={"Mcp-Method": "tools/call", "Mcp-Name": "read_file"},
            )

        assert captured, "a denied call must be audited"
        event = captured[-1]
        assert event["rule_id"] == "MCP-ARG-006"
        assert event["details"]["argument"] == "path"
        assert secret_path not in str(event), "audit must not carry the argument value"

    @pytest.mark.asyncio
    async def test_allowed_arguments_are_forwarded(self) -> None:
        app = create_app(_config())
        stub = _StubTransport()
        app.state.mcp_transport = stub
        app.state.tenant_registry = _registry(
            _policy(
                "read_file",
                _rule("path", ConstraintKind.PATH_PREFIX, ["/srv/data"]),
                _rule("path", ConstraintKind.REQUIRED),
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            # Prime the signed catalog baseline so the rug-pull layer is not the
            # thing under test here (it runs after the argument gate).
            await client.post(
                "/v1/mcp",
                json={"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}},
                headers={"Mcp-Method": "tools/list"},
            )
            r = await client.post(
                "/v1/mcp",
                json=_call("read_file", {"path": "/srv/data/report.txt"}),
                headers={"Mcp-Method": "tools/call", "Mcp-Name": "read_file"},
            )
        assert r.status_code == 200
        assert stub.calls[-1]["method"] == "tools/call"

    @pytest.mark.asyncio
    async def test_escalate_constraint_refuses_fail_closed(self) -> None:
        app = create_app(_config())
        app.state.mcp_transport = _StubTransport()
        app.state.tenant_registry = _registry(
            _policy(
                "read_file",
                _rule("path", ConstraintKind.PATH_PREFIX, ["/srv"], ConstraintAction.ESCALATE),
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            r = await client.post(
                "/v1/mcp",
                json=_call("read_file", {"path": "/etc/passwd"}),
                headers={"Mcp-Method": "tools/call", "Mcp-Name": "read_file"},
            )
        assert r.status_code == 403
        assert r.json()["error"] == "mcp_argument_escalated"

    @pytest.mark.asyncio
    async def test_header_gate_still_runs_first(self) -> None:
        """A denied tool is refused on headers, before arguments are consulted."""
        app = create_app(_config())
        stub = _StubTransport()
        app.state.mcp_transport = stub
        app.state.tenant_registry = _registry(
            McpToolPolicy(
                tool_rules={"read_file": ToolAction.DENY},
                tool_constraints={"read_file": [_rule("path", ConstraintKind.REQUIRED)]},
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            r = await client.post(
                "/v1/mcp",
                json=_call("read_file", {"path": "/srv/ok"}),
                headers={"Mcp-Method": "tools/call", "Mcp-Name": "read_file"},
            )
        assert r.json()["rule_id"] == "MCP-GATE-DENY-001"
        assert stub.calls == []


class TestTenantFileShape:
    """The documented YAML shape must load into the policy the route consults."""

    def test_yaml_shaped_constraints_load_and_enforce(self) -> None:
        from neuralguard.tenants.config import TenantConfig

        cfg = TenantConfig.model_validate(
            {
                "tenant_id": "acme",
                "mcp": {
                    "tool_rules": {"delete_file": "deny"},
                    "tool_constraints": {
                        "read_file": [
                            {"argument": "path", "kind": "required"},
                            {
                                "argument": "path",
                                "kind": "path_prefix",
                                "values": ["/srv/tenant-data"],
                                "reason": "tenant data lives under /srv/tenant-data",
                            },
                        ]
                    },
                },
            }
        )
        assert cfg.mcp is not None
        policy = cfg.mcp
        assert evaluate_arguments(
            policy, "read_file", {"path": "/srv/tenant-data/a.csv"}
        ).action is (ToolAction.ALLOW)
        denied = evaluate_arguments(policy, "read_file", {"path": "/etc/shadow"})
        assert denied.action is ToolAction.DENY
        assert "tenant data lives under" in denied.reason

    def test_unknown_constraint_key_is_rejected_at_load(self) -> None:
        import pydantic

        from neuralguard.tenants.config import TenantConfig

        with pytest.raises(pydantic.ValidationError):
            TenantConfig.model_validate(
                {
                    "tenant_id": "acme",
                    "mcp": {
                        "tool_constraints": {
                            "read_file": [{"argument": "path", "kind": "path_prefix", "bogus": 1}]
                        }
                    },
                }
            )
