"""NG-12 / NG-14: upstream credentials selected per tool and per agent.

The property under test is not "a header is set" but **which** credential a
call carries, and that the failure modes fail closed: a mapped-but-missing
secret is refused rather than falling back to the broad default, an
agent-scoped tool never degrades to the default for a caller it has no entry
for, and a caller cannot supply its own upstream token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import NeuralGuardConfig
from neuralguard.main import create_app
from neuralguard.mcp.credentials import (
    CredentialError,
    ToolCredentialResolver,
    split_mapping_key,
    validate_mapping_key,
    validate_reference,
)
from neuralguard.mcp.transport import McpTransport

# ── Reference validation ───────────────────────────────────────────────────


class TestReferenceValidation:
    def test_env_reference_accepted_and_trimmed(self) -> None:
        assert validate_reference("  env:READONLY_TOKEN ") == "env:READONLY_TOKEN"

    def test_file_reference_accepted(self) -> None:
        assert validate_reference("file:/run/secrets/ro") == "file:/run/secrets/ro"

    @pytest.mark.parametrize(
        "bad",
        [
            "plain-secret-value",  # a secret pasted where a reference belongs
            "env:",  # no var name
            "env:NAME=1",  # env assignment syntax
            "env:has space",
            "file:",  # no path
            "",
        ],
    )
    def test_malformed_references_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError):
            validate_reference(bad)

    def test_settings_reject_a_bad_reference_at_load(self) -> None:
        from pydantic import ValidationError

        from neuralguard.config.settings import McpGatewaySettings

        with pytest.raises(ValidationError):
            McpGatewaySettings(tool_credentials={"read_file": "not-a-reference"})


# ── NG-14: mapping keys (agent|tool) ───────────────────────────────────────


class TestMappingKeys:
    def test_tool_wide_key_has_no_agent(self) -> None:
        assert split_mapping_key("read_file") == (None, "read_file")

    def test_agent_key_is_split_and_normalized(self) -> None:
        assert split_mapping_key(" Billing-Bot | refund ") == ("billing-bot", "refund")
        assert validate_mapping_key("Billing-Bot|refund") == "billing-bot|refund"

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   ",
            "|refund",  # empty agent half
            "billing-bot|",  # empty tool half
            "a|b|refund",  # two separators
            "bad agent|refund",  # agent charset
            "-leading|refund",  # agent must start alphanumeric
            "agent|two words",  # whitespace in a tool name
        ],
    )
    def test_malformed_keys_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError):
            split_mapping_key(bad)

    def test_settings_reject_a_bad_key_at_load(self) -> None:
        from pydantic import ValidationError

        from neuralguard.config.settings import McpGatewaySettings

        with pytest.raises(ValidationError):
            McpGatewaySettings(tool_credentials={"|refund": "env:RO"})

    def test_settings_reject_a_duplicate_after_normalization(self) -> None:
        from pydantic import ValidationError

        from neuralguard.config.settings import McpGatewaySettings

        with pytest.raises(ValidationError):
            McpGatewaySettings(tool_credentials={"Bot|refund": "env:A", "bot|refund": "env:B"})

    def test_settings_normalize_keys_and_references(self) -> None:
        from neuralguard.config.settings import McpGatewaySettings

        settings = McpGatewaySettings(tool_credentials={" Bot | refund ": " env:REFUND_BOT "})
        assert settings.tool_credentials == {"bot|refund": "env:REFUND_BOT"}


# ── Resolver ───────────────────────────────────────────────────────────────


class TestResolver:
    def test_mapped_tool_gets_its_own_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NG_TEST_RO", "ro-token")
        resolver = ToolCredentialResolver(
            {"read_file": "env:NG_TEST_RO"}, default_token="broad-token"
        )
        resolved = resolver.for_tool("read_file")
        assert resolved.token == "ro-token"
        assert resolved.source == "env:NG_TEST_RO"

    def test_unmapped_tool_inherits_the_default(self) -> None:
        resolver = ToolCredentialResolver({}, default_token="broad-token")
        assert resolver.for_tool("anything").token == "broad-token"

    def test_no_default_means_no_header(self) -> None:
        resolver = ToolCredentialResolver({}, default_token="")
        assert resolver.for_tool("anything").token == ""

    def test_mapped_but_missing_denies_and_never_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NG_TEST_ABSENT", raising=False)
        resolver = ToolCredentialResolver(
            {"admin_sql": "env:NG_TEST_ABSENT"}, default_token="broad-token"
        )
        with pytest.raises(CredentialError):
            resolver.for_tool("admin_sql")

    def test_file_reference_reads_and_strips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = tmp_path / "token"
        secret.write_text("  file-token\n")
        resolver = ToolCredentialResolver({"x": f"file:{secret}"})
        assert resolver.for_tool("x").token == "file-token"

    def test_unreadable_file_denies_without_leaking_contents(self, tmp_path: Path) -> None:
        resolver = ToolCredentialResolver({"x": f"file:{tmp_path / 'absent'}"})
        with pytest.raises(CredentialError) as excinfo:
            resolver.for_tool("x")
        assert "absent" in str(excinfo.value)

    def test_oversized_secret_denied(self, tmp_path: Path) -> None:
        secret = tmp_path / "big"
        secret.write_text("x" * 5000)
        resolver = ToolCredentialResolver({"x": f"file:{secret}"})
        with pytest.raises(CredentialError):
            resolver.for_tool("x")

    def test_require_tool_credential_refuses_unmapped(self) -> None:
        resolver = ToolCredentialResolver({}, default_token="broad", require_tool_credential=True)
        with pytest.raises(CredentialError):
            resolver.for_tool("read_file")

    def test_unresolved_reports_names_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NG_TEST_GONE", raising=False)
        resolver = ToolCredentialResolver({"a": "env:NG_TEST_GONE"})
        assert resolver.unresolved() == ["env:NG_TEST_GONE"]

    def test_source_labels_never_contain_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NG_TEST_SECRET", "super-secret-value")
        resolver = ToolCredentialResolver({"a": "env:NG_TEST_SECRET"}, default_token="d")
        assert resolver.source_for("a") == "env:NG_TEST_SECRET"
        assert "super-secret-value" not in resolver.source_for("a")


# ── NG-14: per-(agent, tool) resolution ────────────────────────────────────


class TestAgentScopedResolution:
    def test_agent_credential_beats_the_tool_wide_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NG_REFUND_READONLY", "shared-readonly")
        monkeypatch.setenv("NG_REFUND_BILLING", "billing-only")
        resolver = ToolCredentialResolver(
            {"refund": "env:NG_REFUND_READONLY", "billing-bot|refund": "env:NG_REFUND_BILLING"}
        )
        assert resolver.for_tool("refund", agent="billing-bot").token == "billing-only"
        # Every other agent still gets the tool-wide credential.
        assert resolver.for_tool("refund", agent="support-bot").token == "shared-readonly"
        assert resolver.for_tool("refund").token == "shared-readonly"

    def test_agent_scoped_tool_never_degrades_to_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NG_BILLING_REFUND", "billing-token")
        resolver = ToolCredentialResolver(
            {"billing-bot|refund": "env:NG_BILLING_REFUND"}, default_token="broad-token"
        )
        # A caller with no agent identity is NOT silently handed the broad token.
        with pytest.raises(CredentialError) as excinfo:
            resolver.for_tool("refund")
        assert "agent-scoped" in str(excinfo.value)
        with pytest.raises(CredentialError):
            resolver.for_tool("refund", agent="support-bot")
        assert "broad-token" not in str(excinfo.value)

    def test_agent_scoped_missing_secret_denies_that_agent_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NG_BILLING_GONE", raising=False)
        monkeypatch.setenv("NG_SUPPORT_REFUND", "support-token")
        resolver = ToolCredentialResolver(
            {
                "billing-bot|refund": "env:NG_BILLING_GONE",
                "support-bot|refund": "env:NG_SUPPORT_REFUND",
            },
            default_token="broad-token",
        )
        with pytest.raises(CredentialError):
            resolver.for_tool("refund", agent="billing-bot")
        # The other agent's credential is unaffected — no cross-agent leak.
        assert resolver.for_tool("refund", agent="support-bot").token == "support-token"

    def test_source_marks_an_unmapped_agent_scoped_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NG_BILLING_REFUND", "t")
        resolver = ToolCredentialResolver(
            {"billing-bot|refund": "env:NG_BILLING_REFUND"}, default_token="broad"
        )
        assert resolver.source_for("refund", agent="billing-bot") == "env:NG_BILLING_REFUND"
        # Audit must show why, never the default reference it did not use.
        assert resolver.source_for("refund", agent="support-bot") == "agent-scoped-unmapped"
        assert resolver.source_for("refund") == "agent-scoped-unmapped"

    def test_introspection_lists_agent_scoped_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NG_A", "a")
        monkeypatch.setenv("NG_B", "b")
        resolver = ToolCredentialResolver(
            {
                "read_file": "env:NG_A",
                "bot|refund": "env:NG_B",
                "bot|delete_db": "env:NG_B",
            }
        )
        assert resolver.agent_scoped_tools() == ["delete_db", "refund"]
        assert resolver.is_agent_scoped("refund") is True
        assert resolver.is_agent_scoped("read_file") is False
        assert resolver.is_mapped("refund") is False
        assert resolver.is_mapped("refund", agent="bot") is True

    def test_unresolved_covers_agent_scoped_references(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NG_SCOPED_GONE", raising=False)
        resolver = ToolCredentialResolver({"bot|refund": "env:NG_SCOPED_GONE"})
        assert resolver.unresolved() == ["env:NG_SCOPED_GONE"]

    def test_require_mapped_still_applies_to_unmapped_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NG_ONLY_REFUND", "t")
        resolver = ToolCredentialResolver(
            {"bot|refund": "env:NG_ONLY_REFUND"},
            default_token="broad",
            require_tool_credential=True,
        )
        with pytest.raises(CredentialError):
            resolver.for_tool("read_file", agent="bot")


# ── Transport ──────────────────────────────────────────────────────────────


def _transport_with_config(token: str) -> tuple[McpTransport, list[dict[str, str]]]:
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({k.lower(): v for k, v in request.headers.items()})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    config = NeuralGuardConfig(environment="development")
    config.mcp.upstream_url = "http://fake.local"
    config.mcp.upstream_auth_token = token
    transport = McpTransport(config.mcp)
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return transport, seen


class TestTransportCredentialSelection:
    @pytest.mark.asyncio
    async def test_explicit_token_overrides_the_default(self) -> None:
        transport, seen = _transport_with_config("default-token")
        await transport.forward({"method": "tools/call"}, upstream_token="per-tool-token")
        assert seen[0]["authorization"] == "Bearer per-tool-token"
        await transport.aclose()

    @pytest.mark.asyncio
    async def test_none_uses_the_configured_default(self) -> None:
        transport, seen = _transport_with_config("default-token")
        await transport.forward({"method": "tools/list"})
        assert seen[0]["authorization"] == "Bearer default-token"
        await transport.aclose()

    @pytest.mark.asyncio
    async def test_empty_token_sends_no_auth_header(self) -> None:
        transport, seen = _transport_with_config("default-token")
        await transport.forward({"method": "tools/call"}, upstream_token="")
        assert "authorization" not in seen[0]
        await transport.aclose()

    @pytest.mark.asyncio
    async def test_caller_authorization_header_never_wins(self) -> None:
        transport, seen = _transport_with_config("default-token")
        await transport.forward(
            {"method": "tools/call"},
            headers={"authorization": "Bearer attacker-token", "X-Trace": "1"},
            upstream_token="per-tool-token",
        )
        assert seen[0]["authorization"] == "Bearer per-tool-token"
        assert seen[0]["x-trace"] == "1"
        await transport.aclose()


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
                        {"name": "read_file", "description": "r", "inputSchema": {}},
                        {"name": "admin_sql", "description": "a", "inputSchema": {}},
                        {"name": "refund", "description": "f", "inputSchema": {}},
                    ]
                },
            }
        return {"jsonrpc": "2.0", "id": 1, "result": {"content": []}}


def _app(**mcp: Any) -> tuple[Any, _StubTransport, list[dict[str, Any]]]:
    config = NeuralGuardConfig(environment="development")
    return _assemble(config, mcp)


def _app_with_auth(
    api_keys: list[str], **mcp: Any
) -> tuple[Any, _StubTransport, list[dict[str, Any]]]:
    """Gateway app whose API keys bind NG-11 workload agents (``key|tenant|agent``)."""
    from neuralguard.config.settings import AuthSettings

    config = NeuralGuardConfig(
        environment="development",
        auth=AuthSettings(enabled=True, api_keys=api_keys, public_endpoints=[]),
    )
    return _assemble(config, mcp)


def _assemble(
    config: NeuralGuardConfig, mcp: dict[str, Any]
) -> tuple[Any, _StubTransport, list[dict[str, Any]]]:
    for key, value in {
        "enabled": True,
        "upstream_url": "http://fake-mcp-server.local",
        "timeout_seconds": 5.0,
        "server_id": "test-mcp",
        **mcp,
    }.items():
        setattr(config.mcp, key, value)
    # setattr bypasses validation, so re-validate the mapping the way boot would
    # (NG-14 keys included: 'tool' or 'agent|tool').
    config.mcp.tool_credentials = {
        validate_mapping_key(map_key): validate_reference(ref)
        for map_key, ref in config.mcp.tool_credentials.items()
    }
    app = create_app(config)
    stub = _StubTransport()
    app.state.mcp_transport = stub
    captured: list[dict[str, Any]] = []
    app.state.audit_logger.log_mcp_event = lambda **kw: captured.append(kw)  # type: ignore[method-assign]
    return app, stub, captured


def _call(tool: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": {}},
    }


async def _prime(client: AsyncClient, key: str | None = None) -> None:
    headers = {"Mcp-Method": "tools/list"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    await client.post(
        "/v1/mcp",
        json={"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}},
        headers=headers,
    )


class TestRouteCredentials:
    @pytest.mark.asyncio
    async def test_mapped_tool_carries_its_credential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NG_ROUTE_RO", "ro-secret")
        app, stub, _ = _app(
            upstream_auth_token="broad-secret",
            tool_credentials={"read_file": "env:NG_ROUTE_RO"},
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            r = await client.post(
                "/v1/mcp",
                json=_call("read_file"),
                headers={"Mcp-Method": "tools/call", "Mcp-Name": "read_file"},
            )
        assert r.status_code == 200
        assert stub.tokens[-1] == "ro-secret", "the mapped credential must win"

    @pytest.mark.asyncio
    async def test_unmapped_tool_uses_the_default(self) -> None:
        app, stub, _ = _app(upstream_auth_token="broad-secret")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            await client.post(
                "/v1/mcp",
                json=_call("admin_sql"),
                headers={"Mcp-Method": "tools/call", "Mcp-Name": "admin_sql"},
            )
        assert stub.tokens[-1] == "broad-secret"

    @pytest.mark.asyncio
    async def test_missing_secret_refuses_the_call_503(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NG_ROUTE_ABSENT", raising=False)
        app, stub, captured = _app(
            upstream_auth_token="broad-secret",
            tool_credentials={"admin_sql": "env:NG_ROUTE_ABSENT"},
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            r = await client.post(
                "/v1/mcp",
                json=_call("admin_sql"),
                headers={"Mcp-Method": "tools/call", "Mcp-Name": "admin_sql"},
            )
        assert r.status_code == 503
        assert r.json()["error"] == "mcp_credential_unavailable"
        # Only the tools/list prime reached the transport; the refused call did
        # not, and it certainly did not fall back to the broad default token.
        assert [c["method"] for c in stub.calls] == ["tools/list"]
        event = captured[-1]
        assert event["rule_id"] == "MCP-CRED-001"
        assert "broad-secret" not in str(event)

    @pytest.mark.asyncio
    async def test_require_mapped_refuses_unmapped_tool(self) -> None:
        app, stub, _ = _app(upstream_auth_token="broad-secret", require_tool_credential=True)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            r = await client.post(
                "/v1/mcp",
                json=_call("read_file"),
                headers={"Mcp-Method": "tools/call", "Mcp-Name": "read_file"},
            )
        assert r.status_code == 503
        assert stub.calls[-1]["method"] == "tools/list"

    @pytest.mark.asyncio
    async def test_catalog_methods_still_use_the_default(self) -> None:
        app, stub, _ = _app(
            upstream_auth_token="broad-secret",
            tool_credentials={"read_file": "env:NG_UNUSED"},
            require_tool_credential=True,
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
        assert stub.tokens == [None], "catalog methods leave the token to the transport default"

    @pytest.mark.asyncio
    async def test_audit_records_the_credential_source_not_the_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NG_ROUTE_AUDIT", "audit-secret-value")
        app, _stub, captured = _app(tool_credentials={"read_file": "env:NG_ROUTE_AUDIT"})
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client)
            await client.post(
                "/v1/mcp",
                json=_call("read_file"),
                headers={"Mcp-Method": "tools/call", "Mcp-Name": "read_file"},
            )
        forwarded = [e for e in captured if e["details"].get("outcome") == "forwarded"]
        assert forwarded[-1]["details"]["credential_source"] == "env:NG_ROUTE_AUDIT"
        assert "audit-secret-value" not in str(captured)


# ── NG-14 route wiring: identity selects the upstream principal ────────────

_BILLING_BINDING = "billingkey|acme|billing-bot"
_SUPPORT_BINDING = "supportkey|acme|support-bot"
_TENANT_BINDING = "tenantkey|acme"
_BILLING_KEY = "billingkey"
_SUPPORT_KEY = "supportkey"
_TENANT_KEY = "tenantkey"


async def _post_call(client: AsyncClient, tool: str, key: str) -> httpx.Response:
    return await client.post(
        "/v1/mcp",
        json=_call(tool),
        headers={
            "Mcp-Method": "tools/call",
            "Mcp-Name": tool,
            "Authorization": f"Bearer {key}",
        },
    )


class TestRouteAgentScopedCredentials:
    @pytest.mark.asyncio
    async def test_each_agent_carries_its_own_upstream_principal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NG_R_BILLING", "billing-secret")
        monkeypatch.setenv("NG_R_SUPPORT", "support-secret")
        app, stub, _ = _app_with_auth(
            [_BILLING_BINDING, _SUPPORT_BINDING],
            upstream_auth_token="broad-secret",
            tool_credentials={
                "billing-bot|refund": "env:NG_R_BILLING",
                "support-bot|refund": "env:NG_R_SUPPORT",
            },
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client, _BILLING_KEY)
            assert (await _post_call(client, "refund", _BILLING_KEY)).status_code == 200
            assert (await _post_call(client, "refund", _SUPPORT_KEY)).status_code == 200
        # The prime used the transport default; the two calls used their agents'.
        assert stub.tokens == [None, "billing-secret", "support-secret"]

    @pytest.mark.asyncio
    async def test_unmapped_agent_on_an_agent_scoped_tool_is_refused_not_shared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NG_R_BILLING2", "billing-secret")
        app, stub, captured = _app_with_auth(
            [_BILLING_BINDING, _SUPPORT_BINDING, _TENANT_BINDING],
            upstream_auth_token="broad-secret",
            tool_credentials={"billing-bot|refund": "env:NG_R_BILLING2"},
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client, _BILLING_KEY)
            r_support = await _post_call(client, "refund", _SUPPORT_KEY)
            r_tenant = await _post_call(client, "refund", _TENANT_KEY)
            r_billing = await _post_call(client, "refund", _BILLING_KEY)
        for refused in (r_support, r_tenant):
            assert refused.status_code == 503
            assert refused.json()["error"] == "mcp_credential_unavailable"
        assert r_billing.status_code == 200
        # Only the tools/list prime and the mapped agent's call reached upstream:
        # nothing fell through to the broad default.
        assert [c["method"] for c in stub.calls] == ["tools/list", "tools/call"]
        assert stub.tokens == [None, "billing-secret"]
        denied = [e for e in captured if e["details"].get("outcome") == "credential_unavailable"]
        assert len(denied) == 2
        assert all(e["rule_id"] == "MCP-CRED-001" for e in denied)
        assert all(e["details"]["credential_source"] == "agent-scoped-unmapped" for e in denied)
        # The denial reason names the tool and the agent, never a secret.
        assert all("refund" in e["details"]["reason"] for e in denied)
        assert "broad-secret" not in str(captured)
        assert "billing-secret" not in str(captured)

    @pytest.mark.asyncio
    async def test_audit_attributes_the_scoped_reference_to_the_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NG_R_BILLING3", "billing-secret")
        app, _stub, captured = _app_with_auth(
            [_BILLING_BINDING],
            tool_credentials={"billing-bot|refund": "env:NG_R_BILLING3"},
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client, _BILLING_KEY)
            await _post_call(client, "refund", _BILLING_KEY)
        forwarded = [e for e in captured if e["details"].get("outcome") == "forwarded"]
        assert forwarded[-1]["details"]["agent_id"] == "billing-bot"
        assert forwarded[-1]["details"]["credential_source"] == "env:NG_R_BILLING3"
        assert "billing-secret" not in str(captured)

    @pytest.mark.asyncio
    async def test_tool_wide_entry_still_covers_every_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NG_R_SHARED", "shared-secret")
        monkeypatch.setenv("NG_R_BILLING4", "billing-secret")
        app, stub, _ = _app_with_auth(
            [_BILLING_BINDING, _SUPPORT_BINDING],
            tool_credentials={
                "refund": "env:NG_R_SHARED",
                "billing-bot|refund": "env:NG_R_BILLING4",
            },
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            await _prime(client, _BILLING_KEY)
            await _post_call(client, "refund", _BILLING_KEY)
            await _post_call(client, "refund", _SUPPORT_KEY)
        assert stub.tokens == [None, "billing-secret", "shared-secret"]
