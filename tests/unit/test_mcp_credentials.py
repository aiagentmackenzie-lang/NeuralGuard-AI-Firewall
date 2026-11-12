"""NG-12: per-tool upstream credentials.

The property under test is not "a header is set" but **which** credential a call
carries, and that the failure modes fail closed: a mapped-but-missing secret is
refused rather than falling back to the broad default, and a caller cannot
supply its own upstream token.
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
                    ]
                },
            }
        return {"jsonrpc": "2.0", "id": 1, "result": {"content": []}}


def _app(**mcp: Any) -> tuple[Any, _StubTransport, list[dict[str, Any]]]:
    config = NeuralGuardConfig(environment="development")
    for key, value in {
        "enabled": True,
        "upstream_url": "http://fake-mcp-server.local",
        "timeout_seconds": 5.0,
        "server_id": "test-mcp",
        **mcp,
    }.items():
        setattr(config.mcp, key, value)
    # setattr bypasses validation, so re-validate the credential mapping the
    # same way boot would.
    config.mcp.tool_credentials = {
        tool: validate_reference(ref) for tool, ref in config.mcp.tool_credentials.items()
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


async def _prime(client: AsyncClient) -> None:
    await client.post(
        "/v1/mcp",
        json={"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}},
        headers={"Mcp-Method": "tools/list"},
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
