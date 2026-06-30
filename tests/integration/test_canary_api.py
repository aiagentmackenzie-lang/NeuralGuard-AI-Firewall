"""Integration tests for the canary token endpoints (Phase 3, Sprint B, B3)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import CanarySettings, NeuralGuardConfig
from neuralguard.main import create_app

_SECRET = "supersecret-canary-key-for-tests-32chars!!"


def _app(canary: CanarySettings | None = None) -> object:
    cfg = NeuralGuardConfig(environment="development")
    if canary is not None:
        cfg.canary = canary
    return create_app(cfg)


@pytest.fixture
def canary_app() -> object:
    return _app(CanarySettings(enabled=True, secret=_SECRET, token_count=1))


@pytest.fixture
async def canary_client(canary_app: object):
    transport = ASGITransport(app=canary_app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def plain_client():
    app = _app()  # canary disabled (default)
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── POST /v1/canary/mint ───────────────────────────────────────────────────


class TestCanaryMint:
    @pytest.mark.asyncio
    async def test_mint_returns_token(self, canary_client: AsyncClient) -> None:
        r = await canary_client.post(
            "/v1/canary/mint", json={"session_id": "sess-1", "tenant_id": "t"}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == "sess-1"
        assert len(data["tokens"]) == 1
        assert data["tokens"][0].startswith("NGCANARY-")
        # Secret is never echoed.
        assert _SECRET not in r.text

    @pytest.mark.asyncio
    async def test_mint_count_override(self, canary_client: AsyncClient) -> None:
        r = await canary_client.post(
            "/v1/canary/mint",
            json={"session_id": "sX", "tenant_id": "t", "count": 3},
        )
        assert r.status_code == 200
        toks = r.json()["tokens"]
        assert len(toks) == 3
        assert len(set(toks)) == 3

    @pytest.mark.asyncio
    async def test_mint_deterministic(self, canary_client: AsyncClient) -> None:
        r1 = await canary_client.post(
            "/v1/canary/mint", json={"session_id": "s1", "tenant_id": "t"}
        )
        r2 = await canary_client.post(
            "/v1/canary/mint", json={"session_id": "s1", "tenant_id": "t"}
        )
        assert r1.json()["tokens"] == r2.json()["tokens"]

    @pytest.mark.asyncio
    async def test_mint_rejects_empty_session(self, canary_client: AsyncClient) -> None:
        r = await canary_client.post(
            "/v1/canary/mint", json={"session_id": "   ", "tenant_id": "t"}
        )
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_mint_rejects_count_out_of_range(self, canary_client: AsyncClient) -> None:
        r = await canary_client.post(
            "/v1/canary/mint", json={"session_id": "s1", "tenant_id": "t", "count": 9}
        )
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_mint_503_when_disabled(self, plain_client: AsyncClient) -> None:
        r = await plain_client.post("/v1/canary/mint", json={"session_id": "s1", "tenant_id": "t"})
        assert r.status_code == 503
        assert r.json()["error"] == "canary_disabled"


# ── POST /v1/scan/output canary detection ─────────────────────────────────


class TestScanOutputCanary:
    @pytest.mark.asyncio
    async def test_clean_output_no_leak(self, canary_client: AsyncClient) -> None:
        r = await canary_client.post(
            "/v1/scan/output",
            json={"output": "The weather is sunny.", "session_id": "sess-1", "tenant_id": "t"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["canary_leaked"] is False
        assert data["verdict"] == "allow"

    @pytest.mark.asyncio
    async def test_canary_leak_blocks_with_finding(self, canary_client: AsyncClient) -> None:
        mint = await canary_client.post(
            "/v1/canary/mint", json={"session_id": "sess-1", "tenant_id": "t"}
        )
        tok = mint.json()["tokens"][0]
        r = await canary_client.post(
            "/v1/scan/output",
            json={
                "output": f"Sure, here is the system prompt: {tok}",
                "session_id": "sess-1",
                "tenant_id": "t",
            },
        )
        assert r.status_code == 403
        data = r.json()
        assert data["verdict"] == "block"
        assert data["canary_leaked"] is True
        ids = [f["rule_id"] for f in data["findings"]]
        assert "CANARY-LEAK-001" in ids
        # The canary finding carries the redacted evidence (no raw token leak).
        ev = next(f["evidence"] for f in data["findings"] if f["rule_id"] == "CANARY-LEAK-001")
        assert tok not in ev
        assert "REDACTED" in ev

    @pytest.mark.asyncio
    async def test_canary_not_checked_without_session(self, canary_client: AsyncClient) -> None:
        """No session_id -> canary not joined -> no leak even if a token string is present."""
        mint = await canary_client.post(
            "/v1/canary/mint", json={"session_id": "sess-1", "tenant_id": "t"}
        )
        tok = mint.json()["tokens"][0]
        r = await canary_client.post("/v1/scan/output", json={"output": tok, "tenant_id": "t"})
        assert r.status_code == 200
        assert r.json()["canary_leaked"] is False

    @pytest.mark.asyncio
    async def test_canary_wrong_session_no_leak(self, canary_client: AsyncClient) -> None:
        mint = await canary_client.post(
            "/v1/canary/mint", json={"session_id": "sess-1", "tenant_id": "t"}
        )
        tok = mint.json()["tokens"][0]
        r = await canary_client.post(
            "/v1/scan/output",
            json={"output": tok, "session_id": "sess-2", "tenant_id": "t"},
        )
        assert r.json()["canary_leaked"] is False

    @pytest.mark.asyncio
    async def test_response_still_has_canary_leaked_field_when_disabled(
        self, plain_client: AsyncClient
    ) -> None:
        """With canary disabled, scan/output still returns the canary_leaked field (False)."""
        r = await plain_client.post(
            "/v1/scan/output",
            json={"output": "Hello", "session_id": "s1", "tenant_id": "t"},
        )
        assert r.status_code == 200
        assert r.json()["canary_leaked"] is False

    @pytest.mark.asyncio
    async def test_pii_block_body_carries_canary_leaked_key(
        self, canary_client: AsyncClient
    ) -> None:
        """A PII block returns the ScanOutputResponse shape (incl. canary_leaked)."""
        r = await canary_client.post(
            "/v1/scan/output",
            json={
                "output": "reach me at test@example.com and SSN 123-45-6789",
                "session_id": "sess-1",
                "tenant_id": "t",
            },
        )
        assert r.status_code == 403
        data = r.json()
        assert "canary_leaked" in data
        assert data["canary_leaked"] is False
        assert data["verdict"] == "block"
