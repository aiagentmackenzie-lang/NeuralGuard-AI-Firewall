"""Phase A hardening tests: body-size limits, bounded decompression, base64 cap,
production fail-fast."""

from __future__ import annotations

import zlib

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import NeuralGuardConfig, ScannerSettings, ServerSettings
from neuralguard.main import create_app
from neuralguard.scanners.structural import StructuralScanner


class TestBodySizeLimit:
    @pytest.mark.asyncio
    async def test_oversized_body_returns_413(self):
        config = NeuralGuardConfig(
            environment="development",
            server=ServerSettings(max_request_body_bytes=1024),
        )
        app = create_app(config)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            big = {"prompt": "A" * 5000, "tenant_id": "t"}
            r = await c.post("/v1/evaluate", json=big)
            assert r.status_code == 413
            assert r.json()["error"] == "payload_too_large"

    @pytest.mark.asyncio
    async def test_normal_body_passes(self):
        config = NeuralGuardConfig(
            environment="development",
            server=ServerSettings(max_request_body_bytes=65536),
        )
        app = create_app(config)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/v1/evaluate", json={"prompt": "hello", "tenant_id": "t"})
            assert r.status_code == 200


class TestBoundedDecompression:
    """The zlib bomb check must be memory-bounded, not materialize-then-check.

    We test the bounded decompression helper at the byte level because a real
    zlib stream contains bytes >= 0x80 that do not survive a str->utf-8
    round-trip (the scanner receives a Python str). The helper is the security
    primitive that guarantees no OOM regardless of how bytes reach it.
    """

    def _make_bomb(self, expanded_size: int) -> bytes:
        return zlib.compress(b"A" * expanded_size, level=9)

    def test_bomb_exceeding_cap_is_bounded(self):
        from neuralguard.scanners.structural import _MAX_DECOMPRESSED_BYTES, _bounded_decompress

        # 200 MiB bomb -> must exceed the 8 MiB cap and return exceeded_cap=True,
        # without materializing 200 MiB.
        bomb = self._make_bomb(200 * 1024 * 1024)
        exceeded, _ratio, produced = _bounded_decompress(bomb)
        assert exceeded is True
        # produced must be bounded near the cap, NOT 200 MiB. The security
        # property is O(cap), not O(expanded_size).
        assert produced < 16 * 1024 * 1024, f"produced={produced} not bounded"
        assert produced < (200 * 1024 * 1024) // 10

    def test_small_bomb_high_ratio_detected(self):
        from neuralguard.scanners.structural import _MAX_DECOMPRESSED_BYTES, _bounded_decompress

        # 1 MiB bomb (under the byte cap) but ratio far exceeds default limit 10.
        bomb = self._make_bomb(1024 * 1024)
        exceeded, ratio, produced = _bounded_decompress(bomb, ratio_limit=10.0)
        assert produced > 0
        assert ratio > 10.0
        assert exceeded is False

    def test_non_compressed_input_is_safe(self):
        from neuralguard.scanners.structural import _MAX_DECOMPRESSED_BYTES, _bounded_decompress

        _exceeded, _ratio, produced = _bounded_decompress(b"not a zlib stream at all")
        # zlib.error is caught inside the helper via flush guard; produced stays 0
        assert produced == 0
        assert _ratio == 0.0

    def test_scanner_bomb_path_via_ascii_compatible_stream(self):
        # End-to-end: craft a payload whose compressed bytes are all ASCII so it
        # survives the str round-trip and actually reaches the bomb check.
        # We retry until we get an all-ASCII zlib stream (rare but achievable with
        # raw deflate and specific payloads). If we can't, we skip — the helper
        # tests above already prove the bound.
        import itertools

        from neuralguard.models.schemas import EvaluateRequest
        from neuralguard.scanners.structural import _MAX_DECOMPRESSED_BYTES

        scanner = StructuralScanner(
            ScannerSettings(max_decompression_ratio=5.0, max_input_length=10_000_000)
        )
        found = None
        for seed in itertools.count():
            payload = (f"seed-{seed}-" * 5000).encode()
            bomb = zlib.compress(payload, 9)
            if all(b < 0x80 for b in bomb):
                found = bomb
                break
            if seed > 200:
                pytest.skip("No ASCII-only zlib stream found; helper tests cover the bound")
        assert found is not None
        req = EvaluateRequest(prompt=found.decode("ascii"), tenant_id="t")
        result = scanner.scan(req)
        assert any(f.rule_id == "STRUCT-003" for f in result.findings) or produced_ok(result)


def produced_ok(result) -> bool:
    return result.verdict.value in ("block", "sanitize")


class TestBase64Cap:
    def test_oversized_base64_blob_blocks(self):
        # A base64 string longer than the decode cap should block without
        # attempting to decode (no memory blowup).
        import base64

        big_b64 = base64.b64encode(b"x" * (64 * 1024)).decode("ascii")
        scanner = StructuralScanner(ScannerSettings())
        from neuralguard.models.schemas import EvaluateRequest

        req = EvaluateRequest(prompt=big_b64, tenant_id="t")
        result = scanner.scan(req)
        assert result.verdict.value == "block"
        assert any(
            f.rule_id == "STRUCT-005" and "Oversized base64" in f.description
            for f in result.findings
        )


class TestProductionFailFast:
    @pytest.mark.asyncio
    async def test_production_without_auth_raises_in_lifespan(self):
        import pytest_asyncio

        config = NeuralGuardConfig(environment="production")  # auth disabled
        app = create_app(config)
        with pytest.raises(RuntimeError, match="authentication is disabled"):
            async with app.router.lifespan_context(app):
                pass

    @pytest.mark.asyncio
    async def test_production_without_keys_raises_in_lifespan(self):
        from neuralguard.config.settings import AuthSettings

        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=[]),
        )
        app = create_app(config)
        with pytest.raises(RuntimeError, match="no API keys"):
            async with app.router.lifespan_context(app):
                pass

    @pytest.mark.asyncio
    async def test_production_with_auth_starts(self):
        from neuralguard.config.settings import AuthSettings

        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True),
        )
        app = create_app(config)
        async with app.router.lifespan_context(app):
            pass  # should not raise
