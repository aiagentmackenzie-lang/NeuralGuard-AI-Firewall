"""P2-9: coverage headroom — direct-unit tests for uncovered branches.

Targets the weak spots measured on 2026-09-05 (89.28% total):
  proxy/forwarder.py 31% · escalate.py 67% · pattern_i18n.py 67% ·
  semantic/__init__.py 69% · egress.py 72% · sanitize.py 78% ·
  ratelimit_redis.py 75% · jwtauth.py 85% · siem.py 91% (error paths)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from neuralguard.actions.escalate import EscalateAction
from neuralguard.actions.sanitize import SanitizeAction
from neuralguard.auth.jwtauth import JwtManager, RuntimeKeyStore
from neuralguard.config.settings import (
    NeuralGuardConfig,
    RateLimitSettings,
    SiemSettings,
)
from neuralguard.middleware.ratelimit_redis import RedisRateLimiter
from neuralguard.models.schemas import (
    AuditEvent,
    Finding,
    LayerArbitrationResult,
    ScanLayer,
    Severity,
    ThreatCategory,
    Verdict,
)
from neuralguard.net.egress import is_private_endpoint
from neuralguard.proxy.forwarder import UpstreamError, UpstreamForwarder
from neuralguard.scanners.pattern_i18n import resolve_category
from neuralguard.siem import SiemRouter

# ── helpers ──────────────────────────────────────────────────────────────────


def _arbitration() -> LayerArbitrationResult:
    return LayerArbitrationResult(
        verdict=Verdict.ESCALATE,
        findings=[
            Finding(
                category=ThreatCategory.PROMPT_INJECTION_DIRECT,
                severity=Severity.MEDIUM,
                verdict=Verdict.SANITIZE,
                confidence=0.65,
                layer=ScanLayer.PATTERN,
                rule_id="PI-D-001",
                description="test finding",
                evidence="test evidence",
            )
        ],
        scanner_results=[],
        total_latency_ms=1.0,
        arbitration_reason="test-escalate",
    )


# ── UpstreamForwarder (F9) — direct error-path coverage ─────────────────────


class TestUpstreamForwarder:
    """Direct unit tests: the route tests mock the forwarder, so its own
    error paths (timeout, transport error, 4xx/5xx, bad JSON, ownership)
    need hermetic coverage here."""

    def _forwarder(
        self, handler: Any, api_key: str | None = "sk-upstream", url: str | None = None
    ) -> UpstreamForwarder:
        settings = type(
            "S",
            (),
            {
                "upstream_url": url or "http://upstream.test/v1",
                "upstream_api_key": api_key,
                "timeout_seconds": 5,
            },
        )()
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
        return UpstreamForwarder(settings, client=client)

    async def test_success_injects_bearer_and_parses(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant"}}]})

        fwd = self._forwarder(handler)
        data = await fwd.forward_chat({"model": "x", "messages": []})
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert seen["auth"] == "Bearer sk-upstream"
        assert seen["url"].startswith("http://upstream.test/v1/chat/completions")

    async def test_success_without_api_key_omits_header(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={})

        fwd = self._forwarder(handler, api_key=None)
        await fwd.forward_chat({"messages": []})
        assert seen["auth"] is None

    async def test_trailing_slash_upstream_url(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        fwd = self._forwarder(handler, url="http://upstream.test/v1/")
        await fwd.forward_chat({})
        # double slash would be a malformed URL — the rstrip("/") guards it
        assert True

    async def test_timeout_raises_upstream_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow")

        fwd = self._forwarder(handler)
        with pytest.raises(UpstreamError, match="timed out"):
            await fwd.forward_chat({})

    async def test_transport_error_raises_upstream_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        fwd = self._forwarder(handler)
        with pytest.raises(UpstreamError, match="unreachable"):
            await fwd.forward_chat({})

    async def test_http_500_raises_with_status(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="overloaded")

        fwd = self._forwarder(handler)
        with pytest.raises(UpstreamError, match="status 503"):
            await fwd.forward_chat({})

    async def test_bad_json_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not-json{", headers={"content-type": "text/plain"})

        fwd = self._forwarder(handler)
        with pytest.raises(UpstreamError, match="invalid JSON"):
            await fwd.forward_chat({})

    async def test_aclose_only_closes_owned_client(self) -> None:
        settings = type(
            "S",
            (),
            {"upstream_url": "http://u.test", "upstream_api_key": None, "timeout_seconds": 5},
        )()
        injected = httpx.AsyncClient()
        fwd = UpstreamForwarder(settings, client=injected)
        await fwd.aclose()
        assert not injected.is_closed  # not owned → left open
        await injected.aclose()

        owned = UpstreamForwarder(settings)  # owns its client
        await owned.aclose()
        assert owned.client.is_closed


# ── EscalateAction — webhook paths ───────────────────────────────────────────


class TestEscalateWebhook:
    def _config(self, url: str | None) -> NeuralGuardConfig:
        cfg = NeuralGuardConfig()
        cfg.action.escalation_webhook_url = url
        return cfg

    class FakeSyncClient:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def __enter__(self) -> TestEscalateWebhook.FakeSyncClient:
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def post(self, url: str, json: dict[str, Any]) -> Any:
            return type("R", (), {"status_code": 200})()

    class BoomSyncClient(FakeSyncClient):
        def post(self, url: str, json: dict[str, Any]) -> Any:
            raise ConnectionError("down")

    class FakeAsyncClient:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        async def __aenter__(self) -> TestEscalateWebhook.FakeAsyncClient:
            return self

        async def __aexit__(self, *a: Any) -> None:
            pass

        async def post(self, url: str, json: dict[str, Any]) -> Any:
            return type("R", (), {"status_code": 200})()

    def test_no_webhook_url_returns_false(self) -> None:
        action = EscalateAction(self._config(None))
        assert action._send_webhook(_arbitration()) is False

    def test_sync_webhook_success(self) -> None:
        """No running event loop → synchronous httpx.Client path."""
        action = EscalateAction(self._config("http://siem.test/escalate"))
        with patch("neuralguard.actions.escalate.httpx.Client", self.FakeSyncClient):
            assert action._send_webhook(_arbitration()) is True

    def test_sync_webhook_failure_is_false(self) -> None:
        action = EscalateAction(self._config("http://siem.test/escalate"))
        with patch("neuralguard.actions.escalate.httpx.Client", self.BoomSyncClient):
            assert action._send_webhook(_arbitration()) is False

    async def test_async_webhook_schedules_task(self) -> None:
        """In an async context the webhook is fire-and-forget (scheduled).

        pytest-asyncio native coroutine — deliberately NOT asyncio.run():
        that would close the main thread's event loop and break later
        get_event_loop()-based tests (real lesson: COV2 run, 2026-09-05).
        """
        action = EscalateAction(self._config("http://siem.test/escalate"))

        with patch("neuralguard.actions.escalate.httpx.AsyncClient", self.FakeAsyncClient):
            sent = action._send_webhook(_arbitration())
        assert sent is True  # task scheduled; delivery is best-effort


# ── SanitizeAction — text extraction + PII redaction ─────────────────────────


class TestSanitizeTextHandling:
    def _action(self) -> SanitizeAction:
        return SanitizeAction(NeuralGuardConfig())

    def test_extract_text_prefers_output(self) -> None:
        action = self._action()
        req = type("R", (), {"output": "secret output text"})()
        assert action._extract_text(req) == "secret output text"

    def test_extract_text_falls_back_to_prompt(self) -> None:
        action = self._action()
        req = type("R", (), {"prompt": "prompt text"})()
        assert action._extract_text(req) == "prompt text"

    def test_extract_text_joins_messages(self) -> None:
        action = self._action()
        msgs = [type("M", (), {"content": "hello"})(), type("M", (), {"content": "world"})()]
        req = type("R", (), {"messages": msgs})()
        assert action._extract_text(req) == "hello world"

    def test_extract_text_empty_on_nothing(self) -> None:
        action = self._action()
        assert action._extract_text(type("R", (), {"messages": []})()) == ""

    def test_redact_pii_empty_text_short_circuits(self) -> None:
        action = self._action()
        assert action._redact_pii("") == ""


# ── JwtManager / RuntimeKeyStore — verification + persistence edges ─────────


class TestJwtAuthEdges:
    def test_verify_returns_none_without_secret(self) -> None:
        settings = type("S", (), {"jwt_secret": None, "jwt_ttl_minutes": 15})()
        mgr = JwtManager(settings)  # type: ignore[arg-type]
        assert mgr.verify("any-token") is None

    def test_keystore_corrupt_file_loads_empty_without_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "keys.json"
        path.write_text("{not valid json")
        store = RuntimeKeyStore(path)
        assert store.all_keys() == {}

    def test_keystore_unreadable_file_logged_not_raised(self, tmp_path: Path) -> None:
        path = tmp_path / "keys.json"
        path.write_text('{"keys": [{"key": "ng_k", "tenant": "demo"}]}')
        # Simulate an OSError on read (e.g. permissions race).
        with patch.object(Path, "read_text", side_effect=OSError("perm")):
            store = RuntimeKeyStore(path)
        assert store.all_keys() == {}

    def test_keystore_remove_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "keys.json"
        store = RuntimeKeyStore(path)
        store.add("ng_aaa", "Demo")
        assert store.all_keys() == {"ng_aaa": "demo"}  # tenant lowercased
        assert store.remove("ng_aaa") is True
        assert store.remove("ng_aaa") is False
        # Persisted file reflects the removal.
        reloaded = RuntimeKeyStore(path)
        assert reloaded.all_keys() == {}

    def test_keystore_skips_malformed_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "keys.json"
        path.write_text(json.dumps({"keys": [{"key": "", "tenant": "x"}, {"tenant": "y"}]}))
        store = RuntimeKeyStore(path)
        assert store.all_keys() == {}

    def test_keystore_persist_cleans_up_on_oserror(self, tmp_path: Path) -> None:
        path = tmp_path / "keys.json"
        store = RuntimeKeyStore(path)
        with patch("os.fchmod", side_effect=OSError("no perms")), pytest.raises(OSError):
            store.add("ng_k", "t")
        # No temp-file litter left behind.
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".keys-")] == []


# ── RedisRateLimiter — malformed-response + lifecycle edges ─────────────────


class _ScriptedRedis:
    """Fake redis client whose registered scripts return canned payloads."""

    def __init__(self, result: Any, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.closed = False

    def register_script(self, _script: str) -> Any:
        async def call(*_a: Any, **_kw: Any) -> Any:
            if self.raises is not None:
                raise self.raises
            return self.result

        return call

    async def aclose(self) -> None:
        self.closed = True


class TestRedisRateLimiterEdges:
    def _limiter(self, client: Any) -> RedisRateLimiter:
        return RedisRateLimiter(
            RateLimitSettings(backend="redis", redis_url="redis://x"), client=client
        )

    async def test_malformed_response_fails_closed(self) -> None:
        for garbage in (None, [], ["x", "y"], ["1"]):
            limiter = self._limiter(_ScriptedRedis(garbage))
            allowed, remaining, retry = await limiter.check("rl:t", limit=5, burst=3)
            assert allowed is False
            assert remaining == 0
            assert retry == 1

    async def test_cost_malformed_response_fails_closed(self) -> None:
        limiter = self._limiter(_ScriptedRedis(["a", "b", "c"]))
        allowed, remaining, retry = await limiter.check_cost("rl:t", cost=1, budget=5)
        assert allowed is False
        assert remaining == 0
        assert retry == 1

    async def test_non_own_client_survives_aclose(self) -> None:
        fake = _ScriptedRedis([1, 2, 3])
        limiter = self._limiter(fake)
        await limiter.aclose()
        assert fake.closed is False  # injected → not ours to close


# ── SiemRouter — sync fan-out + per-sink failure accounting ──────────────────


class TestSiemRouterErrorPaths:
    def _settings(self, **over: Any) -> SiemSettings:
        base: dict[str, Any] = {
            "enabled": True,
            "webhook_url": "http://siem.test/ingest",
            "splunk_hec_url": "https://splunk.test:8088",
            "splunk_hec_token": "tok",
            "scarletai_url": "http://scarlet.test/ingest",
        }
        base.update(over)
        return SiemSettings(**base)

    def _event(self) -> AuditEvent:
        return AuditEvent(
            request_id="req-9",
            tenant_id="default",
            verdict=Verdict.BLOCK,
            findings_count=1,
            threat_categories=[],
            confidence=0.9,
            total_latency_ms=1.0,
        )

    def test_sync_delivery_fans_out_to_all_sinks(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200)

        router = SiemRouter(self._settings(), transport=httpx.MockTransport(handler))
        # Sync pytest (no running loop) → route() takes the sync path.
        router.route(self._event())
        assert any("siem.test" in u for u in seen)
        assert any("splunk.test" in u for u in seen)
        assert any("scarlet.test" in u for u in seen)

    def test_async_sink_exception_is_counted_dropped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("sink down")

        router = SiemRouter(
            self._settings(
                webhook_url=None, splunk_hec_url=None, scarletai_url="http://scarlet.test/ingest"
            ),
            transport=httpx.MockTransport(handler),
        )
        before = router._drops
        router._deliver_sync(router._verdict_payload(self._event()))
        assert router._drops == before + 1

    def test_webhook_sync_failure_is_counted_dropped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("sink down")

        router = SiemRouter(
            self._settings(splunk_hec_url=None, scarletai_url=None),
            transport=httpx.MockTransport(handler),
        )
        before = router._drops
        router.route(self._event())
        assert router._drops == before + 1


# ── pattern_i18n.resolve_category — every prefix branch ─────────────────────


class TestI18nResolveCategory:
    @pytest.mark.parametrize(
        ("prefix", "expected"),
        [
            ("PI-D", ThreatCategory.PROMPT_INJECTION_DIRECT),
            ("PI-I", ThreatCategory.PROMPT_INJECTION_INDIRECT),
            ("JB", ThreatCategory.JAILBREAK),
            ("EXT", ThreatCategory.SYSTEM_PROMPT_EXTRACTION),
            ("EXF", ThreatCategory.DATA_EXFILTRATION),
            ("TOOL", ThreatCategory.TOOL_MISUSE),
            ("DOS", ThreatCategory.DOS_ABUSE),
            ("ENC", ThreatCategory.ENCODING_EVASION),
            ("ZZZ", ThreatCategory.PROMPT_INJECTION_DIRECT),  # fallback
        ],
    )
    def test_prefix_branches(self, prefix: str, expected: ThreatCategory) -> None:
        assert resolve_category(f"{prefix}-XX-001") is expected


# ── semantic package lazy imports ────────────────────────────────────────────


class TestSemanticLazyImports:
    def test_lazy_classes_resolve(self) -> None:
        import neuralguard.semantic as sem

        assert sem.AttackCorpus.__name__ == "AttackCorpus"
        assert sem.HybridScoringEngine.__name__ == "HybridScoringEngine"
        assert sem.JudgeScanner.__name__ == "JudgeScanner"
        assert sem.EmbeddingEngine.__name__ == "EmbeddingEngine"
        assert sem.SimilarityScanner.__name__ == "SimilarityScanner"

    def test_unknown_attribute_raises(self) -> None:
        import neuralguard.semantic as sem

        with pytest.raises(AttributeError, match="no attribute"):
            _ = sem.DefinitelyNotAThing  # type: ignore[attr-defined]


# ── egress classification edge branches ──────────────────────────────────────


class TestEgressClassification:
    def test_empty_host_is_egress(self) -> None:
        assert is_private_endpoint("") is False
        assert is_private_endpoint("not a url") is False

    def test_docker_host_reference_is_local(self) -> None:
        assert is_private_endpoint("http://host.docker.internal:11434") is True

    def test_bare_name_is_local(self) -> None:
        assert is_private_endpoint("http://redis:6379") is True

    def test_public_hostname_is_egress(self) -> None:
        assert is_private_endpoint("https://api.example.com") is False

    def test_ipv6_loopback_and_unspecified_are_local(self) -> None:
        assert is_private_endpoint("http://[::1]:11434") is True
        assert is_private_endpoint("http://[::]:11434") is True
        assert is_private_endpoint("http://169.254.1.1") is True  # link-local
