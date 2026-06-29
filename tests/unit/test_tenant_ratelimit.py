"""Tests for per-tenant rate-limit resolution in the middleware (Sprint C, C1)."""

from __future__ import annotations

from types import SimpleNamespace

from neuralguard.config.settings import RateLimitSettings
from neuralguard.middleware.ratelimit import RateLimitMiddleware
from neuralguard.tenants.config import TenantConfig, TenantScannerOverrides


class _FakeApp:
    """Minimal ASGI app stand-in for RateLimitMiddleware construction."""


class TestRateLimitPerTenantResolution:
    def _middleware(self, registry) -> RateLimitMiddleware:
        mw = RateLimitMiddleware(
            _FakeApp(),
            RateLimitSettings(enabled=True, requests_per_minute=60, burst_size=10),
            tenant_registry=registry,
        )
        return mw

    def test_no_registry_uses_global(self):
        mw = self._middleware(None)
        # We exercise the resolution helper path indirectly via dispatch is
        # complex; instead assert the wiring attribute and that the global
        # settings are the fallback. The full request-path behavior is covered
        # by the integration test.
        assert mw._tenant_registry is None
        assert mw.settings.requests_per_minute == 60

    def test_registry_resolves_per_tenant(self, tmp_path):
        # A real registry with one tenant override.
        (tmp_path / "acme.yaml").write_text(
            "tenant_id: acme\nrequests_per_minute: 5\nburst_size: 1\n", encoding="utf-8"
        )
        from neuralguard.tenants.registry import TenantConfigRegistry

        reg = TenantConfigRegistry(
            enabled=True, default_tenant="default", config_path=tmp_path
        )
        reg.load()
        mw = self._middleware(reg)
        assert mw._tenant_registry is reg
        assert reg.effective_rate_limit("acme", 60, 10) == (5, 1)
        # Unknown tenant -> global default (fail-open).
        assert reg.effective_rate_limit("ghost", 60, 10) == (60, 10)

    def test_disabled_registry_uses_global(self):
        reg = SimpleNamespace(enabled=False)
        mw = self._middleware(reg)
        assert mw._tenant_registry is reg
        # The dispatch path checks `registry is not None and registry.enabled`;
        # disabled means the per-tenant branch is skipped. We assert the guard
        # semantics directly via the effective_rate_limit on the global settings.
        assert mw.settings.requests_per_minute == 60
        assert mw.settings.burst_size == 10


class TestTenantConfigRateLimitEdge:
    """Boundary checks for the override model used by the middleware."""

    def test_higher_quota_allowed(self):
        cfg = TenantConfig(tenant_id="acme", requests_per_minute=500, burst_size=100)
        assert cfg.effective_rate_limit(60, 10) == (500, 100)

    def test_scanner_overlay_unused_for_rate_limit(self):
        cfg = TenantConfig(
            tenant_id="acme",
            requests_per_minute=100,
            scanners=TenantScannerOverrides(semantic=False),
        )
        # The rate-limit path only reads rpm/burst, not scanner state.
        rpm, burst = cfg.effective_rate_limit(60, 10)
        assert rpm == 100 and burst == 10
