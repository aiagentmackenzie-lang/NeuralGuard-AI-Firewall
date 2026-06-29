"""Tests for per-tenant override model (Sprint C, C1 / P1-2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralguard.tenants.config import TenantConfig, TenantScannerOverrides


class TestTenantScannerOverrides:
    def test_defaults_all_none(self):
        o = TenantScannerOverrides()
        assert o.agent_guardian is None
        assert o.semantic is None
        assert o.judge is None

    def test_tri_state(self):
        o = TenantScannerOverrides(agent_guardian=True, semantic=False, judge=None)
        assert o.agent_guardian is True
        assert o.semantic is False
        assert o.judge is None

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            TenantScannerOverrides(structural=False)  # type: ignore[call-arg]


class TestTenantConfig:
    def test_minimal_valid(self):
        cfg = TenantConfig(tenant_id="acme")
        assert cfg.tenant_id == "acme"
        assert cfg.requests_per_minute is None
        assert cfg.burst_size is None
        assert cfg.scanners.agent_guardian is None
        assert cfg.description is None

    def test_tenant_id_rules(self):
        ok = ["acme", "globex-1", "tenant_a", "t.example", "123abc"]
        for tid in ok:
            TenantConfig(tenant_id=tid)  # should not raise

    @pytest.mark.parametrize(
        "bad",
        [
            "",  # empty
            "ACME",  # uppercase
            " acme",  # leading space
            "-acme",  # leading hyphen (must start alphanumeric)
            "acme!",  # invalid char
            "x" * 65,  # too long
            "acme space",  # space
        ],
    )
    def test_tenant_id_rejected(self, bad: str):
        with pytest.raises(ValidationError):
            TenantConfig(tenant_id=bad)

    def test_rpm_floor_one(self):
        with pytest.raises(ValidationError):
            TenantConfig(tenant_id="acme", requests_per_minute=0)
        with pytest.raises(ValidationError):
            TenantConfig(tenant_id="acme", requests_per_minute=-5)
        TenantConfig(tenant_id="acme", requests_per_minute=1)  # ok
        TenantConfig(tenant_id="acme", requests_per_minute=10_000)  # high quota ok

    def test_burst_floor_zero(self):
        with pytest.raises(ValidationError):
            TenantConfig(tenant_id="acme", burst_size=-1)
        TenantConfig(tenant_id="acme", burst_size=0)  # ok

    def test_effective_rate_limit_inherits_global_on_none(self):
        cfg = TenantConfig(tenant_id="acme")
        assert cfg.effective_rate_limit(60, 10) == (60, 10)

    def test_effective_rate_limit_overrides(self):
        cfg = TenantConfig(tenant_id="acme", requests_per_minute=200, burst_size=30)
        assert cfg.effective_rate_limit(60, 10) == (200, 30)

    def test_effective_rate_limit_partial_override(self):
        cfg = TenantConfig(tenant_id="acme", requests_per_minute=200)
        assert cfg.effective_rate_limit(60, 10) == (200, 10)

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            TenantConfig(tenant_id="acme", foo="bar")  # type: ignore[call-arg]

    def test_to_effective_dict_no_secrets(self):
        cfg = TenantConfig(
            tenant_id="acme",
            description="Acme Co",
            requests_per_minute=100,
            burst_size=5,
            scanners=TenantScannerOverrides(semantic=False),
        )
        d = cfg.to_effective_dict()
        assert d["tenant_id"] == "acme"
        assert d["description"] == "Acme Co"
        assert d["requests_per_minute"] == 100
        assert d["burst_size"] == 5
        assert d["scanners"]["semantic"] is False
        # No secret-bearing keys are ever present in the model.
        assert "secret" not in str(d)
