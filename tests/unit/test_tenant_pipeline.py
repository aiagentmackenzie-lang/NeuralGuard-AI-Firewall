"""Tests for per-tenant scanner enforcement in the pipeline (Sprint C, C1)."""

from __future__ import annotations

from types import SimpleNamespace

from neuralguard.config.settings import (
    AgentGuardianSettings,
    NeuralGuardConfig,
)
from neuralguard.models.schemas import EvaluateRequest, ScanLayer
from neuralguard.scanners.pipeline import ScannerPipeline
from neuralguard.tenants.config import TenantScannerOverrides


def _registry(overlay_for_tenant: dict[str, TenantScannerOverrides | None]):
    """Build a minimal fake registry implementing the two methods the pipeline uses."""
    return SimpleNamespace(
        enabled=True,
        effective_scanner_overlay=lambda tid: overlay_for_tenant.get(tid),
    )


class TestGetEnabledLayersTenantCeiling:
    def test_no_registry_keeps_global_behavior(self):
        config = NeuralGuardConfig()  # all optional scanners off by default
        pipeline = ScannerPipeline(config)
        # No registry installed -> global default (structural + pattern only).
        layers = pipeline.get_enabled_layers(EvaluateRequest(prompt="x", tenant_id="acme"))
        assert ScanLayer.STRUCTURAL in layers
        assert ScanLayer.PATTERN in layers
        assert ScanLayer.SEMANTIC not in layers

    def test_registry_disabled_keeps_global_behavior(self):
        config = NeuralGuardConfig()
        pipeline = ScannerPipeline(config)
        pipeline.set_tenant_registry(_registry({}))  # but registry.enabled handled by get path
        # The fake has enabled=True; to test disabled, use SimpleNamespace(enabled=False)
        pipeline._tenant_registry = SimpleNamespace(
            enabled=False,
            effective_scanner_overlay=lambda tid: None,
        )
        layers = pipeline.get_enabled_layers(EvaluateRequest(prompt="x", tenant_id="acme"))
        # Agent guardian off globally -> not in baseline.
        assert ScanLayer.AGENT_GUARDIAN not in layers

    def test_tenant_can_disable_optional_scanner(self):
        config = NeuralGuardConfig(
            agent_guardian=AgentGuardianSettings(enabled=True),
        )
        pipeline = ScannerPipeline(config)
        pipeline.set_tenant_registry(
            _registry(
                {"acme": TenantScannerOverrides(agent_guardian=False)}
            )
        )
        layers = pipeline.get_enabled_layers(
            EvaluateRequest(prompt="x", tenant_id="acme")
        )
        # AG globally on, but tenant disables it -> not in the ceiling.
        assert ScanLayer.AGENT_GUARDIAN not in layers
        # Mandatory layers stay.
        assert ScanLayer.STRUCTURAL in layers
        assert ScanLayer.PATTERN in layers

    def test_tenant_cannot_disable_structural_or_pattern(self):
        config = NeuralGuardConfig()
        pipeline = ScannerPipeline(config)
        # Even if a tenant somehow declared a structural override, it is not
        # modelled — structural + pattern are always on. We verify the
        # override object cannot express structural at all.
        assert not hasattr(TenantScannerOverrides(), "structural")
        assert not hasattr(TenantScannerOverrides(), "pattern")
        layers = pipeline.get_enabled_layers(
            EvaluateRequest(prompt="x", tenant_id="acme")
        )
        assert ScanLayer.STRUCTURAL in layers
        assert ScanLayer.PATTERN in layers

    def test_tenant_cannot_widen_past_global_registration(self):
        config = NeuralGuardConfig()  # semantic globally OFF
        pipeline = ScannerPipeline(config)
        pipeline.set_tenant_registry(
            _registry(
                {"acme": TenantScannerOverrides(semantic=True)}
            )
        )
        layers = pipeline.get_enabled_layers(
            EvaluateRequest(prompt="x", tenant_id="acme")
        )
        # Tenant wants semantic ON, but global config didn't enable it -> off.
        assert ScanLayer.SEMANTIC not in layers

    def test_unknown_tenant_inherits_global(self):
        config = NeuralGuardConfig(
            agent_guardian=AgentGuardianSettings(enabled=True),
        )
        pipeline = ScannerPipeline(config)
        pipeline.set_tenant_registry(_registry({}))  # no override for "ghost"
        layers = pipeline.get_enabled_layers(
            EvaluateRequest(prompt="x", tenant_id="ghost")
        )
        # AG globally on, no tenant override -> inherited.
        assert ScanLayer.AGENT_GUARDIAN in layers

    def test_client_request_only_narrows(self):
        """Client scanners field intersects the tenant ceiling (narrow, never widen)."""
        config = NeuralGuardConfig(
            agent_guardian=AgentGuardianSettings(enabled=True),
        )
        pipeline = ScannerPipeline(config)
        pipeline.set_tenant_registry(
            _registry(
                {"acme": TenantScannerOverrides(agent_guardian=False)}
            )
        )
        # Client tries to force AG back on after the tenant disabled it.
        request = EvaluateRequest(
            prompt="x",
            tenant_id="acme",
            scanners=[ScanLayer.STRUCTURAL, ScanLayer.PATTERN, ScanLayer.AGENT_GUARDIAN],
        )
        layers = pipeline.get_enabled_layers(request)
        # Tenant ceiling wins -> AG stays off.
        assert ScanLayer.AGENT_GUARDIAN not in layers
        assert ScanLayer.PATTERN in layers

    def test_client_can_narrow_within_ceiling(self):
        config = NeuralGuardConfig(
            agent_guardian=AgentGuardianSettings(enabled=True),
        )
        pipeline = ScannerPipeline(config)
        pipeline.set_tenant_registry(_registry({}))  # inherit global
        request = EvaluateRequest(
            prompt="x",
            tenant_id="acme",
            scanners=[ScanLayer.PATTERN],
        )
        layers = pipeline.get_enabled_layers(request)
        assert layers == [ScanLayer.PATTERN]
