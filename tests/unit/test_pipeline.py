"""Unit tests for the scanner pipeline and Layer Arbitration."""

import pytest

from neuralguard.config.settings import NeuralGuardConfig, ScannerSettings
from neuralguard.models.schemas import (
    EvaluateRequest,
    Finding,
    ScanLayer,
    Severity,
    ThreatCategory,
    Verdict,
)
from neuralguard.scanners.base import BaseScanner
from neuralguard.scanners.pipeline import ScannerPipeline
from neuralguard.scanners.structural import StructuralScanner


class MockScanner(BaseScanner):
    """Test scanner with configurable verdict."""

    def __init__(
        self,
        settings: ScannerSettings,
        verdict: Verdict,
        findings: list[Finding] | None = None,
        layer: ScanLayer = ScanLayer.PATTERN,
    ):
        super().__init__(settings)
        self._verdict = verdict
        self._findings = findings or []
        self.layer = layer

    def scan(self, request, context=None):
        import time

        start = time.perf_counter()
        return self._result(self._verdict, self._findings, start)


@pytest.fixture
def config():
    return NeuralGuardConfig()


@pytest.fixture
def pipeline(config):
    p = ScannerPipeline(config)
    p.register_scanner(StructuralScanner(config.scanner))
    return p


class TestPipelineBasics:
    """Basic pipeline execution tests."""

    def test_clean_prompt_passes(self, pipeline):
        result = pipeline.execute(EvaluateRequest(prompt="Hello, how are you?"))
        assert result.verdict == Verdict.ALLOW
        assert result.total_latency_ms < 100

    def test_pipeline_registers_scanner(self, pipeline):
        assert ScanLayer.STRUCTURAL in pipeline._scanners

    def test_pipeline_unregister_scanner(self, pipeline):
        pipeline.unregister_scanner(ScanLayer.STRUCTURAL)
        assert ScanLayer.STRUCTURAL not in pipeline._scanners

    def test_pipeline_enabled_layers_includes_structural(self, pipeline):
        """With config defaults (semantic/judge off), only structural + pattern layers expected."""
        layers = pipeline.get_enabled_layers()
        assert ScanLayer.STRUCTURAL in layers

    def test_pipeline_enabled_layers_with_request_override(self, pipeline):
        request = EvaluateRequest(
            prompt="test",
            scanners=[ScanLayer.STRUCTURAL],
        )
        layers = pipeline.get_enabled_layers(request)
        assert layers == [ScanLayer.STRUCTURAL]


class TestLayerArbitration:
    """Tests for Layer Arbitration — strictest verdict wins."""

    def test_allow_when_all_allow(self, config):
        pipeline = ScannerPipeline(config)
        pipeline.register_scanner(
            MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.STRUCTURAL)
        )
        result = pipeline.execute(EvaluateRequest(prompt="test"))
        assert result.verdict == Verdict.ALLOW

    def test_block_overrides_allow(self, config):
        pipeline = ScannerPipeline(config)
        pipeline.register_scanner(
            MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.STRUCTURAL)
        )
        pipeline.register_scanner(
            MockScanner(
                config.scanner,
                Verdict.BLOCK,
                findings=[
                    Finding(
                        category=ThreatCategory.PROMPT_INJECTION_DIRECT,
                        severity=Severity.HIGH,
                        verdict=Verdict.BLOCK,
                        confidence=0.95,
                        layer=ScanLayer.PATTERN,
                        rule_id="PI-D-001",
                        description="Direct injection",
                    )
                ],
                layer=ScanLayer.PATTERN,
            )
        )
        result = pipeline.execute(EvaluateRequest(prompt="test"))
        assert result.verdict == Verdict.BLOCK

    def test_sanitize_between_allow_and_block(self, config):
        pipeline = ScannerPipeline(config)
        pipeline.register_scanner(
            MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.STRUCTURAL)
        )
        pipeline.register_scanner(
            MockScanner(
                config.scanner,
                Verdict.SANITIZE,
                findings=[
                    Finding(
                        category=ThreatCategory.ENCODING_EVASION,
                        severity=Severity.MEDIUM,
                        verdict=Verdict.SANITIZE,
                        confidence=0.8,
                        layer=ScanLayer.PATTERN,
                        rule_id="ENC-001",
                        description="Encoding evasion",
                    )
                ],
                layer=ScanLayer.PATTERN,
            )
        )
        result = pipeline.execute(EvaluateRequest(prompt="test"))
        assert result.verdict == Verdict.SANITIZE

    def test_fail_closed_no_scanners(self, config):
        config.action.fail_closed = True
        pipeline = ScannerPipeline(config)
        result = pipeline.execute(EvaluateRequest(prompt="test"))
        assert result.verdict == Verdict.BLOCK
        assert "fail-closed" in result.arbitration_reason

    def test_fail_open_no_scanners(self, config):
        config.action.fail_closed = False
        pipeline = ScannerPipeline(config)
        result = pipeline.execute(EvaluateRequest(prompt="test"))
        assert result.verdict == Verdict.ALLOW
        assert "fail-open" in result.arbitration_reason

    def test_early_exit_on_block(self, config):
        """When fail_closed=True, pipeline should stop after BLOCK."""
        config.action.fail_closed = True
        pipeline = ScannerPipeline(config)

        pipeline.register_scanner(
            MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.STRUCTURAL)
        )
        pipeline.register_scanner(
            MockScanner(
                config.scanner,
                Verdict.BLOCK,
                findings=[
                    Finding(
                        category=ThreatCategory.PROMPT_INJECTION_DIRECT,
                        severity=Severity.HIGH,
                        verdict=Verdict.BLOCK,
                        confidence=0.95,
                        layer=ScanLayer.PATTERN,
                        rule_id="PI-D-001",
                        description="Direct injection",
                    )
                ],
                layer=ScanLayer.PATTERN,
            )
        )

        result = pipeline.execute(EvaluateRequest(prompt="test"))
        assert result.verdict == Verdict.BLOCK
