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


class TestJudgeResolvesEscalate:
    """Judge resolution of the ambiguous ESCALATE zone (A2 FPR fix)."""

    def test_clean_judge_allow_resolves_escalate_to_allow(self, config):
        """A clean LLM-Judge ALLOW downgrades a hybrid ESCALATE to ALLOW (opt-in)."""
        config.action.judge_resolves_escalate = True
        config.scanner.semantic_enabled = True
        config.scanner.judge_enabled = True
        config.action.fail_closed = False
        pipeline = ScannerPipeline(config)
        pipeline.register_scanner(
            MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.STRUCTURAL)
        )
        pipeline.register_scanner(
            MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.PATTERN)
        )
        pipeline.register_scanner(
            MockScanner(
                config.scanner,
                Verdict.ESCALATE,
                findings=[
                    Finding(
                        category=ThreatCategory.PROMPT_INJECTION_DIRECT,
                        severity=Severity.MEDIUM,
                        verdict=Verdict.ESCALATE,
                        confidence=0.68,
                        layer=ScanLayer.SEMANTIC,
                        rule_id="SEM-001",
                        description="ambiguous semantic match",
                    )
                ],
                layer=ScanLayer.SEMANTIC,
            )
        )
        pipeline.register_scanner(MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.JUDGE))
        result = pipeline.execute(EvaluateRequest(prompt="test"))
        assert result.verdict == Verdict.ALLOW
        assert "judge_resolve" in result.arbitration_reason

    def test_errored_judge_does_not_resolve_escalate(self, config):
        """A timed-out/errored judge does NOT resolve ESCALATE (pre-judge stands)."""
        from neuralguard.models.schemas import ScannerResult

        config.action.judge_resolves_escalate = True
        config.scanner.semantic_enabled = True
        config.scanner.judge_enabled = True
        config.action.fail_closed = False
        pipeline = ScannerPipeline(config)
        pipeline.register_scanner(
            MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.STRUCTURAL)
        )
        pipeline.register_scanner(
            MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.PATTERN)
        )
        pipeline.register_scanner(
            MockScanner(
                config.scanner,
                Verdict.ESCALATE,
                findings=[
                    Finding(
                        category=ThreatCategory.PROMPT_INJECTION_DIRECT,
                        severity=Severity.MEDIUM,
                        verdict=Verdict.ESCALATE,
                        confidence=0.68,
                        layer=ScanLayer.SEMANTIC,
                        rule_id="SEM-001",
                        description="ambiguous semantic match",
                    )
                ],
                layer=ScanLayer.SEMANTIC,
            )
        )
        timeout_result = ScannerResult(
            layer=ScanLayer.JUDGE,
            verdict=Verdict.ALLOW,
            findings=[],
            latency_ms=1.0,
            error="Judge timed out after 5s",
        )
        judge = MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.JUDGE)
        judge.scan = lambda req, ctx=None: timeout_result
        judge.safe_scan = lambda req, ctx=None: timeout_result
        pipeline.register_scanner(judge)
        result = pipeline.execute(EvaluateRequest(prompt="test"))
        assert result.verdict == Verdict.ESCALATE

    def test_judge_cannot_downgrade_sanitize(self, config):
        """Judge ALLOW cannot downgrade SANITIZE - only ESCALATE."""
        config.action.judge_resolves_escalate = True
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
                        description="encoding evasion",
                    )
                ],
                layer=ScanLayer.PATTERN,
            )
        )
        pipeline.register_scanner(MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.JUDGE))
        result = pipeline.execute(EvaluateRequest(prompt="test"))
        assert result.verdict == Verdict.SANITIZE

    def test_judge_resolves_disabled_keeps_escalate(self, config):
        """With judge_resolves_escalate=False, ESCALATE stands on clean judge ALLOW."""
        config.action.judge_resolves_escalate = False
        config.scanner.semantic_enabled = True
        config.scanner.judge_enabled = True
        config.action.fail_closed = False
        pipeline = ScannerPipeline(config)
        pipeline.register_scanner(
            MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.STRUCTURAL)
        )
        pipeline.register_scanner(
            MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.PATTERN)
        )
        pipeline.register_scanner(
            MockScanner(
                config.scanner,
                Verdict.ESCALATE,
                findings=[
                    Finding(
                        category=ThreatCategory.PROMPT_INJECTION_DIRECT,
                        severity=Severity.MEDIUM,
                        verdict=Verdict.ESCALATE,
                        confidence=0.68,
                        layer=ScanLayer.SEMANTIC,
                        rule_id="SEM-001",
                        description="ambiguous semantic match",
                    )
                ],
                layer=ScanLayer.SEMANTIC,
            )
        )
        pipeline.register_scanner(MockScanner(config.scanner, Verdict.ALLOW, layer=ScanLayer.JUDGE))
        result = pipeline.execute(EvaluateRequest(prompt="test"))
        assert result.verdict == Verdict.ESCALATE
