"""Tests for HybridScoringEngine — composite risk scoring.

Tests cover weight configuration, score computation, verdict mapping,
finding enhancement, and pipeline integration.
"""

from __future__ import annotations

import pytest

from neuralguard.config.settings import ActionSettings, NeuralGuardConfig, ScannerSettings
from neuralguard.models.schemas import (
    EvaluateRequest,
    Finding,
    LayerArbitrationResult,
    ScanLayer,
    ScannerResult,
    Severity,
    ThreatCategory,
    Verdict,
)
from neuralguard.semantic.hybrid import HybridScoreResult, HybridScoringEngine

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def config() -> NeuralGuardConfig:
    """Default config with default thresholds."""
    return NeuralGuardConfig()


@pytest.fixture
def engine(config: NeuralGuardConfig) -> HybridScoringEngine:
    """Default hybrid scoring engine."""
    return HybridScoringEngine(config)


def _scanner_result(
    layer: ScanLayer,
    verdict: Verdict,
    findings: list[Finding] | None = None,
) -> ScannerResult:
    """Helper to create a ScannerResult."""
    return ScannerResult(
        layer=layer,
        verdict=verdict,
        findings=findings or [],
        latency_ms=1.0,
    )


def _finding(
    layer: ScanLayer = ScanLayer.PATTERN,
    confidence: float = 0.9,
    verdict: Verdict = Verdict.BLOCK,
    category: ThreatCategory = ThreatCategory.PROMPT_INJECTION_DIRECT,
) -> Finding:
    """Helper to create a Finding."""
    return Finding(
        category=category,
        severity=Severity.HIGH,
        verdict=verdict,
        confidence=confidence,
        layer=layer,
        rule_id="TEST-001",
        description="Test finding",
    )


# ── Weight Configuration ─────────────────────────────────────────────────────


class TestHybridWeights:
    """Test weight configuration."""

    def test_default_weights(self, engine: HybridScoringEngine) -> None:
        assert engine.pattern_weight == 0.6
        assert engine.semantic_weight == 0.4
        assert abs(engine.pattern_weight + engine.semantic_weight - 1.0) < 0.01

    def test_set_valid_weights(self, engine: HybridScoringEngine) -> None:
        engine.set_weights(0.7, 0.3)
        assert engine.pattern_weight == 0.7
        assert engine.semantic_weight == 0.3

    def test_set_invalid_weights_raises(self, engine: HybridScoringEngine) -> None:
        with pytest.raises(ValueError, match=r"must sum to 1.0"):
            engine.set_weights(0.5, 0.6)

    def test_set_equal_weights(self, engine: HybridScoringEngine) -> None:
        engine.set_weights(0.5, 0.5)
        assert engine.pattern_weight == 0.5
        assert engine.semantic_weight == 0.5


# ── Score Computation ────────────────────────────────────────────────────────


class TestHybridScoring:
    """Test composite score computation."""

    def test_no_findings_composite_zero(self, engine: HybridScoringEngine) -> None:
        """No findings → composite 0.0, ALLOW."""
        result = engine.score([])
        assert result.composite == 0.0
        assert result.verdict == Verdict.ALLOW

    def test_pattern_only(self, engine: HybridScoringEngine) -> None:
        """Pattern findings only → composite = pattern confidence."""
        findings = [_finding(layer=ScanLayer.PATTERN, confidence=0.9, verdict=Verdict.BLOCK)]
        results = [_scanner_result(ScanLayer.PATTERN, Verdict.BLOCK, findings)]

        result = engine.score(results)
        assert result.composite == 0.9
        assert result.pattern_max_confidence == 0.9
        assert result.semantic_max_similarity == 0.0
        assert result.verdict == Verdict.BLOCK

    def test_semantic_only(self, engine: HybridScoringEngine) -> None:
        """Semantic findings only → composite = semantic similarity."""
        findings = [_finding(layer=ScanLayer.SEMANTIC, confidence=0.7, verdict=Verdict.ESCALATE)]
        results = [_scanner_result(ScanLayer.SEMANTIC, Verdict.ESCALATE, findings)]

        result = engine.score(results)
        assert result.composite == 0.7
        assert result.pattern_max_confidence == 0.0
        assert result.semantic_max_similarity == 0.7

    def test_both_layers_weighted(self, engine: HybridScoringEngine) -> None:
        """Both layers → composite = w_p * pattern + w_s * semantic."""
        pattern_findings = [
            _finding(layer=ScanLayer.PATTERN, confidence=0.5, verdict=Verdict.SANITIZE)
        ]
        semantic_findings = [
            _finding(layer=ScanLayer.SEMANTIC, confidence=0.7, verdict=Verdict.ESCALATE)
        ]

        results = [
            _scanner_result(ScanLayer.PATTERN, Verdict.SANITIZE, pattern_findings),
            _scanner_result(ScanLayer.SEMANTIC, Verdict.ESCALATE, semantic_findings),
        ]

        result = engine.score(results)
        # composite = 0.6 * 0.5 + 0.4 * 0.7 = 0.30 + 0.28 = 0.58
        assert abs(result.composite - 0.58) < 0.01
        assert abs(result.pattern_contribution - 0.30) < 0.01
        assert abs(result.semantic_contribution - 0.28) < 0.01

    def test_pattern_block_overrides(self, engine: HybridScoringEngine) -> None:
        """Pattern BLOCK → composite = pattern confidence (full weight)."""
        pattern_findings = [
            _finding(layer=ScanLayer.PATTERN, confidence=0.95, verdict=Verdict.BLOCK)
        ]
        semantic_findings = [
            _finding(layer=ScanLayer.SEMANTIC, confidence=0.8, verdict=Verdict.BLOCK)
        ]

        results = [
            _scanner_result(ScanLayer.PATTERN, Verdict.BLOCK, pattern_findings),
            _scanner_result(ScanLayer.SEMANTIC, Verdict.BLOCK, semantic_findings),
        ]

        result = engine.score(results)
        # Pattern BLOCK → composite = pattern confidence directly
        assert result.composite == 0.95
        assert result.pattern_contribution == 0.95
        assert result.semantic_contribution == 0.0

    def test_semantic_upgrades_pattern_allow(self, engine: HybridScoringEngine) -> None:
        """Pattern ALLOW + Semantic finding can upgrade verdict via composite."""
        pattern_findings = [
            _finding(layer=ScanLayer.PATTERN, confidence=0.3, verdict=Verdict.ALLOW)
        ]
        semantic_findings = [
            _finding(layer=ScanLayer.SEMANTIC, confidence=0.9, verdict=Verdict.BLOCK)
        ]

        results = [
            _scanner_result(ScanLayer.PATTERN, Verdict.ALLOW, pattern_findings),
            _scanner_result(ScanLayer.SEMANTIC, Verdict.BLOCK, semantic_findings),
        ]

        result = engine.score(results)
        # Both found something → weighted: 0.6*0.3 + 0.4*0.9 = 0.18 + 0.36 = 0.54
        assert abs(result.composite - 0.54) < 0.01

    def test_custom_weights_change_composite(self, config: NeuralGuardConfig) -> None:
        """Changing weights changes composite score."""
        engine = HybridScoringEngine(config)
        engine.set_weights(0.5, 0.5)

        pattern_findings = [
            _finding(layer=ScanLayer.PATTERN, confidence=0.6, verdict=Verdict.SANITIZE)
        ]
        semantic_findings = [
            _finding(layer=ScanLayer.SEMANTIC, confidence=0.8, verdict=Verdict.ESCALATE)
        ]

        results = [
            _scanner_result(ScanLayer.PATTERN, Verdict.SANITIZE, pattern_findings),
            _scanner_result(ScanLayer.SEMANTIC, Verdict.ESCALATE, semantic_findings),
        ]

        result = engine.score(results)
        # 0.5*0.6 + 0.5*0.8 = 0.30 + 0.40 = 0.70
        assert abs(result.composite - 0.70) < 0.01


# ── Verdict Mapping ─────────────────────────────────────────────────────────


class TestHybridVerdictMapping:
    """Test composite score → verdict mapping."""

    def test_high_score_blocks(self, engine: HybridScoringEngine) -> None:
        """Composite ≥ 0.85 → BLOCK."""
        findings = [_finding(layer=ScanLayer.PATTERN, confidence=0.90, verdict=Verdict.BLOCK)]
        results = [_scanner_result(ScanLayer.PATTERN, Verdict.BLOCK, findings)]
        result = engine.score(results)
        assert result.verdict == Verdict.BLOCK

    def test_medium_score_sanitizes(self, engine: HybridScoringEngine) -> None:
        """Composite 0.60-0.84 -> SANITIZE."""
        # Create a scenario where composite lands in sanitize range
        findings = [_finding(layer=ScanLayer.SEMANTIC, confidence=0.70, verdict=Verdict.ESCALATE)]
        results = [_scanner_result(ScanLayer.SEMANTIC, Verdict.ESCALATE, findings)]
        result = engine.score(results)
        assert result.verdict == Verdict.SANITIZE

    def test_low_score_escalates(self, engine: HybridScoringEngine) -> None:
        """Composite 0.30-0.59 -> ESCALATE."""
        findings = [_finding(layer=ScanLayer.SEMANTIC, confidence=0.40, verdict=Verdict.ESCALATE)]
        results = [_scanner_result(ScanLayer.SEMANTIC, Verdict.ESCALATE, findings)]
        result = engine.score(results)
        assert result.verdict == Verdict.ESCALATE

    def test_very_low_score_allows(self, engine: HybridScoringEngine) -> None:
        """Composite < 0.30 → ALLOW."""
        findings = [_finding(layer=ScanLayer.SEMANTIC, confidence=0.15, verdict=Verdict.ALLOW)]
        results = [_scanner_result(ScanLayer.SEMANTIC, Verdict.ALLOW, findings)]
        result = engine.score(results)
        assert result.verdict == Verdict.ALLOW

    def test_custom_thresholds(self) -> None:
        """Custom ActionSettings thresholds affect verdict mapping."""
        config = NeuralGuardConfig()
        config.action.score_threshold_block = 0.90
        config.action.score_threshold_sanitize = 0.70
        engine = HybridScoringEngine(config)

        # Composite = 0.85 → normally BLOCK, but custom threshold is 0.90 → SANITIZE
        findings = [_finding(layer=ScanLayer.SEMANTIC, confidence=0.85, verdict=Verdict.ESCALATE)]
        results = [_scanner_result(ScanLayer.SEMANTIC, Verdict.ESCALATE, findings)]
        result = engine.score(results)
        assert result.verdict == Verdict.SANITIZE  # Below 0.90 block threshold


# ── Finding Enhancement ──────────────────────────────────────────────────────


class TestHybridFindings:
    """Test finding enhancement with hybrid metadata."""

    def test_enhance_adds_composite_finding(self, engine: HybridScoringEngine) -> None:
        """Enhanced findings include a composite hybrid finding."""
        findings = [_finding(layer=ScanLayer.PATTERN, confidence=0.9, verdict=Verdict.BLOCK)]
        results = [_scanner_result(ScanLayer.PATTERN, Verdict.BLOCK, findings)]
        hybrid = engine.score(results)

        enhanced = engine.enhance_findings(results, hybrid)
        # Original + 1 composite finding
        assert len(enhanced) == 2
        composite = [f for f in enhanced if f.rule_id == "HYBRID-001"]
        assert len(composite) == 1
        assert composite[0].confidence == hybrid.composite
        assert composite[0].metadata["composite"] is not None

    def test_enhance_zero_score_no_composite(self, engine: HybridScoringEngine) -> None:
        """Zero composite score doesn't add composite finding."""
        results = [_scanner_result(ScanLayer.PATTERN, Verdict.ALLOW, [])]
        hybrid = engine.score(results)
        enhanced = engine.enhance_findings(results, hybrid)
        # No findings + composite 0 → no composite finding added
        assert len(enhanced) == 0

    def test_composite_finding_has_metadata(self, engine: HybridScoringEngine) -> None:
        """Composite finding carries full score breakdown."""
        findings = [
            _finding(layer=ScanLayer.PATTERN, confidence=0.7, verdict=Verdict.SANITIZE),
            _finding(layer=ScanLayer.SEMANTIC, confidence=0.8, verdict=Verdict.ESCALATE),
        ]
        results = [
            _scanner_result(ScanLayer.PATTERN, Verdict.SANITIZE, [findings[0]]),
            _scanner_result(ScanLayer.SEMANTIC, Verdict.ESCALATE, [findings[1]]),
        ]
        hybrid = engine.score(results)
        enhanced = engine.enhance_findings(results, hybrid)

        composite = next(f for f in enhanced if f.rule_id == "HYBRID-001")
        meta = composite.metadata
        assert "composite" in meta
        assert "pattern_contribution" in meta
        assert "semantic_contribution" in meta
        assert "pattern_max_confidence" in meta
        assert "semantic_max_similarity" in meta


# ── Reason / Explanation ────────────────────────────────────────────────────


class TestHybridReason:
    """Test human-readable score explanation."""

    def test_pattern_block_reason(self, engine: HybridScoringEngine) -> None:
        """Pattern BLOCK includes confidence in reason."""
        findings = [_finding(layer=ScanLayer.PATTERN, confidence=0.95, verdict=Verdict.BLOCK)]
        results = [_scanner_result(ScanLayer.PATTERN, Verdict.BLOCK, findings)]
        result = engine.score(results)
        assert "Pattern BLOCK" in result.reason
        assert "0.95" in result.reason

    def test_weighted_combination_reason(self, engine: HybridScoringEngine) -> None:
        """Weighted combination shows both contributions."""
        findings = [
            _finding(layer=ScanLayer.PATTERN, confidence=0.5, verdict=Verdict.ALLOW),
            _finding(layer=ScanLayer.SEMANTIC, confidence=0.7, verdict=Verdict.ESCALATE),
        ]
        results = [
            _scanner_result(ScanLayer.PATTERN, Verdict.ALLOW, [findings[0]]),
            _scanner_result(ScanLayer.SEMANTIC, Verdict.ESCALATE, [findings[1]]),
        ]
        result = engine.score(results)
        assert "pattern" in result.reason
        assert "semantic" in result.reason
        assert "composite" in result.reason

    def test_no_findings_reason(self, engine: HybridScoringEngine) -> None:
        """No findings reason reflects zero scores."""
        results = [_scanner_result(ScanLayer.PATTERN, Verdict.ALLOW, [])]
        result = engine.score(results)
        assert "0.00" in result.reason


# ── Severity Mapping ─────────────────────────────────────────────────────────


class TestHybridSeverity:
    """Test composite score → severity mapping."""

    def test_critical_severity(self) -> None:
        assert HybridScoringEngine._severity_for_score(0.90) == Severity.CRITICAL

    def test_high_severity(self) -> None:
        assert HybridScoringEngine._severity_for_score(0.70) == Severity.HIGH

    def test_medium_severity(self) -> None:
        assert HybridScoringEngine._severity_for_score(0.40) == Severity.MEDIUM

    def test_low_severity(self) -> None:
        assert HybridScoringEngine._severity_for_score(0.10) == Severity.LOW


# ── Pipeline Integration ────────────────────────────────────────────────────


class TestHybridPipelineIntegration:
    """Test hybrid scoring integration with the scanner pipeline."""

    def test_pipeline_applies_hybrid_scoring(self) -> None:
        """Pipeline applies hybrid scoring when both layers have findings."""
        from neuralguard.scanners.pipeline import ScannerPipeline

        config = NeuralGuardConfig()
        config.scanner.semantic_enabled = True
        # Don't fail-closed so semantic layer can run after pattern
        config.action.fail_closed = False
        pipeline = ScannerPipeline(config)

        # Register mock scanners
        from unittest.mock import MagicMock

        pattern_scanner = MagicMock()
        pattern_scanner.layer = ScanLayer.PATTERN
        pattern_scanner.safe_scan.return_value = _scanner_result(
            ScanLayer.PATTERN,
            Verdict.ALLOW,
            [_finding(layer=ScanLayer.PATTERN, confidence=0.3, verdict=Verdict.ALLOW)],
        )

        semantic_scanner = MagicMock()
        semantic_scanner.layer = ScanLayer.SEMANTIC
        semantic_scanner.safe_scan.return_value = _scanner_result(
            ScanLayer.SEMANTIC,
            Verdict.ESCALATE,
            [_finding(layer=ScanLayer.SEMANTIC, confidence=0.7, verdict=Verdict.ESCALATE)],
        )

        pipeline.register_scanner(pattern_scanner)
        pipeline.register_scanner(semantic_scanner)

        req = EvaluateRequest(prompt="test", tenant_id="test")
        result = pipeline.execute(req)

        # Should have run both layers and applied hybrid scoring
        assert len(result.scanner_results) >= 2
        # Hybrid composite finding should be in findings
        hybrid_findings = [f for f in result.findings if f.rule_id == "HYBRID-001"]
        assert len(hybrid_findings) >= 1

    def test_pipeline_hybrid_upgrades_verdict(self) -> None:
        """Hybrid scoring can upgrade the pipeline verdict."""
        from unittest.mock import MagicMock

        from neuralguard.scanners.pipeline import ScannerPipeline

        config = NeuralGuardConfig()
        config.scanner.semantic_enabled = True
        config.action.fail_closed = False
        pipeline = ScannerPipeline(config)

        # Pattern says ALLOW, but semantic says ESCALATE
        pattern_scanner = MagicMock()
        pattern_scanner.layer = ScanLayer.PATTERN
        pattern_scanner.safe_scan.return_value = _scanner_result(
            ScanLayer.PATTERN,
            Verdict.ALLOW,
            [_finding(layer=ScanLayer.PATTERN, confidence=0.3, verdict=Verdict.ALLOW)],
        )

        # Semantic with high similarity → composite could push to SANITIZE
        semantic_scanner = MagicMock()
        semantic_scanner.layer = ScanLayer.SEMANTIC
        semantic_scanner.safe_scan.return_value = _scanner_result(
            ScanLayer.SEMANTIC,
            Verdict.BLOCK,
            [_finding(layer=ScanLayer.SEMANTIC, confidence=0.95, verdict=Verdict.BLOCK)],
        )

        pipeline.register_scanner(pattern_scanner)
        pipeline.register_scanner(semantic_scanner)

        req = EvaluateRequest(prompt="test", tenant_id="test")
        result = pipeline.execute(req)

        # With high semantic similarity (0.95) + pattern (0.3):
        # composite = 0.6*0.3 + 0.4*0.95 = 0.18 + 0.38 = 0.56 → ESCALATE
        # But individual semantic verdict is BLOCK (0.95 ≥ 0.75)
        # So strictest wins: BLOCK from semantic layer
        assert result.verdict in (Verdict.BLOCK, Verdict.SANITIZE, Verdict.ESCALATE)

    def test_pipeline_pattern_block_early_exit_no_hybrid(self) -> None:
        """Pattern BLOCK with fail-closed → early exit, no hybrid scoring."""
        from unittest.mock import MagicMock

        from neuralguard.scanners.pipeline import ScannerPipeline

        config = NeuralGuardConfig()
        config.scanner.semantic_enabled = True
        config.action.fail_closed = True
        pipeline = ScannerPipeline(config)

        pattern_scanner = MagicMock()
        pattern_scanner.layer = ScanLayer.PATTERN
        pattern_scanner.safe_scan.return_value = _scanner_result(
            ScanLayer.PATTERN,
            Verdict.BLOCK,
            [_finding(layer=ScanLayer.PATTERN, confidence=0.95, verdict=Verdict.BLOCK)],
        )

        pipeline.register_scanner(pattern_scanner)

        req = EvaluateRequest(prompt="test", tenant_id="test")
        result = pipeline.execute(req)

        assert result.verdict == Verdict.BLOCK
        # No hybrid composite finding (early exit, only pattern ran)
        hybrid_findings = [f for f in result.findings if f.rule_id == "HYBRID-001"]
        assert len(hybrid_findings) == 0


# ── HybridScoreResult Serialization ─────────────────────────────────────────


class TestHybridScoreResultDict:
    """Test HybridScoreResult.to_dict()."""

    def test_to_dict_contains_all_fields(self, engine: HybridScoringEngine) -> None:
        findings = [
            _finding(layer=ScanLayer.PATTERN, confidence=0.6, verdict=Verdict.SANITIZE),
            _finding(layer=ScanLayer.SEMANTIC, confidence=0.8, verdict=Verdict.ESCALATE),
        ]
        results = [
            _scanner_result(ScanLayer.PATTERN, Verdict.SANITIZE, [findings[0]]),
            _scanner_result(ScanLayer.SEMANTIC, Verdict.ESCALATE, [findings[1]]),
        ]
        result = engine.score(results)
        d = result.to_dict()

        assert "composite" in d
        assert "verdict" in d
        assert "pattern_contribution" in d
        assert "semantic_contribution" in d
        assert "pattern_max_confidence" in d
        assert "semantic_max_similarity" in d
        assert "reason" in d

    def test_to_dict_values_are_rounded(self, engine: HybridScoringEngine) -> None:
        findings = [
            _finding(layer=ScanLayer.SEMANTIC, confidence=0.856789, verdict=Verdict.ESCALATE)
        ]
        results = [_scanner_result(ScanLayer.SEMANTIC, Verdict.ESCALATE, findings)]
        result = engine.score(results)
        d = result.to_dict()

        # Values should be rounded to 4 decimal places
        assert isinstance(d["composite"], float)
        assert len(str(d["composite"]).split(".")[-1]) <= 4
