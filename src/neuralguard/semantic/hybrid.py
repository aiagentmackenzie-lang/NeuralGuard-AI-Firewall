"""Hybrid scoring engine — combines pattern confidence and semantic similarity.

Produces a composite risk score from multiple scanner layers and maps
it to a calibrated verdict. This is what makes NeuralGuard genuinely
better than regex alone — a prompt that barely misses pattern thresholds
but has high semantic similarity still gets flagged.

Scoring formula:
    composite = w_pattern * pattern_max_confidence + w_semantic * semantic_max_similarity

Default weights: w_pattern=0.6, w_semantic=0.4
(pattern gets higher weight because regex matches are more precise)

Verdict mapping:
    composite ≥ 0.85 → BLOCK
    composite ≥ 0.60 → SANITIZE
    composite ≥ 0.30 → ESCALATE (triggers LLM-as-Judge gate in Chunk 2.5)
    composite < 0.30 → ALLOW

Design:
  - Weights and thresholds configurable via env vars
  - Pattern BLOCK always wins (early exit, no semantic needed)
  - Semantic-only findings still contribute to composite score
  - If both layers find nothing, composite = 0.0 → ALLOW
  - Findings carry both the composite score and per-layer breakdown
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from neuralguard.models.schemas import (
    Finding,
    ScanLayer,
    ScannerResult,
    Severity,
    ThreatCategory,
    Verdict,
)

if TYPE_CHECKING:
    from neuralguard.config.settings import NeuralGuardConfig

logger = structlog.get_logger(__name__)


class HybridScoreResult:
    """Result from hybrid scoring engine.

    Attributes:
        composite: Weighted composite risk score (0.0-1.0).
        verdict: Calibrated verdict based on composite score.
        pattern_contribution: Pattern layer's weighted contribution.
        semantic_contribution: Semantic layer's weighted contribution.
        pattern_max_confidence: Raw max confidence from pattern findings.
        semantic_max_similarity: Raw max similarity from semantic findings.
        reason: Human-readable explanation of the score.
    """

    def __init__(
        self,
        composite: float,
        verdict: Verdict,
        pattern_contribution: float,
        semantic_contribution: float,
        pattern_max_confidence: float,
        semantic_max_similarity: float,
        reason: str,
    ) -> None:
        self.composite = composite
        self.verdict = verdict
        self.pattern_contribution = pattern_contribution
        self.semantic_contribution = semantic_contribution
        self.pattern_max_confidence = pattern_max_confidence
        self.semantic_max_similarity = semantic_max_similarity
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for audit/metadata."""
        return {
            "composite": round(self.composite, 4),
            "verdict": self.verdict.value,
            "pattern_contribution": round(self.pattern_contribution, 4),
            "semantic_contribution": round(self.semantic_contribution, 4),
            "pattern_max_confidence": round(self.pattern_max_confidence, 4),
            "semantic_max_similarity": round(self.semantic_max_similarity, 4),
            "reason": self.reason,
        }


class HybridScoringEngine:
    """Combines pattern and semantic scores into a composite risk score.

    Usage:
        engine = HybridScoringEngine(config)
        result = engine.score(scanner_results)
        # result.composite → 0.0 to 1.0
        # result.verdict → BLOCK / SANITIZE / ESCALATE / ALLOW
    """

    # Default weights — pattern is more precise, gets higher weight
    DEFAULT_PATTERN_WEIGHT = 0.6
    DEFAULT_SEMANTIC_WEIGHT = 0.4

    def __init__(self, config: NeuralGuardConfig) -> None:
        self.config = config
        self._pattern_weight = self.DEFAULT_PATTERN_WEIGHT
        self._semantic_weight = self.DEFAULT_SEMANTIC_WEIGHT

    @property
    def pattern_weight(self) -> float:
        return self._pattern_weight

    @property
    def semantic_weight(self) -> float:
        return self._semantic_weight

    def set_weights(self, pattern: float, semantic: float) -> None:
        """Override default weights. Must sum to 1.0."""
        if abs(pattern + semantic - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {pattern} + {semantic} = {pattern + semantic}"
            )
        self._pattern_weight = pattern
        self._semantic_weight = semantic

    def score(self, scanner_results: list[ScannerResult]) -> HybridScoreResult:
        """Compute hybrid score from all scanner results.

        Args:
            scanner_results: Results from all scanner layers that ran.

        Returns:
            HybridScoreResult with composite score and calibrated verdict.
        """
        # Extract per-layer signals
        pattern_max = self._extract_pattern_confidence(scanner_results)
        semantic_max = self._extract_semantic_similarity(scanner_results)

        # Pattern BLOCK is absolute — if pattern says BLOCK, composite = pattern confidence
        pattern_blocked = self._is_pattern_blocked(scanner_results)

        # Weighted composite
        if pattern_blocked:
            # Pattern BLOCK is high-confidence — give it full weight
            composite = pattern_max
            pattern_contrib = pattern_max
            semantic_contrib = 0.0
        elif pattern_max > 0.0 and semantic_max > 0.0:
            # Both layers found something — weighted combination
            pattern_contrib = self._pattern_weight * pattern_max
            semantic_contrib = self._semantic_weight * semantic_max
            composite = pattern_contrib + semantic_contrib
        elif pattern_max > 0.0:
            # Only pattern found something
            composite = pattern_max
            pattern_contrib = pattern_max
            semantic_contrib = 0.0
        elif semantic_max > 0.0:
            # Only semantic found something
            composite = semantic_max
            pattern_contrib = 0.0
            semantic_contrib = semantic_max
        else:
            # Nothing found
            composite = 0.0
            pattern_contrib = 0.0
            semantic_contrib = 0.0

        # Calibrate verdict from composite score
        verdict = self._composite_to_verdict(composite)

        # Build explanation
        reason = self._build_reason(
            composite,
            verdict,
            pattern_max,
            semantic_max,
            pattern_contrib,
            semantic_contrib,
            pattern_blocked,
        )

        logger.debug(
            "hybrid_score",
            composite=f"{composite:.4f}",
            verdict=verdict.value,
            pattern_max=f"{pattern_max:.4f}",
            semantic_max=f"{semantic_max:.4f}",
            pattern_contrib=f"{pattern_contrib:.4f}",
            semantic_contrib=f"{semantic_contrib:.4f}",
        )

        return HybridScoreResult(
            composite=composite,
            verdict=verdict,
            pattern_contribution=pattern_contrib,
            semantic_contribution=semantic_contrib,
            pattern_max_confidence=pattern_max,
            semantic_max_similarity=semantic_max,
            reason=reason,
        )

    def enhance_findings(
        self,
        scanner_results: list[ScannerResult],
        hybrid_result: HybridScoreResult,
    ) -> list[Finding]:
        """Add hybrid score metadata to existing findings.

        Creates a composite finding that carries the hybrid score,
        plus enriches existing findings with the score context.
        """
        findings: list[Finding] = []

        # Collect all existing findings
        for result in scanner_results:
            findings.extend(result.findings)

        # Add hybrid composite finding if score is meaningful
        if hybrid_result.composite > 0.0:
            composite_finding = Finding(
                category=ThreatCategory.SELF_ATTACK,
                severity=self._severity_for_score(hybrid_result.composite),
                verdict=hybrid_result.verdict,
                confidence=hybrid_result.composite,
                layer=ScanLayer.SEMANTIC,  # Hybrid is part of semantic processing
                rule_id="HYBRID-001",
                description=hybrid_result.reason,
                mitigation="Composite risk score from pattern + semantic analysis",
                metadata=hybrid_result.to_dict(),
            )
            findings.append(composite_finding)

        return findings

    def _extract_pattern_confidence(self, results: list[ScannerResult]) -> float:
        """Get max confidence from pattern layer findings."""
        for result in results:
            if result.layer == ScanLayer.PATTERN and result.findings:
                max_conf = max(f.confidence for f in result.findings)
                return max_conf
        return 0.0

    def _extract_semantic_similarity(self, results: list[ScannerResult]) -> float:
        """Get max similarity from semantic layer findings."""
        for result in results:
            if result.layer == ScanLayer.SEMANTIC and result.findings:
                max_sim = max(f.confidence for f in result.findings)
                return max_sim
        return 0.0

    def _is_pattern_blocked(self, results: list[ScannerResult]) -> bool:
        """Check if pattern layer returned a BLOCK verdict."""
        for result in results:
            if result.layer == ScanLayer.PATTERN and result.verdict == Verdict.BLOCK:
                return True
        return False

    def _composite_to_verdict(self, composite: float) -> Verdict:
        """Map composite score to verdict using config thresholds.

        Uses ActionSettings thresholds for consistency with the API.
        """
        block_threshold = self.config.action.score_threshold_block  # 0.85
        sanitize_threshold = self.config.action.score_threshold_sanitize  # 0.60
        escalate_floor = 0.30  # Fixed — triggers LLM-as-Judge gate

        if composite >= block_threshold:
            return Verdict.BLOCK
        if composite >= sanitize_threshold:
            return Verdict.SANITIZE
        if composite >= escalate_floor:
            return Verdict.ESCALATE
        return Verdict.ALLOW

    @staticmethod
    def _severity_for_score(composite: float) -> Severity:
        """Map composite score to severity level."""
        if composite >= 0.85:
            return Severity.CRITICAL
        if composite >= 0.60:
            return Severity.HIGH
        if composite >= 0.30:
            return Severity.MEDIUM
        return Severity.LOW

    def _build_reason(
        self,
        composite: float,
        verdict: Verdict,
        pattern_max: float,
        semantic_max: float,
        pattern_contrib: float,
        semantic_contrib: float,
        pattern_blocked: bool,
    ) -> str:
        """Build human-readable explanation of the hybrid score."""
        parts: list[str] = []

        if pattern_blocked:
            parts.append(f"Pattern BLOCK (confidence={pattern_max:.2f})")
        else:
            if pattern_max > 0.0:
                parts.append(
                    f"pattern={pattern_max:.2f} (w={self._pattern_weight:.1f} → {pattern_contrib:.3f})"
                )
            else:
                parts.append("pattern=0.00 (no findings)")

            if semantic_max > 0.0:
                parts.append(
                    f"semantic={semantic_max:.2f} (w={self._semantic_weight:.1f} → {semantic_contrib:.3f})"
                )
            else:
                parts.append("semantic=0.00 (no matches)")

        parts.append(f"composite={composite:.3f}→{verdict.value}")
        return " | ".join(parts)
