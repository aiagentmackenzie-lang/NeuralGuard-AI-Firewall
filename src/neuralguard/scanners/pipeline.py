"""Scanner pipeline — orchestrates multi-layer scanning, hybrid scoring,
and Layer Arbitration.

Pipeline execution order:
  1. Structural (sanitization + normalization)
  2. Pattern (regex/heuristic - <5ms)
  3. Semantic (embedding/ML - <50ms, Phase 2)
     → Hybrid Scoring (combines pattern + semantic)
  4. Judge (LLM-as-Judge - <500ms, Phase 2, gated by hybrid score)

Arbitration rule (with hybrid scoring):
  - Pattern BLOCK always wins (high-precision, early exit)
  - Hybrid scoring can upgrade verdicts (e.g. ALLOW -> SANITIZE)
  - Judge only fires in ambiguous zone (composite 0.30-0.70)
  - BLOCK cannot be overridden without explicit FORCE_ALLOW audit trail
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from neuralguard.models.schemas import (
    EvaluateRequest,
    Finding,
    LayerArbitrationResult,
    ScanLayer,
    ScannerResult,
    Verdict,
)

if TYPE_CHECKING:
    from neuralguard.config.settings import NeuralGuardConfig
    from neuralguard.scanners.base import BaseScanner

logger = structlog.get_logger(__name__)

# Verdict strictness ordering for arbitration
_VERDICT_PRIORITY: dict[Verdict, int] = {
    Verdict.BLOCK: 6,
    Verdict.SANITIZE: 5,
    Verdict.ESCALATE: 4,
    Verdict.QUARANTINE: 3,
    Verdict.RATE_LIMIT: 2,
    Verdict.ALLOW: 0,
}


class ScannerPipeline:
    """Orchestrates the multi-layer scanner pipeline."""

    def __init__(self, config: NeuralGuardConfig) -> None:
        self.config = config
        self._scanners: dict[ScanLayer, BaseScanner] = {}
        self._layer_order: list[ScanLayer] = [
            ScanLayer.STRUCTURAL,
            ScanLayer.PATTERN,
            ScanLayer.SEMANTIC,
            ScanLayer.JUDGE,
        ]
        self._hybrid_engine: Any = None  # Lazy init

    def register_scanner(self, scanner: BaseScanner) -> None:
        """Register a scanner for its layer."""
        self._scanners[scanner.layer] = scanner
        logger.info("scanner_registered", layer=scanner.layer.value)

    def unregister_scanner(self, layer: ScanLayer) -> None:
        """Remove a scanner by layer."""
        self._scanners.pop(layer, None)
        logger.info("scanner_unregistered", layer=layer.value)

    def get_enabled_layers(self, request: EvaluateRequest | None = None) -> list[ScanLayer]:
        """Determine which layers to run based on config and request overrides."""
        # Request-level override
        if request and request.scanners is not None:
            return [l for l in self._layer_order if l in request.scanners]

        # Config-level defaults
        layers = [ScanLayer.STRUCTURAL, ScanLayer.PATTERN]
        if self.config.scanner.semantic_enabled:
            layers.append(ScanLayer.SEMANTIC)
        if self.config.scanner.judge_enabled:
            layers.append(ScanLayer.JUDGE)
        return layers

    @property
    def hybrid_engine(self) -> Any:
        """Lazy-initialize the hybrid scoring engine."""
        if self._hybrid_engine is None:
            try:
                from neuralguard.semantic.hybrid import HybridScoringEngine

                self._hybrid_engine = HybridScoringEngine(self.config)
            except ImportError:
                logger.debug("hybrid_engine_unavailable", msg="semantic extra not installed")
        return self._hybrid_engine

    def execute(self, request: EvaluateRequest) -> LayerArbitrationResult:
        """Run all enabled scanner layers, apply hybrid scoring, and arbitrate results."""
        start = time.perf_counter()
        layers = self.get_enabled_layers(request)
        results: list[ScannerResult] = []
        all_findings: list[Finding] = []
        context: dict[str, Any] = {}

        logger.info(
            "pipeline_start",
            tenant=request.tenant_id,
            use_case=request.use_case,
            layers=[l.value for l in layers],
        )

        for layer in layers:
            scanner = self._scanners.get(layer)
            if scanner is None:
                logger.debug("scanner_skip_not_registered", layer=layer.value)
                continue

            logger.debug("scanner_start", layer=layer.value)
            result = scanner.safe_scan(request, context)
            results.append(result)

            # Pass findings to context for downstream scanners
            context[f"{layer.value}_verdict"] = result.verdict
            context[f"{layer.value}_findings"] = result.findings
            if result.sanitized_output:
                context["sanitized_input"] = result.sanitized_output

            all_findings.extend(result.findings)

            logger.info(
                "scanner_complete",
                layer=layer.layer.value if hasattr(layer, "layer") else layer.value,
                verdict=result.verdict.value,
                findings=len(result.findings),
                latency_ms=f"{result.latency_ms:.2f}",
                error=result.error,
            )

            # Early exit on BLOCK if fail-closed
            if result.verdict == Verdict.BLOCK and self.config.action.fail_closed:
                logger.info("pipeline_early_exit", reason="block_verdict_fail_closed")
                break

            # After semantic layer: apply hybrid scoring and inject into context
            # so the Judge scanner can use it for its gate check
            if layer == ScanLayer.SEMANTIC:
                self._apply_hybrid_to_context(results, context)

        total_ms = (time.perf_counter() - start) * 1000

        # Final hybrid scoring (if not already done via context injection)
        hybrid_result = context.get("_hybrid_result")
        final_verdict, reason = self._arbitrate(results, hybrid_result)

        # Enhance findings with hybrid metadata
        if hybrid_result is not None and self.hybrid_engine is not None:
            all_findings = self.hybrid_engine.enhance_findings(results, hybrid_result)

        logger.info(
            "pipeline_complete",
            verdict=final_verdict.value,
            total_findings=len(all_findings),
            total_latency_ms=f"{total_ms:.2f}",
            reason=reason,
        )

        return LayerArbitrationResult(
            verdict=final_verdict,
            findings=all_findings,
            scanner_results=results,
            total_latency_ms=total_ms,
            arbitration_reason=reason,
        )

    def _apply_hybrid_to_context(
        self,
        results: list[ScannerResult],
        context: dict[str, Any],
    ) -> None:
        """Apply hybrid scoring and inject result into pipeline context.

        This runs after the semantic layer, so the Judge scanner can check
        the hybrid composite score to decide whether to fire.
        """
        has_pattern = any(r.layer == ScanLayer.PATTERN for r in results)
        has_semantic = any(r.layer == ScanLayer.SEMANTIC for r in results)
        any_findings = any(len(r.findings) > 0 for r in results)

        if has_pattern and has_semantic and any_findings and self.hybrid_engine is not None:
            hybrid_result = self.hybrid_engine.score(results)
            context["_hybrid_result"] = hybrid_result
            logger.info(
                "hybrid_score_applied",
                composite=f"{hybrid_result.composite:.4f}",
                hybrid_verdict=hybrid_result.verdict.value,
                pattern_max=f"{hybrid_result.pattern_max_confidence:.4f}",
                semantic_max=f"{hybrid_result.semantic_max_similarity:.4f}",
            )

    def _arbitrate(
        self,
        results: list[ScannerResult],
        hybrid_result: Any | None = None,
    ) -> tuple[Verdict, str]:
        """Layer Arbitration - strictest verdict wins, hybrid can upgrade.

        Priority: BLOCK > SANITIZE > ESCALATE > QUARANTINE > RATE_LIMIT > ALLOW
        Hybrid scoring can upgrade a verdict but can never downgrade.
        """
        if not results:
            if self.config.action.fail_closed:
                return Verdict.BLOCK, "No scanners executed; fail-closed default"
            return Verdict.ALLOW, "No scanners executed; fail-open default"

        # Find the highest-priority verdict from scanner layers
        max_priority = -1
        winning_verdict = Verdict.ALLOW
        winning_layer = "none"

        for result in results:
            priority = _VERDICT_PRIORITY.get(result.verdict, 0)
            if priority > max_priority:
                max_priority = priority
                winning_verdict = result.verdict
                winning_layer = result.layer.value

        # Check if hybrid scoring would upgrade the verdict
        if hybrid_result is not None:
            hybrid_priority = _VERDICT_PRIORITY.get(hybrid_result.verdict, 0)
            if hybrid_priority > max_priority:
                winning_verdict = hybrid_result.verdict
                winning_layer = "hybrid"
                max_priority = hybrid_priority

        # Build arbitration reason
        verdicts_seen = [f"{r.layer.value}={r.verdict.value}" for r in results]
        hybrid_info = ""
        if hybrid_result is not None:
            hybrid_info = (
                f" | Hybrid: composite={hybrid_result.composite:.3f}->{hybrid_result.verdict.value}"
            )
        reason = (
            f"Strictest verdict: {winning_verdict.value} from {winning_layer} layer. "
            f"All: [{', '.join(verdicts_seen)}]{hybrid_info}"
        )

        return winning_verdict, reason
