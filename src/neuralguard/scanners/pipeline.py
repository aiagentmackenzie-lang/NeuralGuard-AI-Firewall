"""Scanner pipeline — orchestrates multi-layer scanning and Layer Arbitration.

Pipeline execution order:
  1. Structural (sanitization + normalization)
  2. Pattern (regex/heuristic — <5ms)
  3. Semantic (embedding/ML — <50ms, Phase 2)
  4. Judge (LLM-as-Judge — <500ms, Phase 2)

Arbitration rule: Strictest verdict wins.
  BLOCK > SANITIZE > ESCALATE > QUARANTINE > RATE_LIMIT > ALLOW
  BLOCK cannot be overridden without explicit FORCE_ALLOW audit trail.
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

    def execute(self, request: EvaluateRequest) -> LayerArbitrationResult:
        """Run all enabled scanner layers and arbitrate results."""
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
                layer=layer.value,
                verdict=result.verdict.value,
                findings=len(result.findings),
                latency_ms=f"{result.latency_ms:.2f}",
                error=result.error,
            )

            # Early exit on BLOCK if fail-closed (don't waste time on deeper layers)
            if result.verdict == Verdict.BLOCK and self.config.action.fail_closed:
                logger.info("pipeline_early_exit", reason="block_verdict_fail_closed")
                break

        total_ms = (time.perf_counter() - start) * 1000
        final_verdict, reason = self._arbitrate(results)

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

    def _arbitrate(self, results: list[ScannerResult]) -> tuple[Verdict, str]:
        """Layer Arbitration — strictest verdict wins.

        Priority: BLOCK > SANITIZE > ESCALATE > QUARANTINE > RATE_LIMIT > ALLOW
        """
        if not results:
            # No scanners ran — fail-closed returns BLOCK, otherwise ALLOW
            if self.config.action.fail_closed:
                return Verdict.BLOCK, "No scanners executed; fail-closed default"
            return Verdict.ALLOW, "No scanners executed; fail-open default"

        # Find the highest-priority verdict
        max_priority = -1
        winning_verdict = Verdict.ALLOW
        winning_layer = "none"

        for result in results:
            priority = _VERDICT_PRIORITY.get(result.verdict, 0)
            if priority > max_priority:
                max_priority = priority
                winning_verdict = result.verdict
                winning_layer = result.layer.value

        # Build arbitration reason
        verdicts_seen = [f"{r.layer.value}={r.verdict.value}" for r in results]
        reason = f"Strictest verdict: {winning_verdict.value} from {winning_layer} layer. All: [{', '.join(verdicts_seen)}]"

        return winning_verdict, reason
