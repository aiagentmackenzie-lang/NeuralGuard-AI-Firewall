"""Scanner pipeline — orchestrates multi-layer scanning, hybrid scoring,
and Layer Arbitration.

Pipeline execution order:
  1. Structural (sanitization + normalization)
  2. Agent Guardian (cross-turn accumulation signals — records the turn
     into the per-session window; ~0.03 ms regex sweep)
  3. Pattern (regex/heuristic - <5ms)
  4. Semantic (embedding/ML - <50ms, Phase 2)
     → Hybrid Scoring (combines pattern + semantic)
  5. Judge (LLM-as-Judge - <500ms, Phase 2, gated by hybrid score)

Agent Guardian runs BEFORE Pattern (F2): the fail-closed early exit on a
BLOCK verdict stops the pipeline after the blocking layer, so if AG ran
after Pattern, a Pattern-BLOCKed turn would never be recorded into the
session window and cross-turn accumulation would be blind to the
strongest turns. AG-before-Pattern costs nothing measurable and keeps
the accumulation counters complete. Trade-off (accepted): an AG-BLOCKed
turn early-exits before the Pattern layer runs, so Pattern's rule IDs
are absent from that turn's report — the verdict is BLOCK either way.

Arbitration rule (with hybrid scoring):
  - The strictest verdict wins: BLOCK > SANITIZE > ESCALATE > ...
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
    from neuralguard.tenants.registry import TenantConfigRegistry

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
        self._scanners: dict[ScanLayer, BaseScanner[Any]] = {}
        self._layer_order: list[ScanLayer] = [
            ScanLayer.STRUCTURAL,
            # AG before Pattern (F2): a Pattern-BLOCKed turn must still be
            # recorded into AG's session window — the fail-closed early exit
            # would otherwise skip AG entirely on the strongest turns.
            ScanLayer.AGENT_GUARDIAN,
            ScanLayer.PATTERN,
            ScanLayer.SEMANTIC,
            ScanLayer.JUDGE,
        ]
        self._hybrid_engine: Any = None  # Lazy init
        # Per-tenant override registry (Sprint C, C1). None when multi-tenant
        # mode is disabled -> the pipeline falls back to global config only
        # (backward compatible). Set via ``set_tenant_registry``.
        self._tenant_registry: TenantConfigRegistry | None = None

    def set_tenant_registry(self, registry: TenantConfigRegistry | None) -> None:
        """Install the per-tenant override registry (called from main.py)."""
        self._tenant_registry = registry

    def register_scanner(self, scanner: BaseScanner[Any]) -> None:
        """Register a scanner for its layer."""
        self._scanners[scanner.layer] = scanner
        logger.info("scanner_registered", layer=scanner.layer.value)

    def unregister_scanner(self, layer: ScanLayer) -> None:
        """Remove a scanner by layer."""
        self._scanners.pop(layer, None)
        logger.info("scanner_unregistered", layer=layer.value)

    def get_enabled_layers(self, request: EvaluateRequest | None = None) -> list[ScanLayer]:
        """Determine which layers to run based on config, tenant overrides, and request.

        Precedence (most-specific wins, but the tenant config is a *ceiling*):
        1. Resolve the globally-registered + config-enabled layers (baseline).
        2. Apply the per-tenant scanner ceiling (Sprint C, C1): a tenant may
           narrow the optional scanners (agent_guardian / semantic / judge)
           but can never disable the mandatory Structural + Pattern layers,
           and can never widen past what is globally registered.
        3. Apply the client ``request.scanners`` override as a further
           narrowing (intersection) — a client may opt OUT of layers but never
           opt IN past the tenant + global ceiling.
        """
        # 1. Global config baseline.
        layers = [ScanLayer.STRUCTURAL, ScanLayer.PATTERN]
        if self.config.agent_guardian.enabled:
            layers.append(ScanLayer.AGENT_GUARDIAN)
        if self.config.scanner.semantic_enabled:
            layers.append(ScanLayer.SEMANTIC)
        if self.config.scanner.judge_enabled:
            layers.append(ScanLayer.JUDGE)

        # 2. Per-tenant ceiling (only narrows the optional layers).
        registry = self._tenant_registry
        if registry is not None and registry.enabled and request is not None:
            overlay = registry.effective_scanner_overlay(request.tenant_id)
            if overlay is not None:
                optional_map = {
                    ScanLayer.AGENT_GUARDIAN: overlay.agent_guardian,
                    ScanLayer.SEMANTIC: overlay.semantic,
                    ScanLayer.JUDGE: overlay.judge,
                }
                narrowed: list[ScanLayer] = [ScanLayer.STRUCTURAL, ScanLayer.PATTERN]
                for layer in layers:
                    if layer in (ScanLayer.STRUCTURAL, ScanLayer.PATTERN):
                        continue
                    flag = optional_map.get(layer)
                    # flag is None -> inherit global (keep); True/False -> enforce.
                    if flag is False:
                        logger.info(
                            "tenant_scanner_disabled",
                            tenant=request.tenant_id,
                            layer=layer.value,
                        )
                        continue
                    if flag is True and layer not in layers:
                        # Tenant wants a scanner the global config didn't enable
                        # / register. We cannot conjure an unregistered scanner;
                        # keep it off and log (honest, fail-safe).
                        logger.info(
                            "tenant_scanner_unavailable",
                            tenant=request.tenant_id,
                            layer=layer.value,
                            msg="tenant enabled a scanner not registered globally",
                        )
                        continue
                    narrowed.append(layer)
                layers = narrowed

        # 3. Client request override — intersection only (narrow, never widen).
        if request and request.scanners is not None:
            layers = [l for l in self._layer_order if l in request.scanners and l in layers]

        # Canonical execution order (F2): project the resolved layer set onto
        # _layer_order so AG runs before Pattern regardless of how the set was
        # assembled (config baseline, tenant ceiling, or request override).
        return [l for l in self._layer_order if l in layers]

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

        # Judge resolution of the ambiguous zone (A2 FPR fix). An ESCALATE
        # verdict means "ambiguous, awaiting judge resolution." If the LLM
        # Judge ran and cleanly returned ALLOW (benign), it is the authoritative
        # resolver and downgrades ESCALATE -> ALLOW. The judge cannot downgrade
        # SANITIZE/BLOCK/QUARANTINE/RATE_LIMIT (only ESCALATE). A skipped,
        # timed-out, or errored judge does NOT resolve — its result carries an
        # error string, so the pre-judge ESCALATE stands (fail-closed on judge
        # uncertainty). This is what lets the judge drop the semantic-layer FPR
        # on benign creative/translation prompts that ESCALATE on a lone
        # ambiguous semantic match.
        if self.config.action.judge_resolves_escalate and winning_verdict == Verdict.ESCALATE:
            judge_results = [r for r in results if r.layer == ScanLayer.JUDGE]
            clean_allow = [r for r in judge_results if r.verdict == Verdict.ALLOW and not r.error]
            if judge_results and len(clean_allow) == len(judge_results):
                winning_verdict = Verdict.ALLOW
                winning_layer = "judge_resolve"
                max_priority = _VERDICT_PRIORITY[Verdict.ALLOW]

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
