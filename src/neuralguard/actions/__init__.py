"""Actions package — response action handlers.

Orchestrates BLOCK / SANITIZE / ESCALATE / QUARANTINE / RATE_LIMIT
execution after Layer Arbitration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuralguard.models.schemas import (
    EvaluateRequest,
    LayerArbitrationResult,
    ScanOutputRequest,
    Verdict,
)

from .base import ActionResult, BaseAction
from .block import BlockAction
from .escalate import EscalateAction
from .quarantine import QuarantineAction
from .ratelimit import RateLimitAction
from .sanitize import SanitizeAction

if TYPE_CHECKING:
    from neuralguard.config.settings import NeuralGuardConfig

__all__ = [
    "ActionDispatcher",
    "ActionResult",
    "BaseAction",
    "BlockAction",
    "EscalateAction",
    "QuarantineAction",
    "RateLimitAction",
    "SanitizeAction",
]


class ActionDispatcher:
    """Dispatches LayerArbitrationResult to the correct action handler."""

    VERDICT_MAP: dict[Verdict, type[BaseAction]] = {
        Verdict.BLOCK: BlockAction,
        Verdict.SANITIZE: SanitizeAction,
        Verdict.ESCALATE: EscalateAction,
        Verdict.QUARANTINE: QuarantineAction,
        Verdict.RATE_LIMIT: RateLimitAction,
    }

    def __init__(self, config: NeuralGuardConfig) -> None:
        self.config = config
        self._handlers: dict[Verdict, BaseAction] = {
            verdict: cls(config) for verdict, cls in self.VERDICT_MAP.items()
        }

    def execute(
        self,
        arbitration: LayerArbitrationResult,
        request: EvaluateRequest | ScanOutputRequest,
    ) -> ActionResult:
        """Dispatch to the appropriate action handler.

        ALLOW verdict returns a 200 response with all EvaluateResponse fields.
        """
        handler = self._handlers.get(arbitration.verdict)
        if handler is None:
            # ALLOW verdict — return full response body matching EvaluateResponse schema
            confidence = max((f.confidence for f in arbitration.findings), default=0.0)
            layers_used = [r.layer.value for r in arbitration.scanner_results]
            body: dict[str, Any] = {
                "verdict": arbitration.verdict.value,
                "findings": [f.model_dump() for f in arbitration.findings],
                "confidence": confidence,
                "scan_layers_used": layers_used,
                "total_latency_ms": arbitration.total_latency_ms,
            }
            # Add sanitized_content for SANITIZE verdict
            if arbitration.verdict == Verdict.SANITIZE:
                body["sanitized_content"] = None
            return ActionResult(
                status_code=200,
                body=body,
                headers={"X-NeuralGuard-Verdict": arbitration.verdict.value},
            )
        return handler.execute(arbitration, request)
