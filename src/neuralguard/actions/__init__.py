"""Actions package — response action handlers.

Orchestrates BLOCK / SANITIZE / ESCALATE / QUARANTINE / RATE_LIMIT
execution after Layer Arbitration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

        Unknown verdicts (including ALLOW) fall back to 200 OK.
        """
        handler = self._handlers.get(arbitration.verdict)
        if handler is None:
            confidence = max((f.confidence for f in arbitration.findings), default=0.0)
            return ActionResult(
                status_code=200,
                body={
                    "verdict": arbitration.verdict.value,
                    "findings": [f.model_dump() for f in arbitration.findings],
                    "confidence": confidence,
                },
                headers={"X-NeuralGuard-Verdict": arbitration.verdict.value},
            )
        return handler.execute(arbitration, request)
