"""RATE_LIMIT action handler — return 429 with Retry-After."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ActionResult, BaseAction

if TYPE_CHECKING:
    from neuralguard.models.schemas import LayerArbitrationResult


class RateLimitAction(BaseAction):
    """Handle RATE_LIMIT verdict."""

    def execute(
        self,
        arbitration: LayerArbitrationResult,
        request: object,  # EvaluateRequest | ScanOutputRequest
    ) -> ActionResult:
        retry_after = 60  # seconds

        confidence = max((f.confidence for f in arbitration.findings), default=0.0)

        return ActionResult(
            status_code=429,
            body={
                "error": "rate_limited",
                "message": "Request rate limited due to suspicious activity",
                "verdict": "rate_limit",
                "retry_after": retry_after,
                "findings": [f.model_dump() for f in arbitration.findings],
                "confidence": confidence,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-NeuralGuard-Verdict": "rate_limit",
            },
        )
