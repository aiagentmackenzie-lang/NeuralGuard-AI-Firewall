"""BLOCK action handler — return 403 forbidden."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ActionResult, BaseAction

if TYPE_CHECKING:
    from neuralguard.models.schemas import LayerArbitrationResult


class BlockAction(BaseAction):
    """Handle BLOCK verdict."""

    def execute(
        self,
        arbitration: LayerArbitrationResult,
        request: object,  # EvaluateRequest | ScanOutputRequest
    ) -> ActionResult:
        confidence = max((f.confidence for f in arbitration.findings), default=1.0)
        return ActionResult(
            status_code=403,
            body={
                "error": "request_blocked",
                "message": "Request blocked by NeuralGuard firewall",
                "verdict": "block",
                "findings": [f.model_dump() for f in arbitration.findings],
                "confidence": confidence,
            },
            headers={
                "X-NeuralGuard-Verdict": "block",
            },
        )
