"""QUARANTINE action handler — flag tenant, return 202 with warning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ActionResult, BaseAction

if TYPE_CHECKING:
    from neuralguard.models.schemas import LayerArbitrationResult


class QuarantineAction(BaseAction):
    """Handle QUARANTINE verdict."""

    def execute(
        self,
        arbitration: LayerArbitrationResult,
        request: object,  # EvaluateRequest | ScanOutputRequest
    ) -> ActionResult:
        tenant_id = getattr(request, "tenant_id", "default")
        confidence = max((f.confidence for f in arbitration.findings), default=0.0)

        return ActionResult(
            status_code=202,
            body={
                "verdict": "quarantine",
                "message": "Request quarantined for review. Tenant flagged.",
                "tenant_id": tenant_id,
                "findings": [f.model_dump() for f in arbitration.findings],
                "confidence": confidence,
            },
            headers={
                "X-NeuralGuard-Verdict": "quarantine",
                "X-NeuralGuard-Tenant-Flagged": tenant_id,
            },
        )
