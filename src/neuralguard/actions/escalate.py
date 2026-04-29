"""ESCALATE action handler — send to human review queue / webhook."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ActionResult, BaseAction

if TYPE_CHECKING:
    from neuralguard.models.schemas import LayerArbitrationResult


class EscalateAction(BaseAction):
    """Handle ESCALATE verdict."""

    def execute(
        self,
        arbitration: LayerArbitrationResult,
        request: object,  # EvaluateRequest | ScanOutputRequest
    ) -> ActionResult:
        webhook_sent = False
        if self.config.action.escalation_webhook_url:
            webhook_sent = self._send_webhook(arbitration)

        confidence = max((f.confidence for f in arbitration.findings), default=0.0)

        return ActionResult(
            status_code=202,
            body={
                "verdict": "escalate",
                "message": "Request escalated to human review",
                "findings": [f.model_dump() for f in arbitration.findings],
                "confidence": confidence,
                "webhook_sent": webhook_sent,
            },
            headers={
                "X-NeuralGuard-Verdict": "escalate",
            },
        )

    def _send_webhook(self, arbitration: LayerArbitrationResult) -> bool:
        try:
            import httpx  # type: ignore[import-untyped]

            with httpx.Client(timeout=5.0) as client:
                response = client.post(
                    self.config.action.escalation_webhook_url,
                    json={
                        "verdict": "escalate",
                        "findings_count": len(arbitration.findings),
                        "arbitration_reason": arbitration.arbitration_reason,
                    },
                )
                return response.status_code < 400
        except Exception:
            return False
