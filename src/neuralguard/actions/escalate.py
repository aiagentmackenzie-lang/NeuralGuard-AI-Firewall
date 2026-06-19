"""ESCALATE action handler — send to human review queue / webhook."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx

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
        """Send escalation webhook using async httpx when in an async context.

        Falls back to synchronous httpx.Client if no event loop is available
        (e.g., CLI or test contexts).
        """
        payload = {
            "verdict": "escalate",
            "findings_count": len(arbitration.findings),
            "arbitration_reason": arbitration.arbitration_reason,
        }
        url = self.config.action.escalation_webhook_url
        if url is None:
            return False

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop — use synchronous client
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.post(url, json=payload)
                    return response.status_code < 400
            except Exception:
                return False

        # We're in an async context — schedule the webhook as a fire-and-forget task
        async def _async_send() -> bool:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(url, json=payload)
                    return response.status_code < 400
            except Exception:
                return False

        loop.create_task(_async_send())
        return True  # Task scheduled; actual delivery is best-effort
