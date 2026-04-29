"""SANITIZE action handler — redact PII, return sanitized text."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import regex as re_module

# Import EXF raw patterns to reuse for PII redaction
from neuralguard.scanners.pattern import EXF_PATTERNS

from .base import ActionResult, BaseAction

if TYPE_CHECKING:
    from neuralguard.models.schemas import LayerArbitrationResult


class SanitizeAction(BaseAction):
    """Handle SANITIZE verdict."""

    def __init__(self, config: object) -> None:
        super().__init__(config)
        self._redaction_patterns: list[tuple[re_module.Pattern, str]] = []
        for rule_id, _severity, _confidence, _description, pattern_str in EXF_PATTERNS:
            try:
                compiled = re_module.compile(pattern_str, flags=re_module.IGNORECASE)
                self._redaction_patterns.append((compiled, f"[REDACTED:{rule_id}]"))
            except Exception:
                pass

    def execute(
        self,
        arbitration: LayerArbitrationResult,
        request: object,  # EvaluateRequest | ScanOutputRequest
    ) -> ActionResult:
        # Prefer structural sanitized output if available
        sanitized: str | None = None
        for result in arbitration.scanner_results:
            if result.sanitized_output:
                sanitized = result.sanitized_output
                break

        if sanitized is None:
            text = self._extract_text(request)
            sanitized = self._redact_pii(text)

        confidence = max((f.confidence for f in arbitration.findings), default=0.0)

        return ActionResult(
            status_code=200,
            body={
                "verdict": "sanitize",
                "sanitized_content": sanitized,
                "findings": [f.model_dump() for f in arbitration.findings],
                "confidence": confidence,
            },
            headers={
                "X-NeuralGuard-Verdict": "sanitize",
            },
        )

    def _extract_text(self, request: object) -> str:
        if hasattr(request, "output") and getattr(request, "output", None):
            return str(request.output)
        if hasattr(request, "prompt") and getattr(request, "prompt", None):
            return str(request.prompt)
        if hasattr(request, "messages"):
            msgs = request.messages
            if msgs:
                return " ".join(str(m.content) for m in msgs)
        return ""

    def _redact_pii(self, text: str) -> str:
        if not text:
            return text
        for compiled, replacement in self._redaction_patterns:
            with contextlib.suppress(Exception):
                text = compiled.sub(replacement, text)
        return text
