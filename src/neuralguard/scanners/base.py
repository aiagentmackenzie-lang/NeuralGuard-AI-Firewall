"""Scanner base class and interface contract.

Every scanner inherits from BaseScanner and implements the scan() method.
Scanners are independently toggled per-tenant and must be ReDoS-safe.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from neuralguard.models.schemas import (
    EvaluateRequest,
    Finding,
    ScanLayer,
    ScannerResult,
    Verdict,
)

if TYPE_CHECKING:
    from neuralguard.config.settings import ScannerSettings


class BaseScanner(ABC):
    """Abstract base class for all NeuralGuard scanners.

    Contract:
    - scan() must complete within its timeout budget
    - scan() must never raise — errors are captured in ScannerResult.error
    - scan() must return a valid ScannerResult even on failure
    - On failure with fail_closed=True, verdict defaults to BLOCK
    """

    layer: ScanLayer

    def __init__(self, settings: ScannerSettings) -> None:
        self.settings = settings

    @abstractmethod
    def scan(
        self, request: EvaluateRequest, context: dict[str, Any] | None = None
    ) -> ScannerResult:
        """Execute scan on the request.

        Args:
            request: The evaluation request.
            context: Optional context from previous scanners (e.g., structural flags).

        Returns:
            ScannerResult with verdict, findings, and latency.
        """
        ...

    def safe_scan(
        self, request: EvaluateRequest, context: dict[str, Any] | None = None
    ) -> ScannerResult:
        """Wrapper that guarantees a valid ScannerResult even on exceptions."""
        start = time.perf_counter()
        try:
            result = self.scan(request, context)
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Fail-closed: scanner errors → BLOCK
            default_verdict = Verdict.BLOCK
            return ScannerResult(
                layer=self.layer,
                verdict=default_verdict,
                findings=[],
                latency_ms=elapsed_ms,
                error=f"Scanner {self.layer.value} failed: {exc!r}",
            )

    def _result(
        self,
        verdict: Verdict,
        findings: list[Finding],
        start_time: float,
        sanitized: str | None = None,
        error: str | None = None,
    ) -> ScannerResult:
        """Helper to build a ScannerResult with timing."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return ScannerResult(
            layer=self.layer,
            verdict=verdict,
            findings=findings,
            latency_ms=elapsed_ms,
            error=error,
            sanitized_output=sanitized,
        )
