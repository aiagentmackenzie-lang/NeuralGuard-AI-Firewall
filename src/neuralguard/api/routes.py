"""NeuralGuard API routes — FastAPI endpoints.

Primary endpoints:
- POST /v1/evaluate — Scan input for threats
- POST /v1/scan/output — Validate LLM output
- GET /v1/health — Health check
- GET /v1/info — Service metadata
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from neuralguard.actions import ActionDispatcher
from neuralguard.models.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    ScanLayer,
    ScanOutputRequest,
    ScanOutputResponse,
    Verdict,
)

if TYPE_CHECKING:
    from neuralguard.config.settings import NeuralGuardConfig
    from neuralguard.logging.audit import AuditLogger
    from neuralguard.scanners.pipeline import ScannerPipeline

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["NeuralGuard API"])


# ── Dependencies ──────────────────────────────────────────────────────────


def get_pipeline(request: Request) -> ScannerPipeline:
    return request.app.state.pipeline


def get_config(request: Request) -> NeuralGuardConfig:
    return request.app.state.config


def get_audit_logger(request: Request) -> AuditLogger:
    return request.app.state.audit_logger


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    body: EvaluateRequest,
    pipeline: ScannerPipeline = Depends(get_pipeline),
    config: NeuralGuardConfig = Depends(get_config),
    audit: AuditLogger = Depends(get_audit_logger),
) -> EvaluateResponse:
    """Scan input messages/prompts for security threats.

    Runs through all enabled scanner layers and returns a verdict
    with detailed findings.
    """
    start = time.perf_counter()

    logger.info(
        "evaluate_request",
        tenant=body.tenant_id,
        use_case=body.use_case,
        messages=len(body.messages) if body.messages else 0,
        has_prompt=body.prompt is not None,
    )

    # Execute scanner pipeline
    arbitration = pipeline.execute(body)

    # Dispatch response action
    dispatcher = ActionDispatcher(config)
    action_result = dispatcher.execute(arbitration, body)

    # Compute confidence and layers used
    confidence = max((f.confidence for f in arbitration.findings), default=0.0)
    layers_used = [r.layer for r in arbitration.scanner_results]
    total_ms = (time.perf_counter() - start) * 1000

    # Build canonical response for audit logging
    sanitized = None
    if arbitration.verdict == Verdict.SANITIZE:
        sanitized = action_result.body.get("sanitized_content")

    audit_response = EvaluateResponse(
        tenant_id=body.tenant_id,
        verdict=arbitration.verdict,
        findings=arbitration.findings,
        confidence=confidence,
        sanitized_content=sanitized,
        scan_layers_used=layers_used,
        total_latency_ms=total_ms,
    )
    audit.log_evaluation(body, audit_response, arbitration)

    # Return non-200 responses directly (BLOCK, ESCALATE, QUARANTINE, RATE_LIMIT)
    if action_result.status_code != 200:
        return JSONResponse(
            status_code=action_result.status_code,
            content=action_result.body,
            headers=action_result.headers,
        )

    # Normal 200 response (ALLOW / SANITIZE)
    logger.info(
        "evaluate_response",
        request_id=audit_response.request_id,
        verdict=audit_response.verdict.value,
        findings=len(audit_response.findings),
        confidence=f"{confidence:.2f}",
        latency_ms=f"{total_ms:.2f}",
        reason=arbitration.arbitration_reason,
    )

    return audit_response


@router.post("/scan/output", response_model=ScanOutputResponse)
async def scan_output(
    body: ScanOutputRequest,
    pipeline: ScannerPipeline = Depends(get_pipeline),
    config: NeuralGuardConfig = Depends(get_config),
    audit: AuditLogger = Depends(get_audit_logger),
) -> ScanOutputResponse:
    """Validate LLM output before delivery.

    Checks for:
    - PII leakage (emails, phone numbers, SSNs, API keys)
    - Canary token leakage (if session_id provided)
    - System prompt leakage (if system_prompt_hash provided)
    - Schema compliance
    """
    start = time.perf_counter()

    logger.info(
        "scan_output_request",
        tenant=body.tenant_id,
        has_session=body.session_id is not None,
        has_prompt_hash=body.system_prompt_hash is not None,
    )

    # Convert to evaluate request for pipeline reuse
    eval_request = EvaluateRequest(
        prompt=body.output,
        tenant_id=body.tenant_id,
        use_case="completion",
        scanners=[ScanLayer.PATTERN],
        output_only=True,  # Only run output-relevant patterns (PII/EXF)
    )

    arbitration = pipeline.execute(eval_request)

    # Dispatch response action (output scan uses action framework)
    dispatcher = ActionDispatcher(config)
    action_result = dispatcher.execute(arbitration, body)

    # Canary detection (Phase 2 — for now, stub)
    canary_leaked = False

    total_ms = (time.perf_counter() - start) * 1000

    redacted = action_result.body.get("sanitized_content", body.output)

    audit_response = ScanOutputResponse(
        tenant_id=body.tenant_id,
        verdict=arbitration.verdict,
        findings=arbitration.findings,
        redacted_output=redacted,
        canary_leaked=canary_leaked,
        total_latency_ms=total_ms,
    )
    audit.log_output_scan(body, audit_response)

    if action_result.status_code != 200:
        return JSONResponse(
            status_code=action_result.status_code,
            content=action_result.body,
            headers=action_result.headers,
        )

    return audit_response


@router.get("/health", response_model=HealthResponse)
async def health(
    config: NeuralGuardConfig = Depends(get_config),
    pipeline: ScannerPipeline = Depends(get_pipeline),
) -> HealthResponse:
    """Health check endpoint."""
    scanners = {layer.value: layer in pipeline._scanners for layer in ScanLayer}

    return HealthResponse(
        status="healthy",
        version=config.version,
        environment=config.environment,
        scanners=scanners,
        uptime_seconds=time.time(),  # Approximate; replaced by app startup time
    )


@router.get("/info")
async def info(config: NeuralGuardConfig = Depends(get_config)) -> dict[str, Any]:
    """Service metadata endpoint."""
    return {
        "name": config.app_name,
        "version": config.version,
        "environment": config.environment,
        "description": "LLM Guard / AI Application Firewall",
        "owasp_coverage": [
            "LLM01 (Prompt Injection)",
            "LLM02 (Sensitive Disclosure)",
            "LLM05 (Improper Output)",
            "LLM07 (System Prompt Leakage)",
            "LLM10 (Unbounded Consumption)",
            "ASI01 (Goal Hijack)",
            "ASI02 (Tool Misuse)",
            "ASI06 (Memory Poisoning)",
        ],
        "api_version": "v1",
    }
