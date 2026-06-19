"""NeuralGuard API routes — FastAPI endpoints.

Primary endpoints:
- POST /v1/evaluate — Scan input for threats
- POST /v1/scan/output — Validate LLM output
- GET /v1/health — Health check
- GET /v1/info — Service metadata
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from neuralguard.actions import ActionDispatcher
from neuralguard.metrics import metrics
from neuralguard.models.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    ScanLayer,
    ScanOutputRequest,
    ScanOutputResponse,
    Verdict,
)

# Response models for non-200 status codes
_BLOCK_RESPONSES: dict[int | str, dict[str, Any]] = {
    403: {"description": "Request blocked by firewall", "model": EvaluateResponse},
    429: {"description": "Rate limit exceeded"},
    422: {"description": "Validation error"},
}

if TYPE_CHECKING:
    from neuralguard.config.settings import NeuralGuardConfig
    from neuralguard.logging.audit import AuditLogger
    from neuralguard.scanners.pipeline import ScannerPipeline

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["NeuralGuard API"])


# ── Dependencies ──────────────────────────────────────────────────────────


def get_pipeline(request: Request) -> ScannerPipeline:
    return cast("ScannerPipeline", request.app.state.pipeline)


def get_config(request: Request) -> NeuralGuardConfig:
    return cast("NeuralGuardConfig", request.app.state.config)


def get_audit_logger(request: Request) -> AuditLogger:
    return cast("AuditLogger", request.app.state.audit_logger)


def _auth_tenant(request: Request) -> str | None:
    return getattr(request.state, "auth_tenant", None)


def _check_tenant_binding(request: Request, body_tenant: str) -> None:
    """Reject 403 if the authenticated key is bound to a different tenant."""
    config = request.app.state.config
    auth_tenant = _auth_tenant(request)
    if (
        config.auth.enabled
        and config.auth.enforce_tenant_from_key
        and auth_tenant is not None
        and body_tenant.lower() != auth_tenant.lower()
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "tenant_mismatch",
                "message": "API key is not authorized for the requested tenant_id.",
            },
        )


def _internal_error(exc: Exception, path: str) -> JSONResponse:
    """Log an unexpected error and return a sanitized 500 with a correlation id.

    Wrapped at the route level because Starlette `BaseHTTPMiddleware` re-raises
    route exceptions past `@app.exception_handler`, defeating the global
    handler. This keeps clients from receiving a raw traceback while giving
    operators a correlation id for log lookup.
    """
    import uuid as _uuid

    corr_id = str(_uuid.uuid4())
    logger.error(
        "internal_error",
        correlation_id=corr_id,
        path=path,
        error=repr(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An internal error occurred.",
            "correlation_id": corr_id,
        },
    )


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("/evaluate", response_model=EvaluateResponse, responses=_BLOCK_RESPONSES)
async def evaluate(
    body: EvaluateRequest,
    request: Request,
    pipeline: ScannerPipeline = Depends(get_pipeline),
    config: NeuralGuardConfig = Depends(get_config),
    audit: AuditLogger = Depends(get_audit_logger),
) -> EvaluateResponse | JSONResponse:
    """Scan input messages/prompts for security threats.

    Runs through all enabled scanner layers and returns a verdict
    with detailed findings.
    """
    # Enforce tenant binding against the authenticated API key.
    _check_tenant_binding(request, body.tenant_id)

    start = time.perf_counter()

    logger.info(
        "evaluate_request",
        tenant=body.tenant_id,
        use_case=body.use_case,
        messages=len(body.messages) if body.messages else 0,
        has_prompt=body.prompt is not None,
    )

    # Execute scanner pipeline (fail-closed on unexpected internal error)
    try:
        arbitration = pipeline.execute(body)
    except Exception as exc:
        return _internal_error(exc, "/v1/evaluate")

    # Record metrics: per-scanner latency and verdict
    for r in arbitration.scanner_results:
        metrics.observe_scanner(r.layer.value, r.latency_ms / 1000.0)
    metrics.record_verdict(arbitration.verdict.value)

    # Dispatch response action
    dispatcher = ActionDispatcher(config)
    try:
        action_result = dispatcher.execute(arbitration, body)
    except Exception as exc:
        return _internal_error(exc, "/v1/evaluate")

    # Compute confidence and layers used
    confidence = max((f.confidence for f in arbitration.findings), default=0.0)
    layers_used = [r.layer for r in arbitration.scanner_results]
    total_ms = (time.perf_counter() - start) * 1000
    metrics.observe_pipeline(total_ms / 1000.0)

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


@router.post("/scan/output", response_model=ScanOutputResponse, responses=_BLOCK_RESPONSES)
async def scan_output(
    body: ScanOutputRequest,
    request: Request,
    pipeline: ScannerPipeline = Depends(get_pipeline),
    config: NeuralGuardConfig = Depends(get_config),
    audit: AuditLogger = Depends(get_audit_logger),
) -> ScanOutputResponse | JSONResponse:
    """Validate LLM output before delivery.

    Checks for:
    - PII leakage (emails, phone numbers, SSNs, API keys)
    - Canary token leakage (if session_id provided)
    - System prompt leakage (if system_prompt_hash provided)
    - Schema compliance
    """
    # Enforce tenant binding against the authenticated API key.
    _check_tenant_binding(request, body.tenant_id)

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

    try:
        arbitration = pipeline.execute(eval_request)
    except Exception as exc:
        return _internal_error(exc, "/v1/scan/output")

    # Record metrics
    for r in arbitration.scanner_results:
        metrics.observe_scanner(r.layer.value, r.latency_ms / 1000.0)
    metrics.record_verdict(arbitration.verdict.value)

    # Dispatch response action (output scan uses action framework)
    dispatcher = ActionDispatcher(config)
    try:
        action_result = dispatcher.execute(arbitration, body)
    except Exception as exc:
        return _internal_error(exc, "/v1/scan/output")

    # Canary detection (Phase 3 — not yet implemented; tracked as P2-1)
    canary_leaked = False

    total_ms = (time.perf_counter() - start) * 1000
    metrics.observe_pipeline(total_ms / 1000.0)

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
    request: Request,
    config: NeuralGuardConfig = Depends(get_config),
    pipeline: ScannerPipeline = Depends(get_pipeline),
) -> HealthResponse:
    """Health check endpoint."""
    scanners = {layer.value: layer in pipeline._scanners for layer in ScanLayer}

    # Calculate actual uptime from app start time
    start_time = getattr(request.app.state, "start_time", None)
    uptime = time.time() - start_time if start_time else 0.0

    return HealthResponse(
        status="healthy",
        version=config.version,
        environment=config.environment,
        scanners=scanners,
        uptime_seconds=uptime,
    )


@router.get("/info")
async def info(
    request: Request,
    config: NeuralGuardConfig = Depends(get_config),
) -> dict[str, Any]:
    """Service metadata endpoint.

    Requires authentication (not in `public_endpoints`) so version/environment
    and scanner coverage are not disclosed to unauthenticated callers.
    """
    _ = request  # auth enforced by AuthMiddleware on /v1/* (info is not public)
    return {
        "name": config.app_name,
        "version": config.version,
        "environment": config.environment,
        "description": "LLM Guard / AI Application Firewall",
        "owasp_coverage": {
            "dedicated_rules": [
                "LLM01 (Prompt Injection)",
                "LLM02 (Sensitive Disclosure)",
                "LLM05 (Improper Output)",
                "LLM07 (System Prompt Leakage)",
                "LLM10 (Unbounded Consumption)",
                "ASI01 (Goal Hijack)",
                "ASI02 (Tool Misuse)",
                "ASI06 (Memory Poisoning)",
            ],
            # Corpus-assisted only: no dedicated detection rules. Tracked for
            # honesty so customers do not rely on coverage that is incidental.
            "corpus_assisted_only": ["ASI04 (Supply Chain)", "ASI10 (Rogue Agents)"],
        },
        "api_version": "v1",
    }


@router.get("/metrics")
async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint.

    Exposes counters/histograms for SOC observability. Auth-protected (not in
    `public_endpoints`) so external observers cannot scrape internal signal.
    """
    _ = request  # auth enforced by AuthMiddleware
    m = request.app.state.metrics
    payload, content_type = m.expose()
    return Response(content=payload, media_type=content_type)
