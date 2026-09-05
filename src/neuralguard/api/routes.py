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
from neuralguard.api.readiness import check_readiness
from neuralguard.metrics import metrics
from neuralguard.models.schemas import (
    AnalyzeTemplateRequest,
    AnalyzeTemplateResponse,
    CanaryMintRequest,
    CanaryMintResponse,
    EvaluateRequest,
    EvaluateResponse,
    Finding,
    HealthResponse,
    ScanLayer,
    ScanOutputRequest,
    ScanOutputResponse,
    Severity,
    TemplateSinkFinding,
    TenantInfoResponse,
    TenantListResponse,
    TenantScannerOverridesView,
    ThreatCategory,
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


def get_canary_manager(request: Request) -> Any:
    """Return the CanaryManager installed on app state, or None if disabled.

    The manager is constructed in ``create_app`` only when canary is enabled,
    so its absence means the feature is off. ``Any`` keeps the import out of
    the request hot path (the canary module is loaded lazily by main.py).
    """
    return getattr(request.app.state, "canary_manager", None)


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
    response: Response,
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

    # Normal 200 response (ALLOW / SANITIZE). The verdict header is set here
    # too (F15): headers previously survived only on non-200 responses.
    logger.info(
        "evaluate_response",
        request_id=audit_response.request_id,
        verdict=audit_response.verdict.value,
        findings=len(audit_response.findings),
        confidence=f"{confidence:.2f}",
        latency_ms=f"{total_ms:.2f}",
        reason=arbitration.arbitration_reason,
    )
    response.headers["X-NeuralGuard-Verdict"] = arbitration.verdict.value

    return audit_response


@router.post("/scan/output", response_model=ScanOutputResponse, responses=_BLOCK_RESPONSES)
async def scan_output(
    body: ScanOutputRequest,
    request: Request,
    response: Response,
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

    # Canary detection (B3) — per-session system-prompt exfiltration signal.
    # The manager is installed on app state only when canary is enabled; its
    # absence means the feature is off. ``check_leak`` is safe-by-default
    # (returns None on misconfiguration) so it never breaks the output scan.
    # Runs BEFORE the action dispatcher so a leaked canary (verdict=BLOCK)
    # drives the dispatched response (403), not just the audit body.
    canary_leaked = False
    canary_manager = getattr(request.app.state, "canary_manager", None)
    leaked_token: str | None = None
    if canary_manager is not None and body.session_id:
        try:
            leaked_token = canary_manager.check_leak(body.session_id, body.output)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("canary_check_failed", error=repr(exc))
            leaked_token = None
    if leaked_token:
        canary_leaked = True
        # A leaked canary is a hard system-prompt-exfiltration signal: force
        # BLOCK and attach a finding so the audit trail + response carry it.
        arbitration.findings.append(
            Finding(
                category=ThreatCategory.SYSTEM_PROMPT_EXTRACTION,
                severity=Severity.HIGH,
                verdict=Verdict.BLOCK,
                confidence=0.95,
                layer=ScanLayer.PATTERN,
                rule_id="CANARY-LEAK-001",
                description=(
                    "Canary token leaked in LLM output — the system prompt "
                    "has been exfiltrated (the model repeated a canary that was "
                    "only present in the confidential system prompt)."
                ),
                mitigation=(
                    "Block the output; rotate the canary secret; investigate "
                    "the exfiltration path (likely a prompt-injection / extraction "
                    "attack on this session)."
                ),
                evidence=f"[REDACTED:canary:{leaked_token[:9]}...]",
            )
        )
        arbitration.verdict = Verdict.BLOCK
        arbitration.arbitration_reason = (
            arbitration.arbitration_reason or ""
        ) + " | canary token leaked in output"
        metrics.record_verdict(Verdict.BLOCK.value)

    # Dispatch response action (output scan uses action framework)
    dispatcher = ActionDispatcher(config)
    try:
        action_result = dispatcher.execute(arbitration, body)
    except Exception as exc:
        return _internal_error(exc, "/v1/scan/output")

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
        # Return the full ScanOutputResponse shape (carries canary_leaked,
        # redacted_output, findings) at the action's status code, so the
        # operator gets the exfiltration / redaction signal even on a block.
        # The generic action body is dropped in favour of the audit response.
        return JSONResponse(
            status_code=action_result.status_code,
            content=audit_response.model_dump(mode="json"),
            headers=action_result.headers,
        )

    # F15: the verdict header is set on 200s too — previously it survived
    # only on non-200 responses.
    response.headers["X-NeuralGuard-Verdict"] = arbitration.verdict.value

    return audit_response


@router.post("/analyze/template", response_model=AnalyzeTemplateResponse)
async def analyze_template(
    body: AnalyzeTemplateRequest,
    request: Request,
) -> AnalyzeTemplateResponse | JSONResponse:
    """Statically analyze a system-prompt template for injection sinks (B2).

    Pure static analysis — no LLM call, no scanner pipeline, no state. Returns
    a list of sinks (untrusted-variable interpolation, missing delimiter
    fences, ambiguous instruction precedence, action-adjacent variables, raw
    structured-data injection) with severity + remediation. Use this to shift
    injection-sink detection left, before a template is deployed.
    """
    # Enforce tenant binding against the authenticated API key.
    _check_tenant_binding(request, body.tenant_id)

    start = time.perf_counter()
    logger.info(
        "analyze_template_request",
        tenant=body.tenant_id,
        template_len=len(body.template),
    )

    try:
        from neuralguard.analysis import TemplateAnalyzer

        analyzer = TemplateAnalyzer()
        result = analyzer.analyze(body.template)
    except Exception as exc:
        return _internal_error(exc, "/v1/analyze/template")

    total_ms = (time.perf_counter() - start) * 1000
    sinks = [
        TemplateSinkFinding(
            rule_id=s.rule_id,
            severity=s.severity,  # type: ignore[arg-type]
            description=s.description,
            remediation=s.remediation,
            evidence=s.evidence,
            location=s.location,
        )
        for s in result.sinks
    ]
    response = AnalyzeTemplateResponse(
        tenant_id=body.tenant_id,
        is_clean=result.is_clean,
        sink_count=len(sinks),
        sinks=sinks,
        total_latency_ms=total_ms,
    )
    logger.info(
        "analyze_template_response",
        tenant=body.tenant_id,
        is_clean=result.is_clean,
        sink_count=len(sinks),
        latency_ms=f"{total_ms:.2f}",
    )
    return response


@router.post("/canary/mint", response_model=CanaryMintResponse)
async def mint_canary(
    body: CanaryMintRequest,
    request: Request,
) -> CanaryMintResponse | JSONResponse:
    """Mint per-session canary token(s) for system-prompt exfiltration detection (B3).

    Inject the returned token(s) into the LLM system prompt before serving
    the turn. If a token later appears in the model output, the
    ``/v1/scan/output`` endpoint flags it as a system-prompt exfiltration
    signal (verdict=block, ``canary_leaked=true``).

    Derivation is deterministic (HMAC-SHA256 of ``session_id|label`` keyed by
    the server secret), so you do not need to store the token between this
    call and the output scan — only the ``session_id`` is the join key.

    Returns 503 when the canary feature is disabled or misconfigured, so a
    client cannot silently fall back to a no-canary flow in production.
    """
    _check_tenant_binding(request, body.tenant_id)

    start = time.perf_counter()
    manager = get_canary_manager(request)
    if manager is None or not manager.enabled:
        return JSONResponse(
            status_code=503,
            content={
                "error": "canary_disabled",
                "message": "Canary feature is disabled. Set NEURALGUARD_CANARY_ENABLED=true and a secret.",
            },
        )
    try:
        tokens = manager.mint(body.session_id, body.count)
    except Exception as exc:  # misconfigured secret / validation
        logger.error("canary_mint_failed", error=repr(exc))
        return JSONResponse(
            status_code=503,
            content={
                "error": "canary_misconfigured",
                "message": "Canary feature is enabled but misconfigured. Set a valid NEURALGUARD_CANARY_SECRET.",
            },
        )

    total_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "canary_mint_response",
        tenant=body.tenant_id,
        session=body.session_id,
        token_count=len(tokens),
        latency_ms=f"{total_ms:.2f}",
    )
    return CanaryMintResponse(
        tenant_id=body.tenant_id,
        session_id=body.session_id,
        tokens=tokens,
        total_latency_ms=total_ms,
    )


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


@router.get("/ready")
async def ready(
    request: Request,
    config: NeuralGuardConfig = Depends(get_config),
) -> JSONResponse:
    """Readiness probe — can this worker safely accept traffic?

    Returns 200 with ``status=healthy|degraded`` when the core (structural +
    pattern scanners, required Redis) is functional, and 503
    ``unhealthy`` when it is not. Optional components (semantic, judge,
    postgres audit) degrading yields 200 ``degraded`` — the firewall still
    serves with deterministic detection and JSONL audit fallback.

    Auth-protected by default; add ``/v1/ready`` to
    ``NEURALGUARD_AUTH_PUBLIC_ENDPOINTS`` for an unauthenticated kubelet probe.
    """
    _ = config  # auth enforced by AuthMiddleware on /v1/*
    result = await check_readiness(request)
    code = 200 if result.ready else 503
    return JSONResponse(status_code=code, content=result.to_dict())


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
    from neuralguard.net.egress import is_private_endpoint

    # F9/F10.3 posture: where does data go? Surfaced so nobody is surprised.
    proxy_info: dict[str, Any] | None = None
    if config.proxy.enabled:
        proxy_info = {
            "enabled": True,
            "upstream_egress": (
                "local" if is_private_endpoint(config.proxy.upstream_url) else "cloud"
            ),
        }
    judge_egress: str | None = None
    if config.scanner.judge_enabled:
        judge_egress = (
            "local"
            if is_private_endpoint(config.scanner.judge_ollama_url)
            else ("explicit-egress" if config.scanner.judge_allow_egress else "blocked-in-prod")
        )
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
        "judge_egress": judge_egress,
        "proxy": proxy_info,
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


# ── Per-tenant config (Sprint C, C1) ───────────────────────────────────────
# Read-only effective-config surface. Auth-gated (not in public_endpoints).
# Never exposes secrets — TenantConfig carries none. An unauthenticated or
# wrong-tenant caller is rejected by AuthMiddleware + the tenant-binding
# check below before any config is returned.


def _resolve_effective_scanners(
    config: NeuralGuardConfig,
    overlay: TenantScannerOverridesView | None,
) -> dict[str, bool]:
    """Resolve the per-tenant scanner enable state against the global config.

    Rule: a tenant may narrow (disable) an optional scanner but cannot widen
    past global registration. Effective = ``base AND (overlay ?? True)`` —
    ``overlay=None`` inherits (True), ``overlay=True`` keeps base (no widen),
    ``overlay=False`` forces False. Structural + Pattern are always True.
    """
    base = {
        "structural": True,
        "pattern": True,
        "agent_guardian": config.agent_guardian.enabled,
        "semantic": config.scanner.semantic_enabled,
        "judge": config.scanner.judge_enabled,
    }
    if overlay is None:
        return base
    for key, val in (
        ("agent_guardian", overlay.agent_guardian),
        ("semantic", overlay.semantic),
        ("judge", overlay.judge),
    ):
        if val is not None:
            base[key] = base[key] and val
    return base


@router.get("/tenants", response_model=TenantListResponse)
async def list_tenants(
    request: Request,
    config: NeuralGuardConfig = Depends(get_config),
) -> TenantListResponse | JSONResponse:
    """List all configured tenants with their effective config (read-only).

    Auth-protected. Returns 404 if multi-tenant mode is disabled.
    """
    registry = getattr(request.app.state, "tenant_registry", None)
    if registry is None or not registry.enabled:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "tenants_disabled",
                "message": "Multi-tenant mode is disabled. Set NEURALGUARD_TENANT_ENABLED=true.",
            },
        )
    try:
        items: list[TenantInfoResponse] = []
        for cfg in registry.list_tenants():
            items.append(_tenant_info_from_config(config, cfg))
        return TenantListResponse(tenants=items, count=len(items))
    except Exception as exc:
        return _internal_error(exc, "/v1/tenants")


@router.get("/tenants/{tenant_id}", response_model=TenantInfoResponse)
async def get_tenant(
    tenant_id: str,
    request: Request,
    config: NeuralGuardConfig = Depends(get_config),
) -> TenantInfoResponse | JSONResponse:
    """Return the effective config for one tenant (read-only, no secrets).

    Auth-protected. A tenant with no override file returns ``configured=false``
    with the global defaults resolved as the effective values — this is NOT a
    404 (a config miss is fail-open, never a denial). The authenticated caller
    must be bound to the same tenant (or be the default tenant) to read another
    tenant's effective config.
    """
    registry = getattr(request.app.state, "tenant_registry", None)
    if registry is None or not registry.enabled:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "tenants_disabled",
                "message": "Multi-tenant mode is disabled. Set NEURALGUARD_TENANT_ENABLED=true.",
            },
        )
    # Tenant-binding enforcement: an authenticated key bound to a tenant may
    # only read its own effective config (the default tenant may read any).
    auth_tenant = _auth_tenant(request)
    if (
        config.auth.enabled
        and config.auth.enforce_tenant_from_key
        and auth_tenant is not None
        and auth_tenant.lower() != "default"
        and auth_tenant.lower() != tenant_id.lower()
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "tenant_mismatch",
                "message": "API key is not authorized for the requested tenant_id.",
            },
        )
    try:
        cfg = registry.get(tenant_id)
        return _tenant_info_from_config(config, cfg, tenant_id=tenant_id)
    except Exception as exc:
        return _internal_error(exc, f"/v1/tenants/{tenant_id}")


def _tenant_info_from_config(
    config: NeuralGuardConfig,
    cfg: Any | None,
    *,
    tenant_id: str | None = None,
) -> TenantInfoResponse:
    """Resolve a TenantConfig (or a miss) into the public effective view."""
    from neuralguard.tenants.config import TenantConfig

    if cfg is None:
        tid = tenant_id or config.tenant.default_tenant
        overlay_view = TenantScannerOverridesView()
        rpm_eff = config.rate_limit.requests_per_minute
        burst_eff = config.rate_limit.burst_size
        return TenantInfoResponse(
            tenant_id=tid,
            description=None,
            configured=False,
            requests_per_minute=None,
            burst_size=None,
            effective_requests_per_minute=rpm_eff,
            effective_burst_size=burst_eff,
            scanners=overlay_view,
            effective_scanners=_resolve_effective_scanners(config, None),
        )
    assert isinstance(cfg, TenantConfig)
    overlay_view = TenantScannerOverridesView(
        agent_guardian=cfg.scanners.agent_guardian,
        semantic=cfg.scanners.semantic,
        judge=cfg.scanners.judge,
    )
    rpm_eff, burst_eff = cfg.effective_rate_limit(
        config.rate_limit.requests_per_minute,
        config.rate_limit.burst_size,
    )
    return TenantInfoResponse(
        tenant_id=cfg.tenant_id,
        description=cfg.description,
        configured=True,
        requests_per_minute=cfg.requests_per_minute,
        burst_size=cfg.burst_size,
        effective_requests_per_minute=rpm_eff,
        effective_burst_size=burst_eff,
        scanners=overlay_view,
        effective_scanners=_resolve_effective_scanners(config, overlay_view),
    )
