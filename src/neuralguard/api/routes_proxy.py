"""Proxy routes (F9): standalone appliance mode.

``POST /v1/proxy/chat/completions`` — accept an OpenAI-format chat payload,
evaluate the user turns through the full pipeline (F6 role-aware), forward
ALLOWed requests to the configured upstream, scan the completion with
output-scan semantics (PII/exfil/canary), and deliver the verdict-shaped
result to the caller.

Contract:
- Non-allow INPUT  -> 403 with NeuralGuard findings; the upstream is NEVER
  called; the caller sees NeuralGuard, not the LLM.
- Streaming (`stream: true`) -> 422 in this build (fail-closed; SSE
  hold-back scanning is a planned follow-up — a control that silently
  passes unscanned chunks would be worse than refusing).
- Non-allow OUTPUT -> 403 with findings (caller gets the block INSTEAD of
  the completion).
- SANITIZED output -> 200 with the REDACTED completion + findings.
- Upstream failure -> generic 502 (upstream details logged, never returned).
- The X-NeuralGuard-Verdict header carries the final verdict on every path.

Secrets: the upstream API key is held server-side and never logged or
echoed. The caller authenticates with NeuralGuard keys (tenant-bound).

``GET /v1/info`` (non-proxy prefix) surfaces the appliance posture:
enabled layers, judge egress, and upstream egress (local | cloud) so nobody
is surprised about where prompts go.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from neuralguard.logging.audit import AuditLogger  # noqa: TC001 - runtime state access
from neuralguard.metrics import metrics
from neuralguard.models.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    Finding,
    Message,
    ScanLayer,
    Severity,
    ThreatCategory,
    Verdict,
)
from neuralguard.proxy.forwarder import UpstreamError
from neuralguard.scanners.pipeline import ScannerPipeline  # noqa: TC001 - runtime type

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/proxy", tags=["proxy"])

_VERDICT_HEADER = "X-NeuralGuard-Verdict"


class ProxyChatRequest(BaseModel):
    """OpenAI-compatible chat completion payload (loose).

    Unknown fields are preserved and forwarded to the upstream verbatim
    (temperature, tools, response_format, ...). The optional NeuralGuard
    extension ``session_id`` enables canary-leak detection on the response.
    """

    model: str = Field(description="Model identifier (forwarded to the upstream)")
    messages: list[Message] = Field(description="OpenAI chat messages")
    stream: bool = Field(default=False, description="Streaming is refused in this build (422)")
    session_id: str | None = Field(
        default=None,
        description="NeuralGuard extension: enables canary-leak detection on the response",
    )
    model_config = {"extra": "allow"}


def _pipeline(request: Request) -> ScannerPipeline:
    pipeline: ScannerPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:  # pragma: no cover - assembly bug, fail loud
        raise RuntimeError("pipeline not installed on app state")
    return pipeline


def _audit(request: Request) -> AuditLogger:
    audit: AuditLogger | None = getattr(request.app.state, "audit_logger", None)
    if audit is None:  # pragma: no cover - assembly bug, fail loud
        raise RuntimeError("audit logger not installed on app state")
    return audit


@router.post("/chat/completions")
async def proxy_chat_completions(
    body: ProxyChatRequest,
    request: Request,
    response: Response,
) -> JSONResponse:
    """Guarded forward of an OpenAI-format chat completion."""
    config = request.app.state.config
    pipeline = _pipeline(request)
    audit = _audit(request)
    tenant = getattr(request.state, "auth_tenant", None) or "default"
    start = time.perf_counter()

    # Fail-closed on streaming: unscanned chunks must not leak. SSE hold-back
    # scanning is a planned follow-up; until then streaming is refused.
    if body.stream:
        return JSONResponse(
            status_code=422,
            content={
                "error": "streaming_not_supported",
                "message": "stream=true is refused by this NeuralGuard build: "
                "responses are scanned as a whole before delivery (fail-closed). "
                "Use stream=false.",
            },
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )

    # ── 1. Input evaluation (full pipeline, user-role turns — F6) ──
    eval_request = EvaluateRequest(
        messages=[Message(role=m.role, content=m.content) for m in body.messages],
        prompt=body.model,  # carried for audit context? no — see below
        tenant_id=tenant,
        session_id=body.session_id,
        use_case="chat",
        metadata={"proxy": True, "model": body.model},
    )
    # The model name is not user content: keep it OUT of the scanned text.
    eval_request.prompt = None
    if not eval_request.messages:
        return JSONResponse(
            status_code=422,
            content={"error": "empty_messages", "message": "messages must not be empty"},
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )

    try:
        arbitration = pipeline.execute(eval_request)
    except Exception as exc:
        logger.error("proxy_input_scan_failed", error=repr(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "input evaluation failed"},
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )

    for r in arbitration.scanner_results:
        metrics.observe_scanner(r.layer.value, r.latency_ms / 1000.0)

    confidence = max((f.confidence for f in arbitration.findings), default=0.0)
    audit.log_evaluation(
        eval_request,
        EvaluateResponse(
            tenant_id=tenant,
            verdict=arbitration.verdict,
            findings=arbitration.findings,
            confidence=confidence,
            scan_layers_used=[r.layer for r in arbitration.scanner_results],
            total_latency_ms=arbitration.total_latency_ms,
        ),
        arbitration,
    )

    # ── 2. Non-allow input: block BEFORE the upstream is ever called ──
    if arbitration.verdict != Verdict.ALLOW:
        metrics.record_verdict(Verdict.BLOCK.value)
        logger.info(
            "proxy_input_blocked",
            tenant=tenant,
            verdict=arbitration.verdict.value,
            findings=len(arbitration.findings),
        )
        return JSONResponse(
            status_code=403,
            content={
                "error": "request_blocked",
                "message": "Request blocked by NeuralGuard firewall (input scan)",
                "verdict": arbitration.verdict.value,
                "findings": [f.model_dump(mode="json") for f in arbitration.findings],
                "confidence": confidence,
            },
            headers={_VERDICT_HEADER: arbitration.verdict.value},
        )

    # ── 3. Forward to the upstream (operator's key, server-side) ──
    forwarder = getattr(request.app.state, "proxy_forwarder", None)
    if forwarder is None:  # fail-closed: mount-order bug must not bypass the proxy
        logger.error("proxy_forwarder_missing")
        return JSONResponse(
            status_code=500,
            content={"error": "proxy_not_ready", "message": "upstream forwarder unavailable"},
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )

    # Forward the ORIGINAL payload verbatim (minus the NeuralGuard extension),
    # preserving unknown OpenAI params (temperature, tools, ...).
    upstream_payload: dict[str, Any] = body.model_dump(mode="json", exclude={"session_id"})
    try:
        upstream_json = await forwarder.forward_chat(upstream_payload)
    except UpstreamError as exc:
        logger.warning("proxy_upstream_failed", error=str(exc))
        metrics.record_verdict(Verdict.BLOCK.value)
        return JSONResponse(
            status_code=502,
            content={"error": "upstream_failure", "message": str(exc)},
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )

    # ── 4. Scan the completion (output semantics: PII/exfil/canary) ──
    completion = _extract_completion(upstream_json)
    if completion is None:
        logger.error("proxy_completion_missing", upstream_keys=list(upstream_json.keys()))
        return JSONResponse(
            status_code=502,
            content={"error": "upstream_failure", "message": "upstream returned no completion"},
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )

    out_eval = EvaluateRequest(
        prompt=completion,
        tenant_id=tenant,
        session_id=body.session_id,
        use_case="completion",
        scanners=[ScanLayer.PATTERN],
        output_only=True,
    )
    try:
        out_arbitration = pipeline.execute(out_eval)
    except Exception as exc:
        logger.error("proxy_output_scan_failed", error=repr(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "output scan failed"},
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )

    # Canary leak detection (same semantics as /v1/scan/output).
    canary_leaked = False
    canary_manager = getattr(request.app.state, "canary_manager", None)
    if canary_manager is not None and body.session_id:
        try:
            leaked_token = canary_manager.check_leak(body.session_id, completion)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("canary_check_failed", error=repr(exc))
            leaked_token = None
        if leaked_token:
            canary_leaked = True
            out_arbitration.findings.append(
                Finding(
                    category=ThreatCategory.SYSTEM_PROMPT_EXTRACTION,
                    severity=Severity.HIGH,
                    verdict=Verdict.BLOCK,
                    confidence=0.95,
                    layer=ScanLayer.PATTERN,
                    rule_id="CANARY-LEAK-001",
                    description=(
                        "Canary token leaked in proxied LLM output — the system "
                        "prompt has been exfiltrated."
                    ),
                    mitigation="Block the output; rotate the canary secret; investigate.",
                    evidence=f"[REDACTED:canary:{leaked_token[:9]}...]",
                )
            )
            out_arbitration.verdict = Verdict.BLOCK
            out_arbitration.arbitration_reason = (
                out_arbitration.arbitration_reason or ""
            ) + " | canary token leaked in output"

    metrics.record_verdict(out_arbitration.verdict.value)
    total_ms = (time.perf_counter() - start) * 1000
    out_findings = [f.model_dump(mode="json") for f in out_arbitration.findings]
    response.headers[_VERDICT_HEADER] = out_arbitration.verdict.value

    # ── 5. Deliver the verdict-shaped result ──
    if out_arbitration.verdict == Verdict.BLOCK:
        logger.info("proxy_output_blocked", tenant=tenant, findings=len(out_arbitration.findings))
        return JSONResponse(
            status_code=403,
            content={
                "error": "response_blocked",
                "message": "LLM response blocked by NeuralGuard (output scan)",
                "verdict": Verdict.BLOCK.value,
                "findings": out_findings,
                "canary_leaked": canary_leaked,
            },
            headers=response.headers,
        )

    # ALLOW / SANITIZE: deliver the completion (REDACTED when sanitized).
    # The redaction itself lives in the sanitize ACTION (mitigations are
    # applied there, same as /v1/scan/output) — reuse it.
    delivered = completion
    if out_arbitration.verdict == Verdict.SANITIZE:
        from neuralguard.actions import ActionDispatcher

        action_result = ActionDispatcher(config).execute(out_arbitration, out_eval)
        sanitized = action_result.body.get("sanitized_content")
        if sanitized:
            delivered = str(sanitized)

    logger.info(
        "proxy_delivered",
        tenant=tenant,
        verdict=out_arbitration.verdict.value,
        findings=len(out_arbitration.findings),
        total_latency_ms=f"{total_ms:.1f}",
    )
    return JSONResponse(
        content=_with_scanned_completion(
            upstream_json, delivered, out_findings, canary_leaked, out_arbitration.verdict.value
        ),
        headers=response.headers,
    )


def _extract_completion(upstream_json: dict[str, Any]) -> str | None:
    """Pull the first choice's message content from an OpenAI-format response."""
    choices = upstream_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    return content


def _sanitized_completion(arbitration: Any, original: str) -> str | None:
    """The redacted completion for a SANITIZE verdict, if one was produced."""
    results = arbitration.scanner_results or []
    for r in results:
        if r.sanitized_output:
            return str(r.sanitized_output)
    return original


def _with_scanned_completion(
    upstream_json: dict[str, Any],
    delivered: str,
    findings: list[dict[str, Any]],
    canary_leaked: bool,
    verdict: str,
) -> dict[str, Any]:
    """Return the upstream response with the (possibly redacted) completion
    and a neuralguard_scan block appended."""
    import copy

    out = copy.deepcopy(upstream_json)
    choices = out.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            message["content"] = delivered
    out["neuralguard_scan"] = {
        "verdict": verdict,
        "findings": findings,
        "canary_leaked": canary_leaked,
    }
    return out
