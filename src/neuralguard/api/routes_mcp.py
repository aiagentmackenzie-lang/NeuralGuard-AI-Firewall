"""MCP gateway routes (NG-7/NG-8): guarded JSON-RPC passthrough to one MCP server.

``POST /v1/mcp`` — mounted only when ``NEURALGUARD_MCP_ENABLED=true``.

Flow (in order):

1. **NG-8 Intent Gate (pre-body-parse)**: ``Mcp-Method`` / ``Mcp-Name``
   headers are evaluated against the tenant's policy (per-tool / per-method
   allow/deny/escalate) BEFORE the body is parsed — the cheapest rejection
   path and the seam the MCP 2026-07-28 spec created for gateways.
2. **Body parse + smuggling check**: the JSON-RPC body's declared intent
   must agree with the headers (a mismatch is smuggling — BLOCK).
3. **NG-7 baseliner**:
   - ``tools/list``  -> the fetched catalog is checked against the signed
     baseline; strict mode withholds a drifted catalog (403 + drift
     report), advisory passes it with an alert header + audit event.
   - ``tools/call``  -> the tool must be in the last-good baseline (a tool
     the catalog never contained is a rug-pull execute — refused in every
     mode).
   - anything else   -> passthrough (the Intent Gate verdict already applied).
4. **Audit**: every gate decision and baseline transition is a hash-chained
   audit event (P2-10 machinery) — SIEM routing sees MCP denials and drift
   in the same tamper-evident trail as scan verdicts.

Contract details:
- Streaming: a JSON-RPC body asking for SSE (``Accept: text/event-stream``
  ONLY, or protocol-level stream hints) is refused 422 in this build —
  fail-closed, same posture as the chat proxy.
- Upstream failure -> generic 502 (details logged, never returned).
- ``X-NeuralGuard-Verdict`` header on every path.
- Baselines are per-worker in-memory; restart re-baselines loudly
  (``BASELINE_RECREATED_RESTART``) and every historical hash pair lives in
  the audit chain.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from neuralguard.logging.audit import AuditLogger  # noqa: TC001 - runtime state access
from neuralguard.mcp.intent_gate import (
    GateDecision,
    McpToolPolicy,
    ToolAction,
    body_intent,
    check_header_body_agreement,
    evaluate_intent,
)
from neuralguard.mcp.transport import McpUpstreamError
from neuralguard.metrics import metrics
from neuralguard.models.schemas import Severity, ThreatCategory, Verdict
from neuralguard.scanners.pipeline import ScannerPipeline  # noqa: TC001 - runtime type

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/mcp", tags=["mcp"])

_VERDICT_HEADER = "X-NeuralGuard-Verdict"
_DRIFT_HEADER = "X-NeuralGuard-Mcp-Drift"
_BASELINE_HEADER = "X-NeuralGuard-Mcp-Baseline"
_CATALOG_SIG_HEADER = "X-MCP-Catalog-Signature"  # trusted-registry transport

# Method names whose RESPONSE carries the tool catalog (NG-7 surface).
_CATALOG_METHODS = {"tools/list"}


def _mcp_transport(request: Request) -> Any:
    transport = getattr(request.app.state, "mcp_transport", None)
    if transport is None:  # pragma: no cover - assembly bug, fail loud
        raise RuntimeError("MCP transport not installed on app state")
    return transport


def _baseliner(request: Request) -> Any:
    baseliner = getattr(request.app.state, "mcp_baseliner", None)
    if baseliner is None:  # pragma: no cover - assembly bug, fail loud
        raise RuntimeError("MCP baseliner not installed on app state")
    return baseliner


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


def _tenant_policy(request: Request, tenant: str) -> McpToolPolicy:
    """Resolve the tenant's MCP policy: tenant file overlay or default allow."""
    registry = getattr(request.app.state, "tenant_registry", None)
    if registry is not None and getattr(registry, "enabled", False):
        cfg = registry.get(tenant)
        if cfg is not None and cfg.mcp is not None:
            policy = cfg.mcp
            if isinstance(policy, McpToolPolicy):
                return policy
    return McpToolPolicy()


def _finding(decision: GateDecision, description: str) -> dict[str, Any]:
    return {
        "category": ThreatCategory.TOOL_MISUSE.value,
        "severity": Severity.HIGH.value,
        "verdict": decision.verdict.value,
        "confidence": 1.0,
        "layer": "pattern",
        "rule_id": decision.rule_id,
        "description": description,
    }


def _gate_response(
    decision: GateDecision,
    status: int,
    request_id: str,
    latency_ms: float,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "error": "mcp_gate_blocked" if decision.verdict == Verdict.BLOCK else "mcp_gate_escalated",
        "rule_id": decision.rule_id,
        "reason": decision.reason,
        "declared_method": decision.declared_method,
        "declared_tool": decision.declared_tool,
        "findings": [_finding(decision, decision.reason)],
        "request_id": request_id,
    }
    if extra:
        payload.update(extra)
    resp = JSONResponse(status_code=status, content=payload)
    resp.headers[_VERDICT_HEADER] = decision.verdict.value
    return resp


@router.post("")
async def mcp_gateway(request: Request) -> JSONResponse:
    """Guarded JSON-RPC passthrough to the configured MCP server."""
    config = request.app.state.config
    audit = _audit(request)
    transport = _mcp_transport(request)
    baseliner = _baseliner(request)
    tenant = getattr(request.state, "auth_tenant", None) or "default"
    request_id = getattr(request.state, "request_id", None) or f"mcp-{time.time_ns()}"
    start = time.perf_counter()

    def _elapsed() -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    def _audit_event(
        *, verdict: Verdict, decision: GateDecision | None, details: dict[str, Any]
    ) -> None:
        audit.log_mcp_event(
            request_id=request_id,
            tenant_id=tenant,
            verdict=verdict,
            rule_id=decision.rule_id if decision else "MCP-ROUTE",
            method=decision.declared_method if decision else None,
            tool=decision.declared_tool if decision else None,
            details=details,
            total_latency_ms=_elapsed(),
        )

    # ── NG-8: Intent Gate on headers, BEFORE body parse ──
    lower_headers = {k.lower(): v for k, v in request.headers.items()}
    policy = _tenant_policy(request, tenant)
    gate = evaluate_intent(lower_headers, policy, headers_required=config.mcp.headers_required)
    metrics.record_mcp_gate(gate.action.value)

    if gate.action == ToolAction.DENY:
        _audit_event(
            verdict=Verdict.BLOCK,
            decision=gate,
            details={"outcome": "gate_deny"},
        )
        return _gate_response(gate, 403, request_id, _elapsed())
    if gate.action == ToolAction.ESCALATE:
        # HITL posture: the intent is policy-gated for human approval. This
        # build refuses (fail-closed) and audits the escalation — an approval
        # workflow is the operator's follow-up, never a silent pass.
        _audit_event(
            verdict=Verdict.ESCALATE,
            decision=gate,
            details={"outcome": "gate_escalate"},
        )
        return _gate_response(gate, 403, request_id, _elapsed())

    # ── Body parse ──
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_json", "request_id": request_id},
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )
    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_jsonrpc", "request_id": request_id},
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )

    # ── Smuggling check: declared intent (headers) vs actual intent (body) ──
    body_method, body_tool = body_intent(payload)
    mismatch = check_header_body_agreement(
        gate.declared_method, gate.declared_tool, body_method, body_tool
    )
    if mismatch is not None:
        metrics.record_mcp_gate("deny")
        _audit_event(verdict=Verdict.BLOCK, decision=mismatch, details={"outcome": "smuggling"})
        return _gate_response(mismatch, 403, request_id, _elapsed())

    method = body_method or ""
    catalog_sig = lower_headers.get(_CATALOG_SIG_HEADER.lower())

    # ── Forward ──
    try:
        upstream = await transport.forward(payload)
    except McpUpstreamError as exc:
        logger.warning("mcp_forward_failed", error=str(exc), method=method)
        return JSONResponse(
            status_code=502,
            content={"error": "upstream_unavailable", "request_id": request_id},
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )

    # ── NG-7: catalog-bearing methods go through the baseliner ──
    if method in _CATALOG_METHODS:
        result = upstream.get("result")
        tools_raw = result.get("tools") if isinstance(result, dict) else None
        tools = tools_raw if isinstance(tools_raw, list) else []
        state = baseliner.check(tools, change_signature_hex=catalog_sig)
        metrics.record_mcp_baseline(state.outcome)
        _audit_event(
            verdict=Verdict.ALLOW if state.decision == "allow" else Verdict.BLOCK,
            decision=gate,
            details={
                "outcome": state.outcome,
                "old_catalog_hash": state.old_catalog_hash,
                "new_catalog_hash": state.new_catalog_hash,
                "drift": state.report,
                "baseline_reason": state.reason,
            },
        )
        if state.decision == "block":
            return JSONResponse(
                status_code=403,
                content={
                    "error": "mcp_tool_catalog_drift",
                    "rule_id": "MCP-RUGPULL-001",
                    "reason": state.reason,
                    "drift": state.report,
                    "request_id": request_id,
                },
                headers={
                    _VERDICT_HEADER: Verdict.BLOCK.value,
                    _DRIFT_HEADER: "1",
                },
            )
        resp = JSONResponse(
            status_code=200,
            content=upstream,
            headers={
                _VERDICT_HEADER: Verdict.ALLOW.value,
                _BASELINE_HEADER: state.new_catalog_hash or "",
            },
        )
        if state.outcome == "DRIFT_PASSED_ADVISORY":
            resp.headers[_DRIFT_HEADER] = "advisory"
        return resp

    if method == "tools/call":
        state = baseliner.evaluate_call(body_tool or "")
        metrics.record_mcp_baseline(state.outcome)
        _audit_event(
            verdict=Verdict.ALLOW if state.decision == "allow" else Verdict.BLOCK,
            decision=gate,
            details={
                "outcome": state.outcome,
                "baseline_reason": state.reason,
                "old_catalog_hash": state.old_catalog_hash,
            },
        )
        if state.decision == "block":
            return JSONResponse(
                status_code=403,
                content={
                    "error": "mcp_tool_not_baselined",
                    "rule_id": "MCP-RUGPULL-002",
                    "reason": state.reason,
                    "request_id": request_id,
                },
                headers={_VERDICT_HEADER: Verdict.BLOCK.value},
            )

    # Passthrough (gate-allowed, non-catalog method or baselined tool call).
    _audit_event(
        verdict=Verdict.ALLOW,
        decision=gate,
        details={"outcome": "forwarded", "body_method": method or None},
    )
    resp = JSONResponse(status_code=200, content=upstream)
    resp.headers[_VERDICT_HEADER] = Verdict.ALLOW.value
    return resp
