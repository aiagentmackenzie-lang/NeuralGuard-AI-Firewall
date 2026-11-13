"""MCP gateway routes (NG-7/NG-8): guarded JSON-RPC passthrough to one MCP server.

``POST /v1/mcp`` — mounted only when ``NEURALGUARD_MCP_ENABLED=true``.

Flow (in order):

1. **NG-8 Intent Gate (pre-body-parse)**: ``Mcp-Method`` / ``Mcp-Name``
   headers are evaluated against the tenant's policy (per-tool / per-method
   allow/deny/escalate) BEFORE the body is parsed — the cheapest rejection
   path and the seam the MCP 2026-07-28 spec created for gateways.
2. **Body parse + smuggling check**: the JSON-RPC body's declared intent
   must agree with the headers (a mismatch is smuggling — BLOCK).
3. **NG-10 argument constraints (pre-forward)**: for ``tools/call``, the
   tenant's per-tool constraints over the parsed arguments (path/URL
   allowlists, enums, lengths, deny patterns, required keys). A violation is
   refused before the upstream is contacted; findings carry the argument
   *path* and rule, never the value.
4. **NG-7 baseliner**:
   - ``tools/list``  -> the fetched catalog is checked against the signed
     baseline; strict mode withholds a drifted catalog (403 + drift
     report), advisory passes it with an alert header + audit event.
   - ``tools/call``  -> the tool must be in the last-good baseline (a tool
     the catalog never contained is a rug-pull execute — refused in every
     mode).
   - anything else   -> passthrough (the Intent Gate verdict already applied).
5. **NG-11 identity**: the workload agent comes from the authenticated
   credential (key binding ``key|tenant|agent`` or a JWT ``agent`` claim) and
   narrows, never widens, the tenant's policy. It is recorded on every audit
   event.
6. **Audit**: every gate decision and baseline transition is a hash-chained
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
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from neuralguard.logging.audit import AuditLogger  # noqa: TC001 - runtime state access
from neuralguard.mcp.argument_gate import evaluate_arguments
from neuralguard.mcp.credentials import CredentialError, ToolCredentialResolver
from neuralguard.mcp.hitl import (
    STATUS_APPROVED,
    STATUS_DENIED,
    STATUS_PENDING,
    HitlCapacityError,
    HitlError,
)
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

if TYPE_CHECKING:
    from neuralguard.mcp.hitl import HitlBroker
    from neuralguard.mcp.provenance import ProvenanceGate

router = APIRouter(prefix="/v1/mcp", tags=["mcp"])

_VERDICT_HEADER = "X-NeuralGuard-Verdict"
_DRIFT_HEADER = "X-NeuralGuard-Mcp-Drift"
_BASELINE_HEADER = "X-NeuralGuard-Mcp-Baseline"
_CATALOG_SIG_HEADER = "X-MCP-Catalog-Signature"  # trusted-registry transport
_PROVENANCE_HEADER = "X-NeuralGuard-Mcp-Provenance"  # NG-9 taint alerts
_SESSION_HEADER = "Mcp-Session-Id"  # MCP spec: session attribution

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


def _credentials(request: Request) -> ToolCredentialResolver:
    resolver: ToolCredentialResolver | None = getattr(request.app.state, "mcp_credentials", None)
    if resolver is None:  # pragma: no cover - assembly bug, fail loud
        raise RuntimeError("MCP credential resolver not installed on app state")
    return resolver


def _provenance_gate(request: Request) -> ProvenanceGate:
    gate: ProvenanceGate | None = getattr(request.app.state, "mcp_provenance", None)
    if gate is None:  # pragma: no cover - assembly bug, fail loud
        raise RuntimeError("MCP provenance gate not installed on app state")
    return gate


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
    # NG-11: workload identity, set by the auth middleware from the credential
    # (key binding or JWT claim). Never read from a header — a caller-supplied
    # agent id would make every per-agent rule decorative.
    agent: str | None = getattr(request.state, "auth_agent", None)
    request_id = getattr(request.state, "request_id", None) or f"mcp-{time.time_ns()}"
    start = time.perf_counter()

    def _elapsed() -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    def _audit_event(
        *,
        verdict: Verdict,
        decision: GateDecision | None,
        details: dict[str, Any],
        rule_id: str | None = None,
    ) -> None:
        audit.log_mcp_event(
            request_id=request_id,
            tenant_id=tenant,
            verdict=verdict,
            rule_id=rule_id or (decision.rule_id if decision else "MCP-ROUTE"),
            method=decision.declared_method if decision else None,
            tool=decision.declared_tool if decision else None,
            # NG-11: attribution is explicit on EVERY event, including the
            # tenant-level ones (agent_id: null) — "which agent did this" must
            # be answerable from the audit trail alone.
            details={**details, "agent_id": agent},
            total_latency_ms=_elapsed(),
        )

    # ── NG-8: Intent Gate on headers, BEFORE body parse ──
    lower_headers = {k.lower(): v for k, v in request.headers.items()}
    policy = _tenant_policy(request, tenant)
    gate = evaluate_intent(
        lower_headers, policy, headers_required=config.mcp.headers_required, agent=agent
    )
    metrics.record_mcp_gate(gate.action.value)

    if gate.action == ToolAction.DENY:
        _audit_event(
            verdict=Verdict.BLOCK,
            decision=gate,
            details={"outcome": "gate_deny"},
        )
        return _gate_response(gate, 403, request_id, _elapsed())

    # ── Body parse ──
    # Moved ahead of the ESCALATE handling for two reasons: NG-15 binds an
    # approval to the exact payload (a digest), and a smuggled request must be
    # BLOCKED before a human is asked to approve it. DENY still precedes the
    # parse, so the cheapest rejection path (the header gate) is unchanged.
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
    params = payload.get("params")
    call_arguments: Any = params.get("arguments") if isinstance(params, dict) else None

    # ── NG-15: human-in-the-loop for ESCALATE ──
    # ``None`` means "approved, continue"; otherwise the refusal response is
    # returned. With HITL disabled the old fail-closed refusal is unchanged.
    broker: HitlBroker | None = getattr(request.app.state, "mcp_hitl", None)

    async def _await_human(
        *,
        rule_id: str,
        reason: str,
        source: str,
        extra: dict[str, Any],
        disabled_outcome: str,
        disabled_response: JSONResponse,
    ) -> JSONResponse | None:
        if not (config.mcp.hitl_enabled and broker is not None):
            # The pre-NG-15 audit vocabulary is preserved verbatim
            # (``gate_escalate`` / ``argument_escalate``): renaming an outcome
            # silently breaks every SIEM rule and dashboard built on it.
            _audit_event(
                verdict=Verdict.ESCALATE,
                decision=gate,
                rule_id=rule_id,
                details={"outcome": disabled_outcome, **extra},
            )
            return disabled_response

        try:
            record = await broker.raise_escalation(
                tenant=tenant,
                agent=agent,
                method=method or gate.declared_method,
                tool=body_tool if method == "tools/call" else None,
                rule_id=rule_id,
                reason=reason,
                source=source,
                payload=payload,
                arguments=call_arguments,
            )
        except HitlCapacityError:
            # Capacity is a refusal, never a queue and never an implicit pass.
            metrics.record_mcp_gate("hitl_capacity")
            _audit_event(
                verdict=Verdict.BLOCK,
                decision=gate,
                rule_id="MCP-HITL-CAPACITY",
                details={"outcome": "escalation_capacity", "source": source, **extra},
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": "mcp_escalation_capacity",
                    "rule_id": "MCP-HITL-CAPACITY",
                    "reason": "too many pending escalations; refused rather than queued",
                    "request_id": request_id,
                },
                headers={_VERDICT_HEADER: Verdict.BLOCK.value},
            )

        # The escalation id is returned ONLY to the waiting request's audit
        # record (never in the caller's response): the caller cannot decide its
        # own escalation, and knowing the id buys nothing.
        _audit_event(
            verdict=Verdict.ESCALATE,
            decision=gate,
            rule_id="MCP-HITL-RAISED",
            details={
                "outcome": "escalation_raised",
                "source": source,
                "escalation_id": record.escalation_id,
                "request_digest": record.digest,
                **extra,
            },
        )
        decided = await broker.wait(record.escalation_id, config.mcp.hitl_timeout_seconds)
        approved = decided is not None and decided.status == STATUS_APPROVED

        if approved:
            try:
                await broker.consume(record.escalation_id)
            except HitlError:
                # Approval vanished between wake and use (expired). Fail closed.
                approved = False

        if not approved:
            denied = decided is not None and decided.status == STATUS_DENIED
            metrics.record_mcp_gate("hitl_denied" if denied else "hitl_timeout")
            _audit_event(
                verdict=Verdict.BLOCK,
                decision=gate,
                rule_id="MCP-HITL-DENIED" if denied else "MCP-HITL-TIMEOUT",
                details={
                    "outcome": "escalation_denied" if denied else "escalation_timeout",
                    "source": source,
                    "escalation_id": record.escalation_id,
                    "decided_by": decided.decided_by if decided is not None else None,
                    "decision_reason": decided.decision_reason if decided is not None else None,
                    "request_digest": record.digest,
                    **extra,
                },
            )
            code = "mcp_escalation_denied" if denied else "mcp_escalation_timeout"
            return JSONResponse(
                status_code=403,
                content={
                    "error": code,
                    "rule_id": "MCP-HITL-DENIED" if denied else "MCP-HITL-TIMEOUT",
                    "reason": reason if not decided else (decided.decision_reason or reason),
                    "escalation_id": record.escalation_id,
                    "request_id": request_id,
                },
                headers={_VERDICT_HEADER: Verdict.BLOCK.value},
            )

        metrics.record_mcp_gate("hitl_approved")
        _audit_event(
            verdict=Verdict.ALLOW,
            decision=gate,
            rule_id="MCP-HITL-APPROVED",
            details={
                "outcome": "escalation_approved",
                "source": source,
                "escalation_id": record.escalation_id,
                "decided_by": record.decided_by,
                "request_digest": record.digest,
                **extra,
            },
        )
        # Approval is not authority: every remaining gate below still runs.
        return None

    if gate.action == ToolAction.ESCALATE:
        refusal = await _await_human(
            rule_id=gate.rule_id,
            reason=gate.reason,
            source="intent_gate",
            extra={},
            disabled_outcome="gate_escalate",
            disabled_response=_gate_response(gate, 403, request_id, _elapsed()),
        )
        if refusal is not None:
            return refusal

    # ── NG-10: argument / resource constraints (parsed body, before forward) ──
    # The Intent Gate decided *whether* this tool may run; these decide *how*
    # it may be asked. Runs pre-network so a denied call never reaches the
    # upstream. Argument VALUES never enter the audit trail — only the path
    # and the rule that fired.
    if method == "tools/call":
        arg_decision = evaluate_arguments(policy, body_tool or "", call_arguments, agent=agent)
        if arg_decision.action is not ToolAction.ALLOW:
            escalated = arg_decision.action is ToolAction.ESCALATE
            metrics.record_mcp_gate("argument_escalate" if escalated else "argument_deny")
            arg_extra: dict[str, Any] = {
                "argument": arg_decision.argument,
                "constraint_kind": arg_decision.kind,
                "reason": arg_decision.reason,
            }
            if escalated:
                refusal = await _await_human(
                    rule_id=arg_decision.rule_id,
                    reason=arg_decision.reason,
                    source="argument_gate",
                    extra=arg_extra,
                    disabled_outcome="argument_escalate",
                    disabled_response=JSONResponse(
                        status_code=403,
                        content={
                            "error": "mcp_argument_escalated",
                            "rule_id": arg_decision.rule_id,
                            "reason": arg_decision.reason,
                            "argument": arg_decision.argument,
                            "tool": body_tool,
                            "request_id": request_id,
                        },
                        headers={_VERDICT_HEADER: Verdict.ESCALATE.value},
                    ),
                )
                if refusal is not None:
                    return refusal
            else:
                _audit_event(
                    verdict=Verdict.BLOCK,
                    decision=None,
                    rule_id=arg_decision.rule_id,
                    details={"outcome": "argument_deny", **arg_extra},
                )
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "mcp_argument_denied",
                        "rule_id": arg_decision.rule_id,
                        "reason": arg_decision.reason,
                        "argument": arg_decision.argument,
                        "tool": body_tool,
                        "request_id": request_id,
                    },
                    headers={_VERDICT_HEADER: Verdict.BLOCK.value},
                )

    catalog_sig = lower_headers.get(_CATALOG_SIG_HEADER.lower())
    # NG-9: session attribution (MCP spec header). None = session-less client;
    # the provenance gate then uses the shared tenant bucket (or refuses,
    # with require_session=true).
    session_id: str | None = lower_headers.get(_SESSION_HEADER.lower()) or None
    provenance_warned = False

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

        # ── NG-9: provenance-lite egress binding ──
        # Runs BEFORE the forward: tainted arguments never leave the trust
        # boundary in block mode. The tenant's egress_tools classification
        # scopes the check; the operator's mode decides warn vs block.
        prov = _provenance_gate(request)
        prov_state = prov.evaluate_egress(
            session_id=session_id,
            tool=body_tool or "",
            arguments=(payload.get("params") or {}).get("arguments"),
            egress_tools=set(policy.egress_tools) if policy else set(),
        )
        if prov_state.decision == "block":
            metrics.record_mcp_gate("provenance_block")
            _audit_event(
                verdict=Verdict.BLOCK,
                decision=gate,
                details={"outcome": "provenance_blocked", "provenance": prov_state.model_dump()},
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "mcp_provenance_taint",
                    "rule_id": prov_state.rule_id,
                    "reason": prov_state.reason,
                    "tool": body_tool,
                    "request_id": request_id,
                },
                headers={
                    _VERDICT_HEADER: Verdict.BLOCK.value,
                    _PROVENANCE_HEADER: "tainted",
                },
            )
        if prov_state.decision == "alert_allow":
            metrics.record_mcp_gate("provenance_warn")
            _audit_event(
                verdict=Verdict.ALLOW,
                decision=gate,
                details={"outcome": "provenance_warned", "provenance": prov_state.model_dump()},
            )
            provenance_warned = True
        elif prov_state.outcome == "clean":
            metrics.record_mcp_gate("provenance_clean")

    # ── NG-12/NG-14: select the upstream credential for THIS agent + tool ──
    # Resolved server-side from the operator's mapping; the more specific
    # (agent, tool) entry wins over the tool-wide one. A mapped-but-unresolvable
    # reference is refused (503) rather than falling back to the default token:
    # falling back would hand a scoped call the broad credential. The same holds
    # when a tool is agent-scoped and this caller has no entry — it is denied,
    # never handed the default.
    credential_source = "default" if config.mcp.upstream_auth_token else "none"
    # None = "let the transport use the configured default" (catalog methods
    # and any non-tool call). Only a resolved per-tool credential pins a value.
    upstream_token: str | None = None
    if method == "tools/call":
        resolver = _credentials(request)
        try:
            resolved = resolver.for_tool(body_tool or "", agent=agent)
        except CredentialError as exc:
            metrics.record_mcp_gate("credential_unavailable")
            _audit_event(
                verdict=Verdict.BLOCK,
                decision=gate,
                rule_id="MCP-CRED-001",
                details={
                    "outcome": "credential_unavailable",
                    "credential_source": resolver.source_for(body_tool or "", agent=agent),
                    # Message carries the reference/path and agent id, never a secret.
                    "reason": str(exc),
                },
            )
            logger.error(
                "mcp_credential_unavailable",
                tool=body_tool,
                agent=agent,
                source=resolver.source_for(body_tool or "", agent=agent),
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": "mcp_credential_unavailable",
                    "rule_id": "MCP-CRED-001",
                    "tool": body_tool,
                    "request_id": request_id,
                },
                headers={_VERDICT_HEADER: Verdict.BLOCK.value},
            )
        upstream_token = resolved.token
        credential_source = resolved.source

    # ── Forward ──
    try:
        upstream = await transport.forward(payload, upstream_token=upstream_token)
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

    # Passthrough (gate-allowed, non-catalog method or baselined tool call).
    _audit_event(
        verdict=Verdict.ALLOW,
        decision=gate,
        details={
            "outcome": "forwarded",
            "body_method": method or None,
            "credential_source": credential_source,
        },
    )
    resp = JSONResponse(status_code=200, content=upstream)

    # ── NG-9: taint the session with the tool result content ──
    # Everything that passes through the gateway's tool results becomes
    # provenance evidence for subsequent egress checks in the same session.
    if method == "tools/call":
        prov = _provenance_gate(request)
        fragments = prov.record_tool_result(session_id, upstream.get("result"))
        if fragments:
            logger.debug("mcp_provenance_tainted", fragments=fragments, tool=body_tool)

    if provenance_warned:
        resp.headers[_PROVENANCE_HEADER] = "tainted-warn"
    resp.headers[_VERDICT_HEADER] = Verdict.ALLOW.value
    return resp


# ── NG-15: the human side of the loop ─────────────────────────────────────
#
# These endpoints are the *operator* surface, not the caller surface. Three
# rules keep the loop from being gamed by the thing it gates:
#
# 1. a decision requires a TENANT-LEVEL credential (no authenticated agent) —
#    an agent raising an escalation for itself and approving it is the abuse
#    case this feature would otherwise create;
# 2. the deciding tenant must be named in ``hitl_approver_tenants`` (default:
#    the admin tenant) — no approver configured means no approval path;
# 3. a non-admin approver tenant is scoped to its OWN escalations. Without
#    that, naming a tenant as an approver would hand it a cross-tenant read of
#    other tenants' tool arguments (the preview) and the ability to approve
#    them. The admin tenant stays global: it is the gateway operator.


def _approver_scope(request: Request) -> tuple[str, bool] | JSONResponse:
    """Return ``(tenant, is_global)``, or a 403 response for non-approvers.

    ``is_global`` is True only for the admin tenant — the operator of the
    gateway, who may decide any tenant's escalation.
    """
    config = request.app.state.config
    request_id = getattr(request.state, "request_id", None) or "mcp-hitl"
    tenant = (getattr(request.state, "auth_tenant", None) or "default").lower()
    agent = getattr(request.state, "auth_agent", None)

    if agent is not None:
        return JSONResponse(
            status_code=403,
            content={
                "error": "mcp_escalation_agent_forbidden",
                "reason": "escalations are decided by a tenant-level credential; "
                "an agent identity cannot approve a tool call (self-approval)",
                "request_id": request_id,
            },
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )

    admin_tenant = config.auth.admin_tenant.lower()
    configured = [t for t in config.mcp.hitl_approver_tenants if t]
    allowed = {t.lower() for t in configured} or {admin_tenant}
    if tenant not in allowed:
        return JSONResponse(
            status_code=403,
            content={
                "error": "mcp_escalation_not_approver",
                "reason": "this tenant is not configured as an escalation approver",
                "request_id": request_id,
            },
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )
    return tenant, tenant == admin_tenant


def _hitl_unavailable(request: Request, request_id: str) -> JSONResponse | None:
    """503 when HITL is disabled or the broker is missing (fail-closed surface)."""
    broker: HitlBroker | None = getattr(request.app.state, "mcp_hitl", None)
    if broker is None or not request.app.state.config.mcp.hitl_enabled:
        return JSONResponse(
            status_code=503,
            content={
                "error": "mcp_hitl_disabled",
                "reason": "human-in-the-loop approval is not enabled on this gateway",
                "request_id": request_id,
            },
        )
    return None


@router.get("/escalations")
async def list_escalations(request: Request) -> JSONResponse:
    """Pending escalations (approver view: ids, digest, bounded argument preview)."""
    request_id = getattr(request.state, "request_id", None) or "mcp-hitl"
    scope = _approver_scope(request)
    if isinstance(scope, JSONResponse):
        return scope
    approver, is_global = scope
    unavailable = _hitl_unavailable(request, request_id)
    if unavailable is not None:
        return unavailable
    broker: HitlBroker = request.app.state.mcp_hitl
    # Oldest first: a queue that hides its age is how escalations get abandoned.
    records = broker.pending_for_approval()
    if not is_global:
        # Least privilege: a tenant approver sees only its own callers' calls.
        records = [record for record in records if record.tenant == approver]
    return JSONResponse(
        {
            "escalations": [record.to_view() for record in records],
            # Same number as the list — a count that disagrees with what the
            # approver can see is how a queue gets abandoned.
            "pending": len(records),
            "scope": "global" if is_global else approver,
            "timeout_seconds": request.app.state.config.mcp.hitl_timeout_seconds,
        }
    )


@router.post("/escalations/{escalation_id}")
async def decide_escalation(escalation_id: str, request: Request) -> JSONResponse:
    """Approve or deny one pending escalation. Single-shot, audited, attributed."""
    request_id = getattr(request.state, "request_id", None) or "mcp-hitl"
    started = time.perf_counter()
    scope = _approver_scope(request)
    if isinstance(scope, JSONResponse):
        return scope
    approver, is_global = scope
    unavailable = _hitl_unavailable(request, request_id)
    if unavailable is not None:
        return unavailable

    try:
        body = await request.json()
    except ValueError:
        return JSONResponse(
            status_code=400, content={"error": "invalid_json", "request_id": request_id}
        )
    if not isinstance(body, dict) or not isinstance(body.get("approve"), bool):
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_decision",
                "reason": 'body must be {"approve": true|false, "reason": optional string}',
                "request_id": request_id,
            },
        )
    approved = body["approve"]
    raw_reason = body.get("reason")
    if raw_reason is not None and not isinstance(raw_reason, str):
        return JSONResponse(
            status_code=400, content={"error": "invalid_decision", "request_id": request_id}
        )
    reason = (raw_reason or "").strip()[:500] or None

    broker: HitlBroker = request.app.state.mcp_hitl
    record = broker.get(escalation_id)
    if record is None:
        return JSONResponse(
            status_code=404,
            content={"error": "mcp_escalation_unknown", "request_id": request_id},
        )
    if not is_global and record.tenant != approver:
        # Not 404: the approver is allowed to know the id exists (it is in the
        # audit trail) but must not decide outside its own tenant.
        return JSONResponse(
            status_code=403,
            content={
                "error": "mcp_escalation_other_tenant",
                "reason": "this escalation belongs to another tenant; only that "
                "tenant or the admin tenant may decide it",
                "request_id": request_id,
            },
            headers={_VERDICT_HEADER: Verdict.BLOCK.value},
        )
    if record.status != STATUS_PENDING:
        return JSONResponse(
            status_code=409,
            content={
                "error": "mcp_escalation_not_pending",
                "status": record.status,
                "request_id": request_id,
            },
        )

    try:
        decided = await broker.decide(
            escalation_id, approved=approved, decided_by=approver, reason=reason
        )
    except HitlError as exc:  # expired between the check above and the decision
        return JSONResponse(
            status_code=409,
            content={
                "error": "mcp_escalation_not_pending",
                "reason": str(exc),
                "request_id": request_id,
            },
        )

    # The decision is audited against the ESCALATION's tenant (that is whose
    # call it was), with the approver recorded separately — never a secret.
    audit = _audit(request)
    audit.log_mcp_event(
        request_id=request_id,
        tenant_id=record.tenant,
        verdict=Verdict.ALLOW if approved else Verdict.BLOCK,
        rule_id="MCP-HITL-DECISION",
        method=record.method,
        tool=record.tool,
        details={
            "outcome": "escalation_approved" if approved else "escalation_denied",
            "escalation_id": record.escalation_id,
            "agent_id": record.agent,
            "decided_by": approver,
            "decision_reason": reason,
            "request_digest": record.digest,
            "escalation_source": record.source,
        },
        # The real decision-request latency, not a placeholder: this field is
        # in the tamper-evident trail and 0.0 would be a false record.
        total_latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return JSONResponse(decided.to_view())
