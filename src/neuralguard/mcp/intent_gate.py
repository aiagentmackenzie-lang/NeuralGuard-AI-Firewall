"""NG-8: header-based per-tool Intent Gate (pure core).

The MCP 2026-07-28 spec added first-class ``Mcp-Method`` / ``Mcp-Name`` HTTP
headers precisely so that "your gateway, rate limiter, or WAF can route and
authorize on those headers instead of parsing JSON bodies." The OWASP Agentic
Top 10 (ASI02, Tool Misuse) mitigation guidance calls this seam an
**Intent Gate**: an enforcement point that decides allow / deny /
require-human-approval **per tool, per tenant, on headers alone**.

Decisions here are pure and made BEFORE the body is parsed — the cheapest
possible rejection path, and immune to body-parser differentials. The one
body-derived check (header/body agreement) happens at the route layer after
parsing: when the declared intent (headers) disagrees with the actual intent
(body), that is smuggling, and it BLOCKS.

Fail-closed postures:
- Required headers missing (when ``headers_required``) -> reject before parse.
- An UNKNOWN tool never grants escalation: policy lookups only narrow.
- DENY and ESCALATE decisions are audit events, never silent.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from neuralguard.mcp.constraints import (
    ArgumentConstraint,  # noqa: TC001 - runtime pydantic field type
)
from neuralguard.models.schemas import Verdict


class ToolAction(StrEnum):
    """What the Intent Gate does with one (method, tool) intent."""

    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"  # require-human-approval (HITL) — never auto-passes


# ToolAction -> HTTP-ish outcome mapping lives at the route layer; the gate
# speaks the domain language. Verdict mapping for findings:
_ACTION_TO_VERDICT: dict[ToolAction, Verdict] = {
    ToolAction.ALLOW: Verdict.ALLOW,
    ToolAction.DENY: Verdict.BLOCK,
    ToolAction.ESCALATE: Verdict.ESCALATE,
}


class AgentToolPolicy(BaseModel):
    """NG-11: what one workload agent may do, inside its tenant.

    Agent rules are **narrowing only**: every applicable action is resolved
    with the tenant's rules and the strictest wins, and agent constraints are
    ADDED to the tenant's (both must pass). There is no field here that can
    widen a tenant policy, which is the property that makes this safe to hand
    to a per-agent config.

    Tool-scoped on purpose: an agent's rules apply to ``tools/call``. Method
    rules stay a tenant-level concern (an agent that could allow
    ``resources/list`` for itself would be re-shaping the gateway, not its own
    authority).
    """

    model_config = {"extra": "forbid"}

    tool_rules: dict[str, ToolAction] = Field(
        default_factory=dict,
        description="Per-tool action for this agent; resolved strictest-wins "
        "against the tenant's rule for the same tool.",
    )
    default_tool_action: ToolAction | None = Field(
        default=None,
        description="Action for a tool with no agent rule. None inherits the "
        "tenant's default_agent_action (which defaults to allow).",
    )
    tool_constraints: dict[str, list[ArgumentConstraint]] = Field(
        default_factory=dict,
        description="NG-10 constraints for this agent, evaluated IN ADDITION "
        "to the tenant's — a violation of either denies.",
    )


class McpToolPolicy(BaseModel):
    """A tenant's per-tool / per-method MCP policy (NG-8).

    Attached to ``TenantConfig.mcp``. Everything is optional/empty by
    default: an empty policy allows all intent — the defense in depth for
    unknown tools is NG-7's baseline (a tool the catalog never contained
    cannot pass a strict gateway regardless of policy). Explicit rules
    always win over the default; the most restrictive interpretation
    applies when a tool matches BOTH a tool rule and a method rule.
    """

    model_config = {"extra": "forbid"}

    tool_rules: dict[str, ToolAction] = Field(
        default_factory=dict,
        description="Exact tool-name -> action. Evaluated on Mcp-Name (tools/call) "
        "or, for catalog methods, not at all (method rules cover those).",
    )
    method_rules: dict[str, ToolAction] = Field(
        default_factory=dict,
        description="Exact MCP method name -> action (e.g. deny 'tools/call', "
        "escalate 'tools/call' wholesale, allow 'tools/list').",
    )
    default_tool_action: ToolAction = Field(
        default=ToolAction.ALLOW,
        description="Action for a tool with no explicit rule. Deliberately "
        "ALLOW (documented): unknown-tool protection is NG-7's baseline job; "
        "tenants who want default-deny set this to deny.",
    )
    default_method_action: ToolAction = Field(
        default=ToolAction.ALLOW,
        description="Action for a method with no explicit rule.",
    )
    egress_tools: list[str] = Field(
        default_factory=list,
        description=(
            "NG-9: tools this tenant classifies as EGRESS-CAPABLE (send data "
            "outside the trust boundary: email, HTTP post, upload, ...). "
            "Provenance checks apply only to these; internal tools pass "
            "regardless. The operator chooses the mode globally "
            "(NEURALGUARD_MCP_PROVENANCE_MODE) — the tenant knows its own "
            "tools."
        ),
    )
    agent_rules: dict[str, AgentToolPolicy] = Field(
        default_factory=dict,
        description=(
            "NG-11: per-workload-agent narrowing, keyed by agent id from the "
            "authenticated credential. Empty = identity is audited but not "
            "used for authorization. Non-empty = the agent layer applies to "
            "every caller, including callers with no agent identity (see "
            "default_agent_action)."
        ),
    )
    default_agent_action: ToolAction = Field(
        default=ToolAction.ALLOW,
        description=(
            "Action for a caller whose agent is not enumerated in agent_rules "
            "(including a tenant-level credential with no agent). Deliberately "
            "ALLOW so adding agent_rules for one bot cannot lock everyone else "
            "out; operators who are enumerating their fleet set this to deny."
        ),
    )
    tool_constraints: dict[str, list[ArgumentConstraint]] = Field(
        default_factory=dict,
        description=(
            "NG-10: per-tool constraints over the parsed call arguments "
            "(path/URL allowlists, enums, lengths, deny patterns, required "
            "keys). Evaluated after the body is parsed and before the call is "
            "forwarded; absent tools are unconstrained. The Intent Gate decides "
            "whether a tool may run, these decide how it may be asked."
        ),
    )

    def agent_policy_for(self, agent: str | None) -> AgentToolPolicy | None:
        """The narrowing layer for this caller, or None if identity is unused.

        An unknown or absent agent is NOT "any agent": when the operator has
        started enumerating agents, an unenumerated caller is governed by
        ``default_agent_action`` instead of silently escaping the agent layer.
        """
        if not self.agent_rules:
            return None
        if agent is not None:
            enumerated = self.agent_rules.get(agent)
            if enumerated is not None:
                return enumerated
        return AgentToolPolicy(default_tool_action=self.default_agent_action)

    def action_for(self, method: str, tool: str | None, agent: str | None = None) -> ToolAction:
        """Most-restrictive-wins resolution for one intent.

        Both the method rule and the tool rule apply; the strictest of the
        applicable actions wins (DENY > ESCALATE > ALLOW). Unknown methods
        take ``default_method_action``; unknown tools take
        ``default_tool_action``.

        NG-11: when the caller carries an agent identity and the tenant has
        agent rules, that agent's layer joins the resolution as one more
        candidate — so an agent can narrow what its tenant allows but can
        never widen it. Agent rules are tool-scoped: for a non-``tools/call``
        method (``tool is None``) the agent layer does not participate.
        """
        candidates: list[ToolAction] = []
        method_rule = self.method_rules.get(method)
        if method_rule is not None:
            candidates.append(method_rule)
        else:
            candidates.append(self.default_method_action)
        if tool is not None:
            tool_rule = self.tool_rules.get(tool)
            if tool_rule is not None:
                candidates.append(tool_rule)
            else:
                candidates.append(self.default_tool_action)
            agent_layer = self.agent_policy_for(agent)
            if agent_layer is not None:
                agent_rule = agent_layer.tool_rules.get(tool)
                if agent_rule is not None:
                    candidates.append(agent_rule)
                else:
                    candidates.append(
                        agent_layer.default_tool_action
                        if agent_layer.default_tool_action is not None
                        else self.default_agent_action
                    )
        # strictness: DENY(2) > ESCALATE(1) > ALLOW(0)
        return max(candidates, key=lambda a: _STRICTNESS[a])

    def to_effective_dict(self) -> dict[str, Any]:
        """Secret-free public view for the tenants API/CLI."""
        return {
            "tool_rules": {k: v.value for k, v in sorted(self.tool_rules.items())},
            "method_rules": {k: v.value for k, v in sorted(self.method_rules.items())},
            "default_tool_action": self.default_tool_action.value,
            "default_method_action": self.default_method_action.value,
            "egress_tools": sorted(self.egress_tools),
            "tool_constraints": {
                tool: [c.model_dump(mode="json") for c in rules]
                for tool, rules in sorted(self.tool_constraints.items())
            },
            "default_agent_action": self.default_agent_action.value,
            "agent_rules": {
                agent: rule.model_dump(mode="json")
                for agent, rule in sorted(self.agent_rules.items())
            },
        }


class GateDecision(BaseModel):
    """The Intent Gate's verdict for one request — made before body parse."""

    action: ToolAction
    verdict: Verdict
    rule_id: str = Field(description="MCP-GATE-* rule id for findings/audit.")
    reason: str
    declared_method: str | None = None
    declared_tool: str | None = None


_STRICTNESS = {
    ToolAction.ALLOW: 0,
    ToolAction.ESCALATE: 1,
    ToolAction.DENY: 2,
}

_MCP_METHOD_HEADER = "Mcp-Method"
_MCP_NAME_HEADER = "Mcp-Name"


def _decision(
    action: ToolAction, rule_id: str, reason: str, method: str | None, tool: str | None
) -> GateDecision:
    return GateDecision(
        action=action,
        verdict=_ACTION_TO_VERDICT[action],
        rule_id=rule_id,
        reason=reason,
        declared_method=method,
        declared_tool=tool,
    )


def evaluate_intent(
    headers: dict[str, str],
    policy: McpToolPolicy,
    *,
    headers_required: bool = True,
    agent: str | None = None,
) -> GateDecision:
    """NG-8 Intent Gate: decide on headers ALONE, before body parse.

    ``headers`` is a case-insensitive-able dict (the route passes
    ``{k.lower(): v}``). Returns the decision; the route turns DENY/ESCALATE
    into the right HTTP outcome + audit event and ALLOW into the parse step.

    ``agent`` (NG-11) is the authenticated workload identity, threaded from
    the credential — never from a header. It participates in resolution only
    (see :meth:`McpToolPolicy.action_for`).
    """
    method = headers.get(_MCP_METHOD_HEADER.lower(), "").strip()
    tool = headers.get(_MCP_NAME_HEADER.lower(), "").strip() or None

    if not method:
        if headers_required:
            return _decision(
                ToolAction.DENY,
                "MCP-GATE-HEADER-001",
                f"Required '{_MCP_METHOD_HEADER}' header missing — the Intent Gate "
                "cannot authorize an undeclared intent (fail-closed).",
                method or None,
                tool,
            )
        return _decision(
            ToolAction.ALLOW,
            "MCP-GATE-HEADER-002",
            "Headers absent and not required — gate bypassed by configuration "
            "(logged: this disables the pre-parse seam).",
            None,
            None,
        )

    action = policy.action_for(method, tool, agent)
    if action == ToolAction.DENY:
        return _decision(
            ToolAction.DENY,
            "MCP-GATE-DENY-001",
            f"Intent ({method=}, {tool=}) matched a deny rule — rejected on "
            "headers alone, body never parsed.",
            method,
            tool,
        )
    if action == ToolAction.ESCALATE:
        return _decision(
            ToolAction.ESCALATE,
            "MCP-GATE-ESC-001",
            f"Intent ({method=}, {tool=}) is policy-gated for human approval.",
            method,
            tool,
        )
    return _decision(
        ToolAction.ALLOW,
        "MCP-GATE-ALLOW-001",
        f"Intent ({method=}, {tool=}) allowed by policy.",
        method,
        tool,
    )


def check_header_body_agreement(
    declared_method: str | None,
    declared_tool: str | None,
    body_method: str | None,
    body_tool: str | None,
) -> GateDecision | None:
    """Post-parse integrity check: declared intent (headers) vs actual intent (body).

    A mismatch is smuggling — an evasion technique against header-gated
    gateways (declare a benign tool, ship a malicious call). Returns a DENY
    decision on mismatch, None when the declared and actual intents agree
    (or when nothing was declared).
    """
    if declared_method is None:
        return None  # gate was bypassed by config; body stands alone
    if (declared_method or "").strip() != (body_method or "").strip():
        return _decision(
            ToolAction.DENY,
            "MCP-GATE-SMUGGLE-001",
            f"Header/body method mismatch: declared {declared_method!r}, "
            f"body carries {body_method!r} — smuggling posture, blocked.",
            declared_method,
            declared_tool,
        )
    if (declared_tool or "") != (body_tool or ""):
        return _decision(
            ToolAction.DENY,
            "MCP-GATE-SMUGGLE-002",
            f"Header/body tool mismatch: declared {declared_tool!r}, "
            f"body carries {body_tool!r} — smuggling posture, blocked.",
            declared_method,
            declared_tool,
        )
    return None


# JSON-RPC extraction helper (pure): the route uses these to compare against
# the declared headers. Kept here so the parsing contract lives next to the
# agreement check.
def body_intent(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract (method, tool_name) from a JSON-RPC body, defensively.

    ``tool_name`` is present only for ``tools/call`` (``params.name``).
    Malformed bodies yield (None, None) — the route's body-parse error path
    handles them; this helper never raises.
    """
    try:
        method = payload.get("method")
        params = payload.get("params") or {}
        tool = params.get("name") if isinstance(params, dict) else None
        if isinstance(method, str):
            return method, str(tool) if tool is not None else None
    except AttributeError:
        pass
    return None, None
