"""NeuralGuard data models — request/response schemas and internal types.

All models use Pydantic v2 with strict validation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Enums ──────────────────────────────────────────────────────────────────


class ThreatCategory(StrEnum):
    """OWASP + Agentic-aligned threat categories."""

    PROMPT_INJECTION_DIRECT = "T-PI-D"
    PROMPT_INJECTION_INDIRECT = "T-PI-I"
    JAILBREAK = "T-JB"
    SYSTEM_PROMPT_EXTRACTION = "T-EXT"
    DATA_EXFILTRATION = "T-EXF"
    TOOL_MISUSE = "T-TOOL"
    AGENT_GOAL_HIJACK = "T-AGT"
    ENCODING_EVASION = "T-ENC"
    DOS_ABUSE = "T-DOS"
    IMPROPER_OUTPUT = "T-OUT"
    MEMORY_POISONING = "T-MEM"
    CASCADING_FAILURE = "T-CASC"
    SELF_ATTACK = "T-NG"


class Verdict(StrEnum):
    """Scanner verdict actions — strictest wins in Layer Arbitration."""

    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"
    ESCALATE = "escalate"
    QUARANTINE = "quarantine"
    RATE_LIMIT = "rate_limit"


class ScanLayer(StrEnum):
    """Scanner pipeline layers."""

    STRUCTURAL = "structural"
    PATTERN = "pattern"
    SEMANTIC = "semantic"
    JUDGE = "judge"
    AGENT_GUARDIAN = "agent_guardian"


class Severity(StrEnum):
    """Finding severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ── Request Models ─────────────────────────────────────────────────────────


class Message(BaseModel):
    """Chat message following OpenAI format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content must not be empty")
        return v


class EvaluateRequest(BaseModel):
    """Primary evaluation endpoint request.

    Accepts a full conversation (messages array) for multi-turn analysis
    or a single prompt string for simple checks.

    At least one of `messages` or `prompt` must be provided.
    """

    messages: list[Message] | None = Field(
        default=None, description="Conversation messages (multi-turn)"
    )
    prompt: str | None = Field(default=None, description="Single prompt string (simple mode)")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    session_id: str | None = Field(
        default=None,
        description="Conversation session ID for multi-turn Agent Guardian state. "
        "When provided and agent_guardian is enabled, the scanner keeps a bounded "
        "per-session sliding window of turns to detect delayed injection, role "
        "drift, and accumulation attacks across turns. Sessions are isolated.",
    )
    use_case: Literal["chat", "agent", "rag", "tool", "completion"] = Field(
        default="chat", description="Use case hint for scanner tuning"
    )
    scanners: list[ScanLayer] | None = Field(
        default=None, description="Override enabled scanners (None = tenant defaults)"
    )
    output_only: bool = Field(
        default=False, description="Only run output-relevant patterns (e.g., PII detection)"
    )
    scan_all_roles: bool = Field(
        default=False,
        description="F6: scan ALL message roles (default: user-role turns only). "
        "The defender's own system prompt ('You are X, you must never...') fires "
        "PI-D/JB patterns on itself when role-blind — harmless for user-turn "
        "traffic, catastrophic in proxy mode where full chat payloads flow. "
        "Opt in only when you understand the false-block tradeoff.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional request metadata")

    def input_texts(self) -> list[str]:
        """Texts to scan: user-role turns only by default (F6).

        ``scan_all_roles=True`` opts back into every role. The pattern layer
        (PI-D/JB) and structural scanner both use this; Agent Guardian always
        filtered roles itself; output scanning uses the completion text, not
        this helper.
        """
        if self.messages:
            if self.scan_all_roles:
                return [m.content for m in self.messages]
            return [m.content for m in self.messages if m.role == "user"]
        return [self.prompt] if self.prompt else []

    @field_validator("tenant_id")
    @classmethod
    def tenant_id_valid(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tenant_id must not be empty")
        if len(v) > 64:
            raise ValueError("tenant_id must be <= 64 chars")
        return v.strip().lower()

    @model_validator(mode="after")
    def validate_scanners_unique(self) -> EvaluateRequest:
        """Remove duplicate scanner layers."""
        if self.scanners is not None:
            seen = set()
            unique = []
            for s in self.scanners:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            if len(unique) < len(self.scanners):
                self.scanners = unique
        return self

    @model_validator(mode="after")
    def validate_input_provided(self) -> EvaluateRequest:
        """Ensure at least one of messages or prompt is provided."""
        if self.messages is None and not self.prompt:
            raise ValueError("At least one of 'messages' or 'prompt' must be provided")
        return self


class ScanOutputRequest(BaseModel):
    """Request to scan LLM output before delivery."""

    output: str = Field(description="LLM response text to validate")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    session_id: str | None = Field(
        default=None, description="Session ID for canary token verification"
    )
    system_prompt_hash: str | None = Field(
        default=None, description="Hash of system prompt for integrity check"
    )

    @field_validator("output")
    @classmethod
    def output_not_empty(cls, v: str) -> str:
        """Validate that output is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("Output must not be empty or whitespace-only")
        return v


class TemplateSinkFinding(BaseModel):
    """A static injection sink found in a prompt template (B2)."""

    rule_id: str
    severity: Literal["high", "medium", "low", "info"]
    description: str
    remediation: str
    evidence: str
    location: int = Field(ge=1, description="1-indexed line number of the sink")


class AnalyzeTemplateRequest(BaseModel):
    """Request to statically analyze a system-prompt template for injection sinks."""

    template: str = Field(
        description=(
            "The system-prompt template to analyze. May contain placeholders "
            "like {{var}}, ${var}, {var}, or <user_input>."
        ),
        min_length=1,
        max_length=64_000,
    )
    tenant_id: str = Field(default="default", description="Tenant identifier")

    @field_validator("template")
    @classmethod
    def template_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Template must not be empty or whitespace-only")
        return v


class AnalyzeTemplateResponse(BaseModel):
    """Response from the template analyzer endpoint."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    is_clean: bool
    sink_count: int
    sinks: list[TemplateSinkFinding] = Field(default_factory=list)
    total_latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CanaryMintRequest(BaseModel):
    """Request to mint per-session canary token(s) (B3).

    The returned token(s) are injected into the LLM system prompt by the
    operator. If a token later appears in the model output, ``/v1/scan/output``
    flags it as a system-prompt exfiltration signal.
    """

    session_id: str = Field(description="Session to bind the canary token(s) to.")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    count: int | None = Field(
        default=None,
        description="Number of distinct canaries to mint (1-8). Defaults to the configured token_count.",
    )

    @field_validator("session_id")
    @classmethod
    def session_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("session_id must not be empty")
        return v.strip()

    @field_validator("count")
    @classmethod
    def validate_count(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if not (1 <= v <= 8):
            raise ValueError("count must be 1-8")
        return v


class CanaryMintResponse(BaseModel):
    """Response from the canary mint endpoint."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    session_id: str
    tokens: list[str] = Field(description="Canary token(s) to inject into the system prompt.")
    total_latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Internal Models ────────────────────────────────────────────────────────


class Finding(BaseModel):
    """Single detection finding from a scanner."""

    category: ThreatCategory
    severity: Severity
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence 0-1")
    layer: ScanLayer
    rule_id: str = Field(description="Pattern/rule identifier (e.g. 'PI-D-001')")
    description: str = Field(description="Human-readable finding description")
    evidence: str | None = Field(default=None, description="Matched text snippet (tokenized)")
    mitigation: str | None = Field(default=None, description="Recommended mitigation")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScannerResult(BaseModel):
    """Result from a single scanner layer."""

    layer: ScanLayer
    verdict: Verdict
    findings: list[Finding] = Field(default_factory=list)
    latency_ms: float = Field(description="Scanner execution time in ms")
    error: str | None = Field(default=None, description="Error if scanner failed")
    sanitized_output: str | None = Field(
        default=None, description="Sanitized content if verdict is SANITIZE"
    )


class LayerArbitrationResult(BaseModel):
    """Final arbitration result across all scanner layers.

    Rule: strictest verdict wins. BLOCK cannot be overridden without
    explicit FORCE_ALLOW audit trail.
    """

    verdict: Verdict
    findings: list[Finding] = Field(default_factory=list)
    scanner_results: list[ScannerResult] = Field(default_factory=list)
    total_latency_ms: float = Field(description="Total pipeline latency in ms")
    arbitration_reason: str = Field(description="Why this verdict was chosen (audit trail)")


# ── Response Models ────────────────────────────────────────────────────────


class EvaluateResponse(BaseModel):
    """Primary evaluation endpoint response."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    verdict: Verdict
    findings: list[Finding] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Max confidence across findings")
    sanitized_content: str | None = Field(
        default=None, description="Sanitized input if verdict is SANITIZE"
    )
    scan_layers_used: list[ScanLayer]
    total_latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScanOutputResponse(BaseModel):
    """Output scan endpoint response."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    verdict: Verdict
    findings: list[Finding] = Field(default_factory=list)
    redacted_output: str | None = Field(default=None, description="PII-redacted output")
    canary_leaked: bool = Field(
        default=False, description="Whether canary token was detected in output"
    )
    total_latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    environment: str
    scanners: dict[str, bool] = Field(description="Scanner availability status")
    uptime_seconds: float


# ── Tenant Models (Sprint C, C1) ──────────────────────────────────────────


class TenantScannerOverridesView(BaseModel):
    """Public view of a tenant's scanner overrides (None = inherit global)."""

    agent_guardian: bool | None = None
    semantic: bool | None = None
    judge: bool | None = None


class TenantInfoResponse(BaseModel):
    """Effective per-tenant configuration (read-only, no secrets).

    Returned by ``GET /v1/tenants/{tenant_id}`` and surfaced in the
    ``neuralguard tenants info`` CLI. The ``effective_*`` fields resolve the
    ``None``-inherits-global overlay against the live global config so an
    operator can see exactly what applies to a tenant without doing the
    inheritance math themselves.
    """

    tenant_id: str
    description: str | None = None
    configured: bool = Field(
        description="False when the tenant has no override file (global defaults apply)."
    )
    requests_per_minute: int | None = None
    burst_size: int | None = None
    effective_requests_per_minute: int
    effective_burst_size: int
    scanners: TenantScannerOverridesView
    effective_scanners: dict[str, bool] = Field(
        description="Resolved per-tenant scanner enable state (layer -> bool)."
    )


class TenantListResponse(BaseModel):
    """List of configured tenants (read-only)."""

    tenants: list[TenantInfoResponse]
    count: int


# ── Audit Models ───────────────────────────────────────────────────────────


class AuditEvent(BaseModel):
    """Structured audit event for logging and compliance.

    Tamper-evidence: every event carries ``worker_id`` (the chain it belongs
    to), ``prev_hash`` (the previous event's ``event_hash`` in this worker's
    chain, or ``None`` for the chain head), and ``event_hash`` (SHA-256 over a
    canonical encoding of the event plus ``prev_hash``). See
    ``neuralguard.logging.chain``.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    tenant_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verdict: Verdict
    findings_count: int
    threat_categories: list[ThreatCategory]
    confidence: float
    total_latency_ms: float
    scanner_details: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    worker_id: str | None = Field(default=None, description="Chain id (per process).")
    prev_hash: str | None = Field(
        default=None, description="Previous event's event_hash, or None for the chain head."
    )
    event_hash: str | None = Field(default=None, description="SHA-256 chain hash of this event.")
