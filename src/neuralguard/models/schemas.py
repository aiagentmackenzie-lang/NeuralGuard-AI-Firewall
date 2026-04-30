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
    use_case: Literal["chat", "agent", "rag", "tool", "completion"] = Field(
        default="chat", description="Use case hint for scanner tuning"
    )
    scanners: list[ScanLayer] | None = Field(
        default=None, description="Override enabled scanners (None = tenant defaults)"
    )
    output_only: bool = Field(
        default=False, description="Only run output-relevant patterns (e.g., PII detection)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional request metadata")

    @field_validator("tenant_id")
    @classmethod
    def tenant_id_valid(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tenant_id must not be empty")
        if len(v) > 64:
            raise ValueError("tenant_id must be <= 64 chars")
        return v.strip().lower()

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


# ── Audit Models ───────────────────────────────────────────────────────────


class AuditEvent(BaseModel):
    """Structured audit event for logging and compliance."""

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
