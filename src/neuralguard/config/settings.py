"""NeuralGuard configuration — environment and file-based settings.

Uses pydantic-settings for layered configuration:
  1. Environment variables (NEURALGUARD_ prefix)
  2. .env file
  3. config.yaml (optional)
  4. Defaults
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ServerSettings(BaseSettings):
    """HTTP server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="Bind address")
    port: int = Field(default=8000, description="Bind port")
    workers: int = Field(default=1, description="Uvicorn workers")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Log level"
    )
    cors_origins: list[str] = Field(
        default_factory=list,
        description="Allowed CORS origins. Empty = no CORS (API-only). "
        'Explicit allowlist for production. NEVER use ["*"] with allow_credentials=true.',
    )
    allow_credentials: bool = Field(
        default=False,
        description="CORS allow_credentials. Only enable if you use authenticated browser sessions.",
    )
    max_request_body_bytes: int = Field(
        default=1_048_576,
        description="Hard cap on inbound request body size (bytes). 413 returned beyond this. Default 1 MiB.",
    )
    allow_insecure_http: bool = Field(
        default=False,
        description="Allow serving plain HTTP in production. When false (default), the "
        "production lifespan logs a loud TLS notice on every boot. Set true only behind "
        "a TLS-terminating reverse proxy.",
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be 1-65535, got {v}")
        return v

    @field_validator("max_request_body_bytes")
    @classmethod
    def validate_body_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"max_request_body_bytes must be > 0, got {v}")
        return v


class ScannerSettings(BaseSettings):
    """Scanner pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_SCANNER_",
        env_file=".env",
        extra="ignore",
    )

    # Structural validation
    max_input_length: int = Field(default=32_000, description="Max input length in chars")
    max_decompression_ratio: float = Field(
        default=10.0, description="Max input/output size ratio (decompression bomb limit)"
    )

    # Pattern scanner
    regex_timeout_ms: int = Field(default=50, description="Regex compilation/execution timeout")
    max_regex_complexity: int = Field(default=20, description="Max quantifier nesting depth")

    # F11: per-text aggregate deadline for the pattern layer. The per-pattern
    # timeout (regex_timeout_ms) bounds each regex individually, but the
    # aggregate across all compiled patterns was unbounded (~5.6s worst case
    # per text for a crafted ReDoS-bait input: 113 patterns x 50ms). Patterns
    # beyond this budget are SKIPPED and a SELF_ATTACK/PATTERN-BUDGET ESCALATE
    # finding is emitted — fail toward review, never silently weaker.
    pattern_budget_ms: int = Field(
        default=300,
        ge=1,
        le=60_000,
        description=(
            "Per-text aggregate deadline (ms) for the pattern layer; exceeding it skips "
            "remaining patterns and emits a PATTERN-BUDGET escalate finding"
        ),
    )

    # Semantic scanner (Phase 2)
    semantic_enabled: bool = Field(default=False, description="Enable semantic classification")
    semantic_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformer model name",
    )
    semantic_onnx_path: str = Field(
        default="models/embedding-onnx",
        description="Path to ONNX model directory",
    )
    semantic_max_seq_length: int = Field(
        default=256, description="Max token sequence length for embedding"
    )
    semantic_intra_threads: int = Field(
        default=0, description="ONNX Runtime intra-op threads (0=auto)"
    )
    semantic_similarity_threshold: float = Field(
        default=0.75, description="Cosine similarity threshold for BLOCK"
    )
    semantic_attack_corpus_path: str = Field(
        default="models/attack_vectors.npy",
        description="Path to pre-computed attack vector embeddings",
    )
    semantic_attack_metadata_path: str = Field(
        default="models/attack_metadata.json",
        description="Path to attack corpus metadata JSON",
    )

    # LLM-as-Judge (Phase 2)
    judge_enabled: bool = Field(default=False, description="Enable LLM-as-Judge")
    judge_model: str = Field(
        default="mistral:7b", description="Judge model identifier (Ollama tag)"
    )
    judge_max_tokens: int = Field(default=512, description="Max tokens for judge response")
    judge_temperature: float = Field(default=0.0, description="Judge sampling temperature")
    judge_ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama base URL for the judge. Production ENFORCES a "
        "loopback/private address unless NEURALGUARD_SCANNER_JUDGE_ALLOW_EGRESS=true "
        "(explicit, logged) — F10.3.",
    )
    judge_timeout_seconds: int = Field(
        default=5,
        description="Hard timeout for one judge HTTP call. Cloud-via-Ollama models "
        "and large local models (cold starts) need more than the old hardcoded 5s "
        "in some environments — raise it deliberately.",
    )
    judge_allow_egress: bool = Field(
        default=False,
        description="F10.3: explicit opt-in to a NON-loopback judge endpoint (cloud-via-Ollama). "
        "When true, prompts leave the trust boundary — the startup logs it loudly and "
        "readiness surfaces it. Never enable for sensitive workloads.",
    )
    judge_resolves_escalate: bool = Field(
        default=False,
        description="F20: when True, a clean LLM-Judge ALLOW verdict resolves an ESCALATE "
        "(ambiguous) verdict to ALLOW — the judge is the authoritative resolver of the "
        "ambiguous zone. The judge cannot downgrade SANITIZE/BLOCK/QUARANTINE. A skipped, "
        "timed-out, or errored judge does NOT resolve (fail-closed). Tradeoff (measured "
        "in A2): with a weak judge this drops benign-prompt FPR to 0% but also lets "
        "through attacks the semantic layer caught as ESCALATE when the judge "
        "false-negatives them (7B mistral false-negatives ContextPoison "
        "extract_system_prompt). Only enable with a judge measured reliable enough "
        "(27B re-measurement, Phase 2). Default False keeps the safe behavior. NOTE: "
        "this field was previously on ActionSettings — passing it there (or the old "
        "NEURALGUARD_ACTION_ prefix) is now dead.",
    )
    judge_max_concurrency: int = Field(
        default=4,
        description="Max concurrent in-flight judge HTTP calls per worker",
    )


class ActionSettings(BaseSettings):
    """Response action configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_ACTION_",
        env_file=".env",
        extra="ignore",
    )

    default_action: Literal["allow", "block", "sanitize"] = Field(
        default="block", description="Default action on scanner failure (fail-closed)"
    )
    fail_closed: bool = Field(default=True, description="Block on any scanner error")
    score_threshold_block: float = Field(
        default=0.85, description="Block threshold (>= this score = BLOCK)"
    )
    score_threshold_sanitize: float = Field(
        default=0.60, description="Sanitize threshold (>= this score = SANITIZE)"
    )
    enable_escalation: bool = Field(default=False, description="Enable HITL escalation webhook")
    escalation_webhook_url: str | None = Field(
        default=None, description="Webhook URL for HITL escalation"
    )
    semantic_sanitize_requires_corroboration: bool = Field(
        default=True,
        description="When True, the hybrid engine will NOT SANITIZE (modify content) on a lone "
        "ambiguous semantic match (similarity below the semantic BLOCK floor). "
        "A single ambiguous semantic signal produces ESCALATE (review / judge) "
        "instead, and SANITIZE requires either pattern corroboration or semantic "
        "similarity at/above the BLOCK floor. This eliminates the semantic-layer "
        "FPR on benign creative/translation prompts (A2 finding). Set False to "
        "restore the pre-fix behavior (semantic-alone can SANITIZE at composite "
        ">= score_threshold_sanitize).",
    )


class AuthSettings(BaseSettings):
    """API-key authentication and authorization configuration.

    API keys are bound to a tenant. The authenticated tenant overrides any
    client-supplied tenant_id, preventing tenant-spoofing rate-limit bypass.
    Key format in config: either a bare key (tenant="default") or
    "<key>|<tenant_id>" to bind a key to a specific tenant.

    In production, the application lifespan refuses to start unless `enabled`
    is true and at least one key is configured.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_AUTH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable API-key authentication")
    api_keys: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        description="API keys. Bare key -> tenant 'default'; '<key>|<tenant>' binds a tenant.",
    )
    public_endpoints: list[str] = Field(
        default_factory=lambda: ["/v1/health"],
        description="Paths accessible without auth (unauthenticated). Kept minimal.",
    )
    enforce_tenant_from_key: bool = Field(
        default=True,
        description="When true, the request tenant_id is forced to the key's bound tenant; "
        "a body/header tenant_id that disagrees is rejected with 403.",
    )

    # ── P2-4: JWT bearer auth + runtime key rotation ──

    jwt_enabled: bool = Field(
        default=False,
        description="Accept short-lived JWT bearer tokens (issued via POST /v1/auth/token)",
    )
    jwt_secret: str | None = Field(
        default=None,
        description="HS256 signing secret for issued tokens. Required (≥32 chars) when "
        "jwt_enabled. Held server-side, never logged.",
    )
    jwt_ttl_minutes: int = Field(default=15, ge=1, le=1440, description="TTL for issued tokens")
    admin_tenant: str = Field(
        default="admin",
        description="Tenant whose keys/tokens may call the rotation API",
    )
    keys_file: Path | None = Field(
        default=None,
        description=(
            "Runtime key store (JSON) for the rotation API. Required for DURABLE "
            "rotation: without it, rotation is runtime-only and refused in "
            "production. Atomic writes, 0600."
        ),
    )

    @field_validator("jwt_secret", mode="after")
    @classmethod
    def _validate_jwt_secret(cls, v: str | None, info: Any) -> str | None:
        if v is None:
            return v
        if len(v) < 32:
            raise ValueError(
                "jwt_secret must be ≥32 characters (use a generated token, "
                "e.g. `python3 -c 'import secrets; print(secrets.token_hex(32))'`)"
            )
        return v

    @model_validator(mode="after")
    def _jwt_requires_secret(self) -> AuthSettings:
        if self.jwt_enabled and (not self.jwt_secret or len(self.jwt_secret) < 32):
            raise ValueError("jwt_enabled=true requires NEURALGUARD_AUTH_JWT_SECRET (≥32 chars)")
        return self

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v: Any) -> list[str]:
        # Accept comma-separated string from env, or a list.
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        if isinstance(v, (list, tuple)):
            return [str(k).strip() for k in v if str(k).strip()]
        return v if isinstance(v, list) else []

    def key_to_tenant(self) -> dict[str, str]:
        """Return {api_key: tenant_id} mapping from the configured key list."""
        mapping: dict[str, str] = {}
        for entry in self.api_keys:
            if "|" in entry:
                key, tenant = entry.split("|", 1)
                mapping[key.strip()] = tenant.strip().lower() or "default"
            else:
                mapping[entry.strip()] = "default"
        return mapping


class AuditSettings(BaseSettings):
    """Audit logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_AUDIT_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable audit logging")
    backend: Literal["jsonl", "postgres"] = Field(default="jsonl", description="Audit backend")
    jsonl_path: Path = Field(
        default=Path("./audit_logs"), description="Directory for JSONL audit files"
    )
    postgres_url: str | None = Field(default=None, description="PostgreSQL connection string")
    retention_days: int = Field(default=30, description="Audit log retention in days")
    tokenize_pii: bool = Field(default=True, description="Tokenize PII in audit logs")
    max_inflight_writes: int = Field(
        default=1000,
        description="Max concurrent in-flight async Postgres writes per worker. "
        "Beyond this, audit events fall back to JSONL to bound memory.",
    )
    signing_key: str | None = Field(
        default=None,
        description=(
            "Ed25519 PRIVATE seed (hex, 32 bytes) for audit-event signing (P2-10). "
            "When set, every persisted event's chain hash is signed (event_sig). "
            "Verify with: neuralguard audit-verify --pubkey <derived pubkey hex>. "
            "Held server-side, never logged. Generate: neuralguard audit-keygen."
        ),
    )

    @field_validator("signing_key", mode="after")
    @classmethod
    def _validate_signing_key(cls, v: str | None) -> str | None:
        if v is None:
            return v
        from neuralguard.logging.signing import SigningKeyError, public_key_from_seed

        try:
            public_key_from_seed(v)
        except SigningKeyError as exc:
            raise ValueError(f"audit signing_key invalid: {exc}") from exc
        return v


class SiemSettings(BaseSettings):
    """SIEM routing + BLOCK-rate spike alerting (P2-7).

    Splunk HEC is a native sink; the generic JSON webhook covers ELK /
    Sentinel via their supported ingestion integrations. Disabled by
    default: zero routing overhead until explicitly enabled.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_SIEM_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable SIEM event routing")
    splunk_hec_url: str | None = Field(
        default=None,
        description="Splunk HTTP Event Collector base URL, e.g. https://splunk.example.com:8088",
    )
    splunk_hec_token: str | None = Field(
        default=None,
        description="Splunk HEC token (held server-side, never logged)",
    )
    splunk_source_type: str = Field(
        default="neuralguard:verdict", description="Splunk sourcetype for routed events"
    )
    webhook_url: str | None = Field(
        default=None,
        description=(
            "Generic JSON webhook sink (ELK webhook input, Sentinel Logic App, any JSON collector)"
        ),
    )
    webhook_token: str | None = Field(
        default=None,
        description="Optional bearer token for the generic webhook (never logged)",
    )
    timeout_seconds: float = Field(
        default=5.0, ge=0.5, le=60, description="Per-delivery HTTP timeout"
    )
    max_inflight: int = Field(
        default=20,
        ge=1,
        le=1000,
        description=(
            "Max concurrent SIEM deliveries; beyond this events are DROPPED "
            "with a warning (bounded memory — no unbounded queue)"
        ),
    )
    spike_window: int = Field(
        default=100,
        ge=10,
        le=100000,
        description="Spike detector: sliding window size in verdicts",
    )
    spike_block_threshold: float = Field(
        default=0.5,
        ge=0.01,
        le=1.0,
        description="Spike detector: BLOCK ratio over the window that triggers an alert",
    )
    spike_cooldown_seconds: int = Field(
        default=300,
        ge=0,
        le=86400,
        description="Minimum seconds between spike alerts (edge-trigger re-arm)",
    )


class TenantSettings(BaseSettings):
    """Multi-tenant configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_TENANT_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable multi-tenant mode")
    default_tenant: str = Field(default="default", description="Default tenant ID")
    config_path: Path = Field(
        default=Path("./tenants"), description="Directory for tenant config files"
    )
    reload_interval_seconds: float = Field(
        default=30.0,
        description="Background hot-reload poll interval for the tenant config dir (0=off). "
        "On mtime change the registry re-parses; a parse error keeps the last-good config.",
    )

    @field_validator("reload_interval_seconds")
    @classmethod
    def _validate_reload_interval(cls, v: float) -> float:
        if v < 0:
            raise ValueError("reload_interval_seconds must be >= 0 (0 disables hot-reload)")
        return v


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration.

    Backend selection:
    - ``memory``: per-process sliding window. Correct for single-worker deploys.
      With ``workers > 1`` each worker keeps its own counter, so a tenant can
      make up to ``(limit + burst) * workers`` requests per window. The
      production lifespan refuses to start in this configuration — use
      ``redis`` for multi-worker deploys.
    - ``redis``: shared sliding-window counter backed by Redis (ZSET + Lua).
      Correct across workers. Requires the ``[redis]`` extra and a reachable
      Redis at ``redis_url``.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_RATELIMIT_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable rate limiting")
    backend: Literal["memory", "redis"] = Field(
        default="memory",
        description="Rate-limit backend. 'redis' is required for multi-worker production.",
    )
    redis_url: str | None = Field(
        default=None,
        description="Redis URL (redis://[:password@]host:port/db). Required when backend=redis.",
    )
    requests_per_minute: int = Field(default=60, description="Default RPM per tenant")
    burst_size: int = Field(default=10, description="Burst allowance")
    cost_based: bool = Field(
        default=False,
        description="Rate limit by estimated LLM cost (request bytes / 4 ≈ tokens) "
        "instead of request count — the T-DOS cost-abuse control. In cost mode the "
        "per-window limit is cost_units_per_minute and burst_size does not apply "
        "(a large request consumes its own cost immediately).",
    )
    cost_units_per_minute: int = Field(
        default=100_000,
        ge=1,
        description="Cost-based mode: per-tenant cost budget per 60s window "
        "(1 unit ≈ 4 bytes of request body ≈ 1 token estimate). A request whose "
        "cost exceeds the remaining budget is rejected (429) — fail-closed.",
    )


class AgentGuardianSettings(BaseSettings):
    """Agent Guardian — multi-turn detection (Phase 3, Sprint B).

    A stateful scanner that keeps a bounded per-session sliding window of
    turns and detects cross-turn attacks a single-turn scanner cannot see:
    delayed / garden-path injection, role drift / persona erosion, and
    accumulation attacks. Deterministic + heuristic (patterns + state),
    optionally augmented by the existing judge. No LLM call in B1.

    Backend selection:
    - ``memory``: per-process sliding window. Correct for single-worker
      deploys. With ``workers > 1`` each worker keeps its own window, so a
      session that lands on different workers is not correlated — the
      production lifespan refuses to start with ``backend=memory`` in
      multi-worker production (mirrors the rate limiter rule).
    - ``redis``: shared per-session signal store backed by Redis (F4; one
      key per session, atomic Lua record, reuses the P1-1 pattern). Same
      scanner semantics across workers. Requires the ``redis`` package and a
      reachable Redis at ``redis_url``. Stores ONLY per-turn signal flags,
      never raw turn text.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_AGENT_GUARDIAN_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable the Agent Guardian scanner")
    session_window_turns: int = Field(
        default=10,
        description="Max turns retained per session in the sliding window.",
    )
    backend: Literal["memory", "redis"] = Field(
        default="memory",
        description="State backend. 'redis' is required for multi-worker production.",
    )
    redis_url: str | None = Field(
        default=None,
        description="Redis URL for the redis backend. Required when backend=redis.",
    )
    session_ttl_seconds: int = Field(
        default=1800,
        description="Redis backend: per-session inactivity TTL in seconds. A "
        "session that stays silent longer than this loses its accumulated "
        "window (bounds redis memory; re-armed on every recorded turn).",
    )
    # Detection thresholds (per session window)
    role_drift_threshold: int = Field(
        default=2,
        description="Min persona-redefinition signals across the window to flag role drift.",
    )
    extraction_probe_threshold: int = Field(
        default=3,
        description="Min system-prompt-extraction probes across the window to flag accumulation.",
    )
    memory_injection_threshold: int = Field(
        default=2,
        description="Min persistent-memory-injection directives across the window to flag accumulation.",
    )
    max_sessions: int = Field(
        default=10_000,
        description="Max sessions tracked in-memory before LRU eviction. Bounds memory.",
    )


class CanarySettings(BaseSettings):
    """Canary token verification (Phase 3, Sprint B, B3).

    Per-session system-prompt exfiltration canaries. When enabled, the
    operator mints a canary token via ``POST /v1/canary/mint`` (or the
    ``neuralguard canary-mint`` CLI), injects it into the LLM system prompt,
    and ``POST /v1/scan/output`` flags the output if the token appears in it
    (system-prompt exfiltration signal). Derivation is deterministic
    (HMAC-SHA256 of ``session_id|label`` keyed by the server secret) so no
    server-side token storage is needed.

    The secret MUST be set when enabled. In production the lifespan refuses
    to start unless the secret is at least 32 characters — a short/empty
    secret makes the canary trivially guessable and defeats the purpose.
    The secret is never logged or echoed back by the mint endpoint.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_CANARY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable canary token mint + leak detection")
    secret: str = Field(
        default="",
        description=(
            "Server secret for HMAC canary derivation. REQUIRED when enabled; "
            ">= 32 chars in production (startup refuses otherwise). NEVER log "
            "or expose this value. Rotating it invalidates all outstanding canaries."
        ),
    )
    token_count: int = Field(
        default=1,
        description="Default canaries minted per session (1-8). More canaries = more positions to detect partial exfiltration.",
    )

    @field_validator("token_count")
    @classmethod
    def validate_token_count(cls, v: int) -> int:
        if not (1 <= v <= 8):
            raise ValueError(f"token_count must be 1-8, got {v}")
        return v


class ProxySettings(BaseSettings):
    """Standalone appliance proxy configuration (F9).

    OFF by default. Enabling turns NeuralGuard into a transparent guardian:
    ``POST /v1/proxy/chat/completions`` accepts an OpenAI-format chat payload,
    evaluates the user turns, forwards ALLOWed requests to the upstream, and
    scans the completion before delivery. ``upstream_api_key`` is the
    OPERATOR's upstream credential — server-side, never logged. Callers
    authenticate to NeuralGuard with NeuralGuard API keys (tenant-bound).
    """

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_PROXY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(
        default=False,
        description="Enable the standalone appliance proxy routes. OFF by default.",
    )
    upstream_url: str = Field(
        default="",
        description="OpenAI-compatible upstream base URL (e.g. http://localhost:11434/v1 "
        "for Ollama, https://api.example.com/v1 for a cloud). Required when enabled.",
    )
    upstream_api_key: str = Field(
        default="",
        description="The operator's upstream API key (server-side secret; never logged). "
        "Omit for keyless local upstreams (Ollama).",
    )
    timeout_seconds: float = Field(
        default=120.0,
        description="Upstream HTTP timeout for one forwarded chat completion.",
    )

    @property
    def is_configured(self) -> bool:
        """Enabled AND pointed at an upstream."""
        return self.enabled and bool(self.upstream_url.strip())


class NeuralGuardConfig(BaseSettings):
    """Top-level configuration aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="NeuralGuard", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    server: ServerSettings = Field(default_factory=ServerSettings)
    scanner: ScannerSettings = Field(default_factory=ScannerSettings)
    action: ActionSettings = Field(default_factory=ActionSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    tenant: TenantSettings = Field(default_factory=TenantSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    agent_guardian: AgentGuardianSettings = Field(default_factory=AgentGuardianSettings)
    canary: CanarySettings = Field(default_factory=CanarySettings)
    proxy: ProxySettings = Field(default_factory=ProxySettings)
    siem: SiemSettings = Field(default_factory=SiemSettings)


def load_config(config_path: Path | None = None) -> NeuralGuardConfig:
    """Load configuration from environment, .env file, and optional YAML override."""
    # Future: support YAML overlay via config_path
    _ = config_path
    return NeuralGuardConfig()


# ── Unknown-key detection (F5) ────────────────────────────────────────────
# F5 root cause: NEURALGUARD_SERVER_* was used by every config surface while
# the real names are NEURALGUARD_* (env_prefix NEURALGUARD_ + bare field
# names) — and pydantic-settings' extra="ignore" made the dead names SILENT.
# This detector turns that failure class loud: any NEURALGUARD_* key that
# maps to no settings field is reported (production refuses to start).


def _iter_settings_classes() -> Iterator[tuple[str, type[BaseSettings]]]:
    """Yield (env_prefix, cls) for every settings class in this module."""
    for obj in vars(sys.modules[__name__]).values():
        if isinstance(obj, type) and issubclass(obj, BaseSettings) and obj is not BaseSettings:
            yield str(obj.model_config.get("env_prefix", "")), obj


def known_env_keys() -> set[str]:
    """Every NEURALGUARD_* env key the settings classes actually read (uppercase)."""
    keys: set[str] = set()
    for prefix, cls in _iter_settings_classes():
        for name in cls.model_fields:
            keys.add(f"{prefix}{name}".upper())
    return keys


def _iter_candidate_env_keys() -> Iterator[str]:
    """NEURALGUARD_* keys from the real environment AND the repo-root .env file."""
    for key in os.environ:
        if key.upper().startswith("NEURALGUARD_"):
            yield key
    env_file = Path(".env")
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key = line.split("=", 1)[0].strip()
                if key.upper().startswith("NEURALGUARD_"):
                    yield key
        except OSError:  # pragma: no cover - unreadable file -> env-only check
            return


def unknown_env_keys() -> list[str]:
    """NEURALGUARD_* keys present (env or .env) that map to NO settings field.

    These are silent no-ops today — typos, stale names (NEURALGUARD_SERVER_*),
    or removed knobs. Sorted for deterministic logs/tests.
    """
    known = known_env_keys()
    return sorted({key.upper() for key in _iter_candidate_env_keys()} - known)
