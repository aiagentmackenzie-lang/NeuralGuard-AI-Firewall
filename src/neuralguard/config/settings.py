"""NeuralGuard configuration — environment and file-based settings.

Uses pydantic-settings for layered configuration:
  1. Environment variables (NEURALGUARD_ prefix)
  2. .env file
  3. config.yaml (optional)
  4. Defaults
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
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
        description="Allow serving plain HTTP in production. MUST be set explicitly; otherwise "
        "production startup fails. Intended only behind a TLS-terminating reverse proxy.",
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
    judge_model: str = Field(default="gpt-4o-mini", description="Judge model identifier")
    judge_max_tokens: int = Field(default=512, description="Max tokens for judge response")
    judge_temperature: float = Field(default=0.0, description="Judge sampling temperature")
    judge_ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama base URL for the judge. In production this MUST resolve to a "
        "loopback/private address to keep prompts inside the trust boundary.",
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
        default=False, description="Rate limit by estimated LLM cost, not request count"
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
    - ``redis``: shared sliding window backed by Redis (reuses the P1-1
      pattern). Correct across workers. Requires the ``[redis]`` extra and a
      reachable Redis at ``redis_url``. (B1 ships the in-memory backend; the
      Redis backend is a B1+ follow-up — the interface is designed for it.)
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


def load_config(config_path: Path | None = None) -> NeuralGuardConfig:
    """Load configuration from environment, .env file, and optional YAML override."""
    # Future: support YAML overlay via config_path
    _ = config_path
    return NeuralGuardConfig()
