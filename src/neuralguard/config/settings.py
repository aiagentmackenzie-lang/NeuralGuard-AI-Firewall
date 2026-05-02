"""NeuralGuard configuration — environment and file-based settings.

Uses pydantic-settings for layered configuration:
  1. Environment variables (NEURALGUARD_ prefix)
  2. .env file
  3. config.yaml (optional)
  4. Defaults
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be 1-65535, got {v}")
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
    """Rate limiting configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NEURALGUARD_RATELIMIT_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable rate limiting")
    requests_per_minute: int = Field(default=60, description="Default RPM per tenant")
    burst_size: int = Field(default=10, description="Burst allowance")
    cost_based: bool = Field(
        default=False, description="Rate limit by estimated LLM cost, not request count"
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
    tenant: TenantSettings = Field(default_factory=TenantSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)


def load_config(config_path: Path | None = None) -> NeuralGuardConfig:
    """Load configuration from environment, .env file, and optional YAML override."""
    # Future: support YAML overlay via config_path
    _ = config_path
    return NeuralGuardConfig()
