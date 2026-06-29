"""Per-tenant override model (Sprint C, C1 / ROADMAP P1-2).

A ``TenantConfig`` is the per-tenant overlay applied on top of the global
``NeuralGuardConfig``. Every field is ``None``-means-inherit-global so a
partial tenant file can never produce an unsafe zero. Structural + Pattern
are non-overridable (always on) and therefore not modelled here.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Tenant id rules: lowercase ascii + digits + ``-``/``_``/``.``, 1..64 chars.
# Matches the constraint on ``EvaluateRequest.tenant_id`` in schemas.py.
_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class TenantScannerOverrides(BaseModel):
    """Per-tenant enable/disable for the three opt-in scanner layers.

    Each field is tri-state:
    - ``None`` (default) -> inherit the global enable state.
    - ``True``  -> enable for this tenant (only effective if the scanner is
      globally registered; an unregistered scanner cannot be conjured).
    - ``False`` -> disable for this tenant (the tenant narrows its own
      defense surface by explicit choice; logged for audit).

    Structural and Pattern are NOT here — they are mandatory and cannot be
    disabled per-tenant. Attempting to model them would imply they are
    optional, which they are not.
    """

    model_config = ConfigDict(extra="forbid")

    agent_guardian: bool | None = Field(
        default=None,
        description="Override Agent Guardian enable. None=inherit global.",
    )
    semantic: bool | None = Field(
        default=None,
        description="Override Semantic scanner enable. None=inherit global.",
    )
    judge: bool | None = Field(
        default=None,
        description="Override LLM-as-Judge enable. None=inherit global.",
    )


class TenantConfig(BaseModel):
    """A single tenant's override overlay.

    The ``tenant_id`` MUST match the filename stem (minus extension) so the
    registry can bind a file to its tenant unambiguously. The registry
    rejects a file whose declared ``tenant_id`` disagrees with the stem —
    this prevents a stale copy of ``acme.yaml`` masquerading as ``globex``.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(
        ...,
        description="Tenant id. MUST equal the config-file stem (e.g. 'acme' for acme.yaml).",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable tenant label (metadata only, never used for routing).",
    )
    requests_per_minute: int | None = Field(
        default=None,
        ge=1,
        description="Per-tenant RPM override. None=inherit global RateLimitSettings.",
    )
    burst_size: int | None = Field(
        default=None,
        ge=0,
        description="Per-tenant burst override. None=inherit global RateLimitSettings.",
    )
    scanners: TenantScannerOverrides = Field(
        default_factory=TenantScannerOverrides,
        description="Per-tenant scanner enable/disable overlay (optional layers only).",
    )

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, v: str) -> str:
        if not _TENANT_ID_RE.match(v):
            raise ValueError(
                "tenant_id must be lowercase ascii + digits/./-/_ , 1..64 chars, "
                "starting alphanumeric."
            )
        return v

    def effective_rate_limit(
        self,
        global_rpm: int,
        global_burst: int,
    ) -> tuple[int, int]:
        """Resolve (rpm, burst) for this tenant, inheriting global on None."""
        return (
            self.requests_per_minute if self.requests_per_minute is not None else global_rpm,
            self.burst_size if self.burst_size is not None else global_burst,
        )

    def to_effective_dict(self) -> dict[str, Any]:
        """Public, secret-free view used by the read-only tenants API/CLI.

        ``description`` is included (metadata); no secrets are ever stored
        on a TenantConfig, so this is safe to expose.
        """
        return {
            "tenant_id": self.tenant_id,
            "description": self.description,
            "requests_per_minute": self.requests_per_minute,
            "burst_size": self.burst_size,
            "scanners": self.scanners.model_dump(),
        }
