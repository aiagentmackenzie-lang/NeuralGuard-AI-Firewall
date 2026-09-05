"""Readiness check — `/v1/ready`.

Distinct from liveness (`/v1/health`, "is the process up"): readiness answers
"can this worker safely accept traffic?". It probes the components the worker
depends on and reports a per-component status plus an aggregate.

Aggregation rules (deliberate for a security firewall):

- **Core** = at least the structural + pattern scanners registered, and any
  *required* shared backend reachable. If the core is broken → 503
  ``unhealthy`` (remove the worker from the load balancer).
- **Optional** layers (semantic, judge, postgres audit) degrade gracefully:
  the firewall still serves with deterministic detection, audit falls back to
  JSONL, the judge circuit-breaks. A degraded optional component → 200
  ``degraded`` (the worker stays in rotation).
- **Redis** when ``rate_limit.backend=redis`` is *required*: the limiter fails
  closed (every request 429) if Redis is unreachable, so a down Redis makes
  the worker unusable → 503 ``unhealthy``.

Component status values: ``ok`` | ``degraded`` | ``fail`` | ``skip``
(``skip`` = feature not enabled, not a defect).

The endpoint is auth-protected by default (not in ``public_endpoints``) so
component connectivity is not disclosed to unauthenticated callers. An
operator who needs the kubelet to probe it without credentials adds
``/v1/ready`` to ``NEURALGUARD_AUTH_PUBLIC_ENDPOINTS``; the response body
contains no secrets or URLs, only coarse status strings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from fastapi import Request

    from neuralguard.config.settings import NeuralGuardConfig
    from neuralguard.scanners.pipeline import ScannerPipeline

logger = structlog.get_logger(__name__)

# Per-component probe budget. Readiness is probed often (k8s default ~10s);
# keep each probe cheap so the endpoint stays sub-100ms.
_PROBE_TIMEOUT_SECONDS = 1.5

ComponentStatus = str  # "ok" | "degraded" | "fail" | "skip"


@dataclass
class ReadinessResult:
    """Aggregate readiness report."""

    ready: bool
    status: str  # "healthy" | "degraded" | "unhealthy"
    components: dict[str, ComponentStatus] = field(default_factory=dict)
    uptime_seconds: float = 0.0
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "status": self.status,
            "components": self.components,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


async def _probe_db(config: NeuralGuardConfig) -> ComponentStatus:
    """Check the Postgres audit backend connectivity."""
    if config.audit.backend != "postgres" or not config.audit.postgres_url:
        return "skip"
    try:
        from neuralguard.db.engine import get_engine

        engine = get_engine()
    except ImportError:
        return "degraded"  # db extra not installed; JSONL fallback active
    if engine is None:
        # Engine is created in the lifespan; if it's not ready yet the audit
        # logger falls back to JSONL, so the worker can still serve.
        return "degraded"
    try:
        from sqlalchemy import text

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:
        logger.warning("readiness_db_fail", error=repr(exc))
        return "degraded"  # JSONL fallback keeps audit integrity


async def _probe_redis(app_state: Any, config: NeuralGuardConfig) -> ComponentStatus:
    """Check the Redis rate-limiter backend."""
    if not config.rate_limit.enabled or config.rate_limit.backend != "redis":
        return "skip"
    limiter = getattr(app_state, "redis_limiter", None)
    client = getattr(limiter, "client", None) if limiter is not None else None
    if client is None:
        return "fail"  # configured but not initialized — core broken
    try:
        if await client.ping() is True:
            return "ok"
        # Some clients return a bool-like; treat anything truthy as ok.
        return "ok" if await client.ping() else "fail"
    except Exception as exc:
        logger.warning("readiness_redis_fail", error=repr(exc))
        return "fail"  # fail-closed limiter → worker unusable


def _probe_scanners(
    pipeline: ScannerPipeline, config: NeuralGuardConfig
) -> dict[str, ComponentStatus]:
    """Check scanner registration and (for semantic) initialization."""
    out: dict[str, ComponentStatus] = {}
    registered = set(pipeline._scanners.keys())

    # Core deterministic layers — must be registered.
    out["structural"] = "ok" if any(l.value == "structural" for l in registered) else "fail"
    out["pattern"] = "ok" if any(l.value == "pattern" for l in registered) else "fail"

    # Optional semantic layer.
    if config.scanner.semantic_enabled:
        sem = pipeline._scanners.get(_layer_value("semantic"))
        if sem is None:
            out["semantic"] = "fail"
        else:
            initialized = getattr(sem, "initialized", True)
            out["semantic"] = "ok" if initialized else "degraded"
    else:
        out["semantic"] = "skip"

    # Optional judge layer. We do NOT ping Ollama on every readiness call —
    # the judge circuit breaker handles upstream liveness at runtime and a
    # ping would add latency + a network hop to a frequently-hit endpoint.
    if config.scanner.judge_enabled:
        out["judge"] = "ok" if any(l.value == "judge" for l in registered) else "fail"
        # F10.3: surface whether judged prompts leave the trust boundary.
        from neuralguard.main import _is_private_judge_url

        out["judge_egress"] = (
            "local"
            if _is_private_judge_url(config.scanner.judge_ollama_url)
            else (
                "explicit-egress" if config.scanner.judge_allow_egress else "egress-refused-in-prod"
            )
        )
    else:
        out["judge"] = "skip"

    return out


def _layer_value(name: str) -> Any:
    from neuralguard.models.schemas import ScanLayer

    return ScanLayer(name)


async def check_readiness(request: Request) -> ReadinessResult:
    """Compute the aggregate readiness for this worker."""
    config: NeuralGuardConfig = request.app.state.config
    pipeline: ScannerPipeline = request.app.state.pipeline

    components: dict[str, ComponentStatus] = {}
    components.update(_probe_scanners(pipeline, config))
    components["audit_db"] = await _probe_db(config)
    components["redis"] = await _probe_redis(request.app.state, config)

    # Aggregate.
    core_ok = components["structural"] == "ok" and components["pattern"] == "ok"
    redis_ok = components["redis"] in ("ok", "skip")
    any_degraded = any(v == "degraded" for v in components.values())
    any_fail_optional = any(
        v == "fail" and k not in {"structural", "pattern", "redis"} for k, v in components.items()
    )

    start_time = getattr(request.app.state, "start_time", None)
    uptime = time.time() - start_time if start_time else 0.0

    if not core_ok or not redis_ok:
        return ReadinessResult(
            ready=False, status="unhealthy", components=components, uptime_seconds=uptime
        )
    if any_degraded or any_fail_optional:
        return ReadinessResult(
            ready=True, status="degraded", components=components, uptime_seconds=uptime
        )
    return ReadinessResult(
        ready=True, status="healthy", components=components, uptime_seconds=uptime
    )
