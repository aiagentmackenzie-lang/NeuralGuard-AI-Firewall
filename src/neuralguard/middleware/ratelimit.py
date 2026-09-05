"""Rate limiting middleware — sliding window per tenant.

Uses in-memory sliding window counters (per-worker).
For multi-worker deployment, configure Redis backend (Phase 2+).

Supports:
- Requests-per-minute per tenant
- Burst allowance
- Cost-based limiting (F7): each request is charged ~tokens (bytes/4)
  against a per-window cost budget — the T-DOS cost-abuse control
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from neuralguard.metrics import metrics

if TYPE_CHECKING:
    from starlette.requests import Request

    from neuralguard.config.settings import RateLimitSettings
    from neuralguard.middleware.ratelimit_redis import RedisRateLimiter
    from neuralguard.tenants.registry import TenantConfigRegistry

logger = structlog.get_logger(__name__)


class SlidingWindowCounter:
    """In-memory sliding window rate limiter.

    Note: This counter is per-process. With multiple uvicorn workers,
    each worker maintains its own counter, so a tenant could make
    up to (limit + burst) x workers requests per minute. For
    multi-worker deployments, use a Redis-backed rate limiter instead.
    """

    def __init__(self, window_seconds: int = 60) -> None:
        self._window = window_seconds
        self._counters: dict[str, list[float]] = defaultdict(list)
        # Cost-based mode store (F7): parallel to _counters but entries carry
        # (timestamp, cost) — request-count mode never touches this.
        self._cost_entries: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self._last_cleanup: float = time.time()
        self._cleanup_interval: float = 300.0  # Cleanup every 5 minutes

    def _cleanup_inactive(self) -> None:
        """Remove counters for tenants with no activity in the last window."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        inactive_keys = [
            key
            for key, timestamps in self._counters.items()
            if not timestamps or now - timestamps[-1] > self._window
        ]
        inactive_keys = [
            key
            for key, timestamps in self._counters.items()
            if not timestamps or now - timestamps[-1] > self._window
        ]
        for key in inactive_keys:
            del self._counters[key]
        inactive_cost_keys = [
            key
            for key, entries in self._cost_entries.items()
            if not entries or now - entries[-1][0] > self._window
        ]
        for key in inactive_cost_keys:
            del self._cost_entries[key]

    def check(self, key: str, limit: int, burst: int) -> tuple[bool, int, int]:
        """Check if request is within limits.

        Logic:
        - Allow up to `limit` requests per window (normal rate)
        - Allow up to `limit + burst` total (burst allowance)
        - Beyond `limit + burst`, reject with retry_after
        - Between `limit` and `limit + burst`, allow but track burst usage

        Returns: (allowed, remaining, retry_after_seconds)
        """
        now = time.time()
        # Clean old entries outside the window
        self._counters[key] = [ts for ts in self._counters[key] if now - ts < self._window]

        # Periodic cleanup of inactive tenants
        self._cleanup_inactive()

        current = len(self._counters[key])

        # Hard limit: beyond burst allowance
        if current >= limit + burst:
            oldest = self._counters[key][0] if self._counters[key] else now
            retry_after = int(self._window - (now - oldest)) + 1
            return False, 0, max(retry_after, 1)

        # Within limits — allow and record
        self._counters[key].append(now)
        remaining = max(0, (limit + burst) - current - 1)
        return True, remaining, 0

    def check_cost(self, key: str, cost: int, limit: int) -> tuple[bool, int, int]:
        """Cost-based variant (F7): charge a token-estimate, not a request.

        Each request contributes ``cost`` units (bytes/4 ≈ tokens) against a
        per-window COST budget. ``burst_size`` does not apply in cost mode —
        a large request consumes its own cost immediately, which IS the burst
        behavior a cost budget wants. A request whose cost exceeds the
        remaining budget is rejected (fail-closed: oversized payloads cannot
        ride through a nearly-empty window).

        Returns: (allowed, remaining_cost_units, retry_after_seconds)
        """
        now = time.time()
        entries = self._cost_entries[key]
        entries[:] = [(ts, c) for (ts, c) in entries if now - ts < self._window]

        # Periodic cleanup of inactive tenants
        self._cleanup_inactive()

        used = sum(c for _, c in entries)
        if used + cost > limit:
            oldest = entries[0][0] if entries else now
            retry_after = int(self._window - (now - oldest)) + 1
            return False, max(0, limit - used), max(retry_after, 1)

        entries.append((now, cost))
        return True, max(0, limit - used - cost), 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware for per-tenant rate limiting.

    Backend is selected by ``RateLimitSettings.backend``:
    - ``memory``: per-process sliding window (single-worker only in prod).
    - ``redis``: shared sliding window across workers (requires ``[redis]``).

    A ``redis_client`` may be injected for testing (e.g. a ``fakeredis``
    instance); otherwise the Redis limiter builds its own client from
    ``settings.redis_url``.
    """

    def __init__(
        self,
        app: Any,
        settings: RateLimitSettings,
        redis_client: Any | None = None,
        tenant_registry: TenantConfigRegistry | None = None,
    ) -> None:
        super().__init__(app)
        self.settings = settings
        self._backend = settings.backend
        self._counter: SlidingWindowCounter | None = None
        self._redis: RedisRateLimiter | None = None
        # Per-tenant override registry (Sprint C, C1). None when multi-tenant
        # mode is disabled -> the middleware uses the global RPM/burst for
        # every tenant (backward compatible).
        self._tenant_registry = tenant_registry
        if settings.backend == "redis":
            if not settings.redis_url and redis_client is None:
                raise ValueError(
                    "Rate-limit backend=redis requires redis_url or an injected client."
                )
            from neuralguard.middleware.ratelimit_redis import RedisRateLimiter

            self._redis = RedisRateLimiter(settings, client=redis_client)
            logger.info("ratelimit_backend", backend="redis")
        else:
            self._counter = SlidingWindowCounter(window_seconds=60)
            logger.info("ratelimit_backend", backend="memory")

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if not self.settings.enabled:
            return cast("Response", await call_next(request))

        # Skip non-API paths
        if not request.url.path.startswith("/v1/"):
            return cast("Response", await call_next(request))

        # Tenant identity: when auth is enabled, the authenticated principal
        # (set by AuthMiddleware on request.state.auth_tenant) is authoritative.
        # Falling back to the client-supplied X-Tenant-ID header is ONLY acceptable
        # when auth is disabled (development). This prevents the rate-limit
        # tenant-spoofing bypass.
        auth_tenant = getattr(request.state, "auth_tenant", None)
        tenant_id = auth_tenant or request.headers.get("X-Tenant-ID", "default")

        # Tenant-specific limits (request-count mode only — in cost mode the
        # limit is the global cost budget; per-tenant cost budgets are a P2
        # follow-up).
        rpm = self.settings.requests_per_minute
        burst = self.settings.burst_size
        limit_header = str(rpm)

        registry = self._tenant_registry
        cost_units: int | None = None
        if self.settings.cost_based:
            # F7 cost-based mode: charge ~tokens (bytes/4) against the per-tenant
            # cost budget instead of counting requests.
            body = await request.body()
            cost_units = max(1, len(body) // 4)
            limit_header = str(self.settings.cost_units_per_minute)
        elif registry is not None and registry.enabled:
            # Resolve per-tenant overrides; None/unknown tenant -> global default
            # (fail-open: a config miss never denies a request). The registry is
            # designed to never raise, but defend-in-depth: if it ever does, fall
            # back to the global limits and log — never crash the request path
            # with an unhandled exception (BaseHTTPMiddleware would bypass the
            # global handler and leak a raw 500).
            try:
                rpm, burst = registry.effective_rate_limit(tenant_id, rpm, burst)
            except (
                Exception
            ) as exc:  # defense-in-depth (covered by TestTenantEndpointExceptionHygiene)
                logger.warning(
                    "tenant_ratelimit_resolve_failed",
                    tenant=tenant_id,
                    error=repr(exc),
                    msg="falling back to global rate-limit defaults",
                )

        allowed: bool
        remaining: int
        retry_after: int
        if cost_units is not None:
            if self._redis is not None:
                allowed, remaining, retry_after = await self._redis.check_cost(
                    key=f"rl:{tenant_id}",
                    cost=cost_units,
                    budget=self.settings.cost_units_per_minute,
                )
            else:
                assert self._counter is not None
                allowed, remaining, retry_after = self._counter.check_cost(
                    key=f"rl:{tenant_id}",
                    cost=cost_units,
                    limit=self.settings.cost_units_per_minute,
                )
        elif self._redis is not None:
            allowed, remaining, retry_after = await self._redis.check(
                key=f"rl:{tenant_id}",
                limit=rpm,
                burst=burst,
            )
        else:
            assert self._counter is not None
            allowed, remaining, retry_after = self._counter.check(
                key=f"rl:{tenant_id}",
                limit=rpm,
                burst=burst,
            )

        if not allowed:
            metrics.record_rate_limit_hit()
            logger.warning(
                "rate_limit_exceeded",
                tenant=tenant_id,
                path=request.url.path,
                retry_after=retry_after,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Retry after {retry_after} seconds.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        response = cast("Response", await call_next(request))
        response.headers["X-RateLimit-Limit"] = limit_header
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
