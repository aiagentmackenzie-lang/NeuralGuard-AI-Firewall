"""Rate limiting middleware — sliding window per tenant.

Uses in-memory sliding window counters (per-worker).
For multi-worker deployment, configure Redis backend (Phase 2+).

Supports:
- Requests-per-minute per tenant
- Burst allowance
- Cost-based limiting (future: token-count weighted)
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from starlette.requests import Request

    from neuralguard.config.settings import RateLimitSettings

logger = structlog.get_logger(__name__)


class SlidingWindowCounter:
    """In-memory sliding window rate limiter."""

    def __init__(self, window_seconds: int = 60) -> None:
        self._window = window_seconds
        self._counters: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str, limit: int, burst: int) -> tuple[bool, int, int]:
        """Check if request is within limits.

        Returns: (allowed, remaining, retry_after_seconds)
        """
        now = time.time()
        # Clean old entries
        self._counters[key] = [ts for ts in self._counters[key] if now - ts < self._window]

        current = len(self._counters[key])

        # Burst check
        if current >= limit + burst:
            oldest = self._counters[key][0] if self._counters[key] else now
            retry_after = int(self._window - (now - oldest)) + 1
            return False, 0, max(retry_after, 1)

        # Rate check
        if current >= limit:
            oldest = self._counters[key][0] if self._counters[key] else now
            retry_after = int(self._window - (now - oldest)) + 1
            return False, 0, max(retry_after, 1)

        # Allow
        self._counters[key].append(now)
        remaining = limit - current - 1
        return True, remaining, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware for per-tenant rate limiting."""

    def __init__(
        self,
        app: Any,
        settings: RateLimitSettings,
    ) -> None:
        super().__init__(app)
        self.settings = settings
        self._counter = SlidingWindowCounter(window_seconds=60)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if not self.settings.enabled:
            return await call_next(request)

        # Skip non-API paths
        if not request.url.path.startswith("/v1/"):
            return await call_next(request)

        # Extract tenant ID from request body or header
        tenant_id = request.headers.get("X-Tenant-ID", "default")

        # Use tenant-specific limits (future: per-tenant config)
        rpm = self.settings.requests_per_minute
        burst = self.settings.burst_size

        allowed, remaining, retry_after = self._counter.check(
            key=f"rl:{tenant_id}",
            limit=rpm,
            burst=burst,
        )

        if not allowed:
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

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
