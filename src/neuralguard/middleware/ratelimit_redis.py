"""Redis-backed sliding-window rate limiter.

Shared across uvicorn workers via a Redis ZSET + an atomic Lua script. This
is the correct backend for multi-worker production deploys — the in-memory
:class:`SlidingWindowCounter` is per-process and allows
``(limit + burst) * workers`` requests per window.

Design:
- One ZSET per rate-limit key (``rl:<tenant>``), members are unique request
  IDs scored by the request timestamp (seconds, float).
- ``ZREMRANGEBYSCORE`` evicts entries older than the window before counting.
- The whole check (evict -> count -> maybe add -> expire) runs as a single
  Lua script so it is atomic across workers — no TOCTOU window between the
  count and the insert.
- Rejected requests are NOT recorded, so a flood of rejected calls does not
  extend a tenant's window.

The limiter accepts an injected ``redis.asyncio.Redis``-compatible client so
tests can pass a ``fakeredis`` instance; in production the middleware
constructs the client from ``RateLimitSettings.redis_url``.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from neuralguard.config.settings import RateLimitSettings

logger = structlog.get_logger(__name__)


# Atomic sliding-window check. Returns [allowed(0/1), remaining, retry_after].
# KEYS[1] = rate-limit key
# ARGV[1] = now (seconds, float-as-string)
# ARGV[2] = window seconds
# ARGV[3] = limit
# ARGV[4] = burst
# ARGV[5] = unique member id
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local burst = tonumber(ARGV[4])
local member = ARGV[5]

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)

if count >= limit + burst then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry = window
    if oldest[2] ~= nil then
        retry = math.ceil(window - (now - tonumber(oldest[2])))
    end
    if retry < 1 then retry = 1 end
    return {0, 0, retry}
end

redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, math.floor(window * 1000) + 1000)
local remaining = (limit + burst) - count - 1
if remaining < 0 then remaining = 0 end
return {1, remaining, 0}
"""


class RedisRateLimiter:
    """Shared sliding-window rate limiter backed by Redis.

    Args:
        settings: Rate-limit settings. Uses ``requests_per_minute`` as the
            window limit and ``burst_size`` as the burst allowance. The window
            is fixed at 60 seconds (matches the in-memory counter).
        client: An async Redis client (``redis.asyncio.Redis`` or a
            ``fakeredis.aioredis.FakeRedis`` for tests). If ``None``, a client
            is built from ``settings.redis_url``.
    """

    def __init__(self, settings: RateLimitSettings, client: Redis | None = None) -> None:
        self.settings = settings
        self._window = 60
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            if not settings.redis_url:
                raise ValueError(
                    "RateLimitSettings.redis_url is required when backend=redis "
                    "and no client is injected."
                )
            from redis.asyncio import Redis as _Redis

            self._client = _Redis.from_url(settings.redis_url, decode_responses=False)
            self._owns_client = True
        # Register the script once; it loads on first call and is cached by SHA.
        self._script = self._client.register_script(_SLIDING_WINDOW_LUA)

    async def check(self, key: str, limit: int, burst: int) -> tuple[bool, int, int]:
        """Check and record a request against the sliding window.

        Returns ``(allowed, remaining, retry_after_seconds)`` — same contract
        as ``SlidingWindowCounter.check``.
        """
        now = time.time()
        member = f"{now}:{uuid.uuid4().hex}"
        try:
            result = await self._script(
                keys=[key],
                args=[now, self._window, limit, burst, member],
            )
        except Exception as exc:
            # Fail-closed: if Redis is unreachable we cannot enforce a shared
            # limit, so we reject rather than let a tenant exceed the budget.
            logger.error("redis_ratelimit_failed", key=key, error=repr(exc))
            return False, 0, 1

        # redis-py returns a list of ints (Lua numbers come back as integers).
        try:
            allowed_int, remaining, retry_after = (int(result[0]), int(result[1]), int(result[2]))
        except (TypeError, IndexError, ValueError) as exc:
            logger.error(
                "redis_ratelimit_bad_response", key=key, response=repr(result), error=repr(exc)
            )
            return False, 0, 1

        return bool(allowed_int), remaining, retry_after

    async def aclose(self) -> None:
        """Release the Redis connection if this limiter owns it."""
        if self._owns_client:
            import contextlib

            with contextlib.suppress(Exception):
                await self._client.aclose()

    # Convenience for tests / introspection.
    @property
    def client(self) -> Any:
        return self._client
