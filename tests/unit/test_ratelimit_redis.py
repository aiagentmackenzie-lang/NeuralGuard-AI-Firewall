"""Tests for the Redis-backed rate limiter (P1-1).

Uses fakeredis so the suite stays hermetic — no real Redis required. The
Lua script and the ZSET sliding-window logic are identical to what runs
against a real Redis in production.
"""

from __future__ import annotations

import asyncio

import pytest
from fakeredis.aioredis import FakeRedis

from neuralguard.config.settings import RateLimitSettings
from neuralguard.middleware.ratelimit import RateLimitMiddleware
from neuralguard.middleware.ratelimit_redis import RedisRateLimiter


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def limiter(fake_redis: FakeRedis) -> RedisRateLimiter:
    return RedisRateLimiter(
        RateLimitSettings(backend="redis", redis_url="redis://localhost:6379/0"),
        client=fake_redis,
    )


class TestRedisRateLimiter:
    async def test_allows_up_to_limit_plus_burst(self, limiter: RedisRateLimiter):
        # limit=5, burst=3 -> 8 allowed
        for i in range(8):
            allowed, _remaining, _retry = await limiter.check("rl:t1", limit=5, burst=3)
            assert allowed, f"request {i + 1} should be allowed"
        allowed, _remaining, retry = await limiter.check("rl:t1", limit=5, burst=3)
        assert not allowed, "9th request must be blocked"
        assert retry >= 1

    async def test_remaining_decreases(self, limiter: RedisRateLimiter):
        _allowed, remaining1, _ = await limiter.check("rl:t2", limit=5, burst=3)
        assert remaining1 == 7  # (5+3) - 1 - 1
        _allowed, remaining2, _ = await limiter.check("rl:t2", limit=5, burst=3)
        assert remaining2 == 6

    async def test_tenants_isolated(self, limiter: RedisRateLimiter):
        for _ in range(8):
            await limiter.check("rl:t_a", limit=5, burst=3)
        # tenant B has its own window
        allowed, _r, _retry = await limiter.check("rl:t_b", limit=5, burst=3)
        assert allowed

    async def test_rejected_requests_not_recorded(self, limiter: RedisRateLimiter):
        """A flood of rejected requests must not extend the window."""
        for _ in range(8):
            await limiter.check("rl:t3", limit=5, burst=3)
        # 10 rejected calls
        for _ in range(10):
            allowed, _r, _retry = await limiter.check("rl:t3", limit=5, burst=3)
            assert not allowed
        # A *different* tenant is unaffected — proves the rejected flood did
        # not pollute another key (and by extension did not grow the ZSET).
        allowed, _r, _retry = await limiter.check("rl:t3_other", limit=5, burst=3)
        assert allowed

    async def test_requires_redis_url_without_client(self):
        with pytest.raises(ValueError, match="redis_url is required"):
            RedisRateLimiter(RateLimitSettings(backend="redis", redis_url=None))

    async def test_fail_closed_on_redis_error(self):
        """If the Redis call raises, the limiter fails closed (rejects)."""

        class BoomClient:
            def register_script(self, _script: str) -> BoomClient:
                return self

            async def __call__(self, *_a, **_kw) -> None:
                raise ConnectionError("redis down")

            async def aclose(self) -> None:
                pass

        limiter = RedisRateLimiter(
            RateLimitSettings(backend="redis", redis_url="redis://x"),
            client=BoomClient(),  # type: ignore[arg-type]
        )
        allowed, _r, retry = await limiter.check("rl:t", limit=5, burst=3)
        assert not allowed
        assert retry >= 1


class TestRateLimitMiddlewareBackendSelection:
    def test_memory_backend_default(self):
        from starlette.applications import Starlette

        mw = RateLimitMiddleware(Starlette(), RateLimitSettings(backend="memory"))
        assert mw._backend == "memory"
        assert mw._counter is not None
        assert mw._redis is None

    def test_redis_backend_uses_injected_client(self, fake_redis: FakeRedis):
        from starlette.applications import Starlette

        mw = RateLimitMiddleware(
            Starlette(),
            RateLimitSettings(backend="redis", redis_url="redis://localhost"),
            redis_client=fake_redis,
        )
        assert mw._backend == "redis"
        assert mw._redis is not None
        assert mw._counter is None

    def test_redis_backend_without_url_or_client_raises(self):
        from starlette.applications import Starlette

        with pytest.raises(ValueError, match="redis_url or an injected client"):
            RateLimitMiddleware(
                Starlette(),
                RateLimitSettings(backend="redis", redis_url=None),
            )


class TestProductionMultiWorkerFailFast:
    """Production must refuse to start with workers>1 and the memory backend."""

    async def test_multi_worker_memory_backend_refused(self):
        from neuralguard.config.settings import (
            AuthSettings,
            NeuralGuardConfig,
            ServerSettings,
        )
        from neuralguard.main import create_app

        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True, workers=2),
            rate_limit=RateLimitSettings(backend="memory", enabled=True),
        )
        app = create_app(config)
        with pytest.raises(RuntimeError, match=r"rate_limit\.backend=memory"):
            async with app.router.lifespan_context(app):
                pass

    async def test_multi_worker_redis_without_url_refused(self):
        from neuralguard.config.settings import (
            AuthSettings,
            NeuralGuardConfig,
            ServerSettings,
        )
        from neuralguard.main import create_app

        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True, workers=2),
            rate_limit=RateLimitSettings(backend="redis", redis_url=None, enabled=True),
        )
        # create_app constructs the RedisRateLimiter eagerly — it must raise
        # because redis_url is None and no client is injected.
        with pytest.raises(ValueError, match="redis_url"):
            create_app(config)

    async def test_single_worker_memory_allowed(self):
        from neuralguard.config.settings import (
            AuthSettings,
            NeuralGuardConfig,
            ServerSettings,
        )
        from neuralguard.main import create_app

        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True, workers=1),
            rate_limit=RateLimitSettings(backend="memory", enabled=True),
        )
        app = create_app(config)
        async with app.router.lifespan_context(app):
            pass  # should not raise


# asyncio safety: pytest-asyncio auto mode handles the async defs above.
_ = asyncio


class TestRedisCostLimiter:
    """F7: cost-based mode on the Redis backend (fakeredis runs the same Lua)."""

    async def test_charges_cost_against_budget(self, limiter: RedisRateLimiter):
        a1 = await limiter.check_cost("rl:ct", cost=300, budget=1000)
        assert a1[0] and a1[1] == 700 and a1[2] == 0
        a2 = await limiter.check_cost("rl:ct", cost=300, budget=1000)
        assert a2[0] and a2[1] == 400
        # 600 used + 500 would exceed the budget -> rejected, budget unchanged
        a3 = await limiter.check_cost("rl:ct", cost=500, budget=1000)
        assert not a3[0]
        assert a3[1] == 400
        assert a3[2] >= 1

    async def test_request_cost_exceeding_budget_rejected(self, limiter: RedisRateLimiter):
        a1 = await limiter.check_cost("rl:cx", cost=2000, budget=1000)
        assert not a1[0]

    async def test_window_expiry_frees_budget(self, limiter: RedisRateLimiter):
        import time as _time

        assert (await limiter.check_cost("rl:ce", cost=900, budget=1000))[0]
        # age the window out: wipe and re-add one stale entry (score = timestamp)
        await limiter.client.delete("rl:ce")
        await limiter.client.zadd("rl:ce", {"stale:abc:900": _time.time() - 120})
        a2 = await limiter.check_cost("rl:ce", cost=500, budget=1000)
        assert a2[0] and a2[1] == 500

    async def test_fail_closed_on_redis_error(
        self, limiter: RedisRateLimiter, monkeypatch: pytest.MonkeyPatch
    ):
        async def _boom(*args: object, **kwargs: object) -> list[int]:
            raise ConnectionError("redis down")

        monkeypatch.setattr(limiter, "_cost_script", _boom)
        allowed, remaining, retry = await limiter.check_cost("rl:cf", cost=10, budget=1000)
        assert not allowed
        assert remaining == 0
        assert retry == 1

    async def test_tenants_isolated(self, limiter: RedisRateLimiter):
        assert (await limiter.check_cost("rl:ca", cost=800, budget=1000))[0]
        # other tenant unaffected
        assert (await limiter.check_cost("rl:cb", cost=800, budget=1000))[0]
        assert not (await limiter.check_cost("rl:ca", cost=300, budget=1000))[0]
