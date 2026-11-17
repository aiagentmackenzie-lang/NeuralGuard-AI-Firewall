"""Contracts for the shared (Redis) HITL escalation store.

NG-15 shipped a **per-worker, in-memory** broker and said so. That is honest for
one replica and wrong for two: an escalation raised on worker A is invisible to
an approver whose request lands on worker B, so the human cannot decide, the
wait times out, and the call is refused. A control that only works on one
worker is not a control.

Moving the registry to Redis must not weaken any of the properties NG-15 argued
for, and those are what these contracts test — the store is a *security* store:

- **Single use, race-free across workers.** Two workers consuming the same
  approval concurrently must produce exactly one success. A compare-and-set that
  is really read-then-write passes a single-threaded test and loses the property
  under load, so the contract runs the race.
- **Single-shot decisions.** A second decision on the same escalation is an
  error, not an overwrite — otherwise a denial could be flipped to an approval.
- **Expiry is a refusal.** An expired record cannot be approved late; with the
  store holding wall-clock TTLs, "gone" and "too late" mean the same thing.
- **Capacity is a refusal, never an implicit queue.**
- **Approval stays bound to the payload digest.**
- **Store failure is a refusal, never an approval.** If Redis is unreachable the
  gateway must fail closed; no code path may return "approved" from an error.

Runs on ``fakeredis[lua]`` so the atomicity contracts exercise real Lua, not a
Python mock of one.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import fakeredis
import pytest


@pytest.fixture
async def redis() -> Any:
    server = fakeredis.FakeServer()
    client = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
    yield client
    await client.aclose()


def _store(redis: Any, **kwargs: Any) -> Any:
    from neuralguard.mcp.hitl_redis import RedisHitlStore

    store = RedisHitlStore(redis=redis, ttl_seconds=kwargs.pop("ttl_seconds", 300.0), **kwargs)
    return store


async def _raise(store: Any, *, tool: str = "delete_repo", args: Any = None) -> Any:
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool, "arguments": args},
    }
    return await store.raise_escalation(
        tenant="default",
        agent="agent-a",
        method="tools/call",
        tool=tool,
        rule_id="MCP-ARG-LIMIT",
        reason="destructive tool needs approval",
        source="argument_gate",
        payload=payload,
        arguments=args,
    )


class TestCrossWorkerVisibility:
    async def test_an_escalation_raised_on_one_worker_is_decided_on_another(
        self, redis: Any
    ) -> None:
        worker_a = _store(redis)  # request arrives here
        worker_b = _store(redis)  # approver's request lands here

        record = await _raise(worker_a)

        listed = await worker_b.pending_for_approval()
        assert [r.escalation_id for r in listed] == [record.escalation_id], (
            "the approver must see an escalation raised on a different worker — this is "
            "the whole defect the shared store exists to close"
        )

        await worker_b.decide(record.escalation_id, approved=True, decided_by="admin", reason="ok")
        decided = await worker_a.wait(record.escalation_id, timeout=2.0)

        assert decided is not None and decided.status == "approved", (
            "worker A's waiter must observe a decision made on worker B; a per-worker "
            "event can never cross that boundary"
        )
        assert decided.decided_by == "admin"


class TestSingleUseIsAtomic:
    async def test_two_workers_consuming_the_same_approval_produce_one_success(
        self, redis: Any
    ) -> None:
        worker_a = _store(redis)
        worker_b = _store(redis)

        record = await _raise(worker_a)
        await worker_a.decide(record.escalation_id, approved=True, decided_by="admin")

        results = await asyncio.gather(
            worker_a.consume(record.escalation_id),
            worker_b.consume(record.escalation_id),
            return_exceptions=True,
        )
        wins = [r for r in results if not isinstance(r, BaseException)]

        assert len(wins) == 1, (
            f"an approval must authorize exactly one call; got {len(wins)} successes. "
            "Two workers must not both consume it (read-then-write is not single-use)."
        )
        assert all(isinstance(r, BaseException) for r in results if r not in wins), (
            "the losing consume must raise, not silently no-op"
        )

    async def test_a_consumed_approval_is_not_listable_again(self, redis: Any) -> None:
        store = _store(redis)
        record = await _raise(store)
        await store.decide(record.escalation_id, approved=True, decided_by="admin")
        await store.consume(record.escalation_id)

        assert await store.pending_for_approval() == []
        assert await store.get(record.escalation_id) is None, (
            "a consumed approval must be gone from the store, not merely marked"
        )


class TestDecisionsAreSingleShot:
    async def test_a_second_decision_cannot_flip_the_first(self, redis: Any) -> None:
        store = _store(redis)
        record = await _raise(store)
        await store.decide(record.escalation_id, approved=False, decided_by="admin", reason="no")

        from neuralguard.mcp.hitl import HitlError

        with pytest.raises(HitlError):
            await store.decide(record.escalation_id, approved=True, decided_by="admin")

        still = await store.get(record.escalation_id)
        assert still is not None and still.status == "denied", (
            "a denial must not be overwritable into an approval"
        )

    async def test_deciding_an_unknown_escalation_is_an_error(self, redis: Any) -> None:
        from neuralguard.mcp.hitl import HitlError

        store = _store(redis)
        with pytest.raises(HitlError):
            await store.decide("does-not-exist", approved=True, decided_by="admin")


class TestExpiryIsARefusal:
    async def test_an_expired_escalation_cannot_be_approved_late(self, redis: Any) -> None:
        from neuralguard.mcp.hitl import HitlError

        store = _store(redis, ttl_seconds=0.05)
        record = await _raise(store)
        await asyncio.sleep(0.2)  # Redis TTL

        with pytest.raises(HitlError):
            await store.decide(record.escalation_id, approved=True, decided_by="admin")
        assert await store.pending_for_approval() == []

    async def test_a_waiter_never_sees_an_approval_after_expiry(self, redis: Any) -> None:
        store = _store(redis, ttl_seconds=0.05)
        record = await _raise(store)
        assert await store.wait(record.escalation_id, timeout=0.4) is None, (
            "an expired escalation resolves to None (refusal), never to an approval"
        )


class TestCapacityIsARefusal:
    async def test_the_pending_cap_is_enforced_across_workers(self, redis: Any) -> None:
        from neuralguard.mcp.hitl import HitlCapacityError

        worker_a = _store(redis, max_pending=2)
        worker_b = _store(redis, max_pending=2)

        await _raise(worker_a)
        await _raise(worker_b)

        with pytest.raises(HitlCapacityError):
            await _raise(worker_a)

    async def test_deciding_frees_capacity(self, redis: Any) -> None:
        store = _store(redis, max_pending=1)
        from neuralguard.mcp.hitl import HitlCapacityError

        record = await _raise(store)
        with pytest.raises(HitlCapacityError):
            await _raise(store)

        await store.decide(record.escalation_id, approved=False, decided_by="admin")
        assert (await _raise(store)).escalation_id  # capacity is free again


class TestApprovalStaysBoundToThePayload:
    async def test_the_digest_survives_the_round_trip(self, redis: Any) -> None:
        store = _store(redis)
        record = await _raise(store, args={"repo": "prod", "force": True})

        seen = await store.get(record.escalation_id)
        assert seen is not None
        assert seen.digest == record.digest
        assert seen.argument_preview == '{"force": true, "repo": "prod"}', (
            "the approver's review surface must arrive intact, or they approve blind"
        )

    async def test_different_arguments_have_different_digests(self, redis: Any) -> None:
        store = _store(redis)
        one = await _raise(store, args={"repo": "prod"})
        two = await _raise(store, args={"repo": "staging"})
        assert one.digest != two.digest


class TestStoreFailureFailsClosed:
    async def test_raise_refuses_when_redis_is_unreachable(self) -> None:
        from neuralguard.mcp.hitl_redis import HitlStoreError

        class _Broken:
            async def eval(self, *args: Any, **kwargs: Any) -> Any:
                raise ConnectionError("redis down")

            async def zremrangebyscore(self, *args: Any, **kwargs: Any) -> Any:
                raise ConnectionError("redis down")

        store = _store(_Broken(), max_pending=4)
        with pytest.raises(HitlStoreError):
            await _raise(store)

    async def test_wait_returns_none_instead_of_approving_when_redis_dies(self) -> None:
        call = {"n": 0}

        class _Flaky:
            async def hgetall(self, key: str) -> dict[str, str]:
                call["n"] += 1
                if call["n"] > 1:
                    raise ConnectionError("redis down")
                return {"status": "pending", "digest": "x", "created_at": "0", "expires_at": "0"}

            async def exists(self, *args: Any, **kwargs: Any) -> int:
                return 1

        store = _store(_Flaky())
        assert await store.wait("some-id", timeout=0.5) is None, (
            "a store error must surface as a refusal (None); returning an approval on "
            "an error path would be the worst possible fail-open in a security control"
        )


class TestProductionPosture:
    """A per-process control must not be claimed in a multi-worker production pod.

    Same rule, same place, as the rate-limiter refusal in the lifespan: if the
    state is per process, more than one process makes the control a timeout.
    """

    async def test_multi_worker_production_refuses_the_per_worker_store(self) -> None:
        from neuralguard.config.settings import (
            McpGatewaySettings,
            NeuralGuardConfig,
            ServerSettings,
        )
        from neuralguard.main import create_app

        cfg = NeuralGuardConfig(
            environment="production",
            server=ServerSettings(workers=2),
            auth={"enabled": True, "api_keys": ["ng_" + "a" * 24 + "|default"]},
            # Isolate the HITL rule: satisfy the sibling multi-worker rate-limit
            # rule (redis backend) so the failure under test is the HITL one.
            rate_limit={
                "enabled": True,
                "backend": "redis",
                "redis_url": "redis://localhost:6379/0",
            },
            mcp=McpGatewaySettings(
                enabled=True,
                upstream_url="http://upstream.invalid",
                hitl_enabled=True,
                hitl_store="memory",
                hitl_approver_tenants=["admin"],
            ),
        )
        app = create_app(cfg)
        with pytest.raises(RuntimeError, match="hitl_store"):
            async with app.router.lifespan_context(app):
                pass

    async def test_single_worker_may_use_the_in_memory_store(self) -> None:
        from neuralguard.config.settings import (
            McpGatewaySettings,
            NeuralGuardConfig,
            ServerSettings,
        )
        from neuralguard.main import create_app

        cfg = NeuralGuardConfig(
            environment="production",
            server=ServerSettings(workers=1),
            auth={"enabled": True, "api_keys": ["ng_" + "a" * 24 + "|default"]},
            mcp=McpGatewaySettings(
                enabled=True,
                upstream_url="http://upstream.invalid",
                hitl_enabled=True,
                hitl_store="memory",
                hitl_approver_tenants=["admin"],
            ),
        )
        assert cfg.mcp.hitl_store == "memory", (
            "a single-worker deployment keeps the in-memory store: the defect only "
            "appears when a second process has its own registry"
        )
        create_app(cfg)  # constructs without a store swap

    def test_redis_store_requires_a_url(self) -> None:
        from neuralguard.config.settings import McpGatewaySettings

        with pytest.raises(ValueError, match="hitl_redis_url"):
            McpGatewaySettings(hitl_enabled=True, hitl_store="redis", hitl_redis_url="")


class TestWaitIsBounded:
    async def test_wait_returns_none_on_timeout_without_a_decision(self, redis: Any) -> None:
        store = _store(redis)
        record = await _raise(store)
        started = time.monotonic()
        decided = await store.wait(record.escalation_id, timeout=0.3)
        elapsed = time.monotonic() - started

        assert decided is None
        assert elapsed < 2.0, (
            "the poll must honour the caller's timeout; a shared store must not turn a "
            "bounded wait into an unbounded one"
        )


class TestTheAppSelectsTheStoreItWasToldToUse:
    """A config value the app ignores is worse than no config value."""

    def test_hitl_store_redis_builds_the_shared_store(self) -> None:
        from neuralguard.config.settings import McpGatewaySettings, NeuralGuardConfig
        from neuralguard.main import create_app
        from neuralguard.mcp.hitl_redis import RedisHitlStore

        cfg = NeuralGuardConfig(
            environment="development",
            mcp=McpGatewaySettings(
                enabled=True,
                upstream_url="http://upstream.invalid",
                hitl_enabled=True,
                hitl_store="redis",
                hitl_redis_url="redis://localhost:6379/0",
                hitl_approver_tenants=["admin"],
            ),
        )
        app = create_app(cfg)
        assert isinstance(app.state.mcp_hitl, RedisHitlStore), (
            "hitl_store=redis must produce the shared store; a per-worker broker "
            "here would silently reintroduce the multi-replica defect"
        )

    def test_hitl_store_memory_stays_per_worker(self) -> None:
        from neuralguard.config.settings import McpGatewaySettings, NeuralGuardConfig
        from neuralguard.main import create_app
        from neuralguard.mcp.hitl import HitlBroker

        cfg = NeuralGuardConfig(
            environment="development",
            mcp=McpGatewaySettings(
                enabled=True,
                upstream_url="http://upstream.invalid",
                hitl_enabled=True,
                hitl_store="memory",
                hitl_approver_tenants=["admin"],
            ),
        )
        assert isinstance(create_app(cfg).state.mcp_hitl, HitlBroker)
