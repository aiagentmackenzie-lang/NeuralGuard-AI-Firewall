"""NG-15 (multi-worker): a shared Redis escalation store.

NG-15 shipped a per-worker, in-memory registry and documented the limit: a
decision must reach the worker holding the pending request, so a deployment with
more than one replica needs sticky routing or a shared store. Sticky routing is
a load-balancer setting that nothing in this repo can prove; a shared store is a
property of the product. This module is the shared store.

Every NG-15 argument survives the move, and three of them had to be *implemented*
rather than inherited from a dict and an ``asyncio.Lock``:

- **Single use.** ``consume`` is a Lua compare-and-set that deletes the record.
  Read-then-write would let two workers both consume one approval; the script
  makes "exactly one" true across processes, not just across coroutines.
- **Single-shot decisions.** ``decide`` transitions only from ``pending``; a
  denial can never be overwritten into an approval.
- **Expiry is a refusal.** Records carry a Redis TTL, so an expired escalation is
  *gone*: it cannot be listed, approved late, or consumed. Redis TTL is the
  authority, not a timestamp compared by the app.
- **Capacity is a refusal.** Trimmed sorted-set cardinality inside the same
  script, so two workers cannot both squeeze past the cap.
- **Fail closed.** Every Redis error becomes :class:`HitlStoreError`; no path
  returns an approval from an error. ``wait`` polls the shared state and returns
  ``None`` (refusal) on timeout, expiry or store failure.

Time is wall clock here (not monotonic) because workers do not share a boot
instant; the gateway's *wait* stays bounded by the caller's timeout either way.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Awaitable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from neuralguard.mcp.hitl import (
    STATUS_APPROVED,
    STATUS_CONSUMED,
    STATUS_DENIED,
    STATUS_PENDING,
    HitlCapacityError,
    HitlError,
    PendingEscalation,
    _preview,
    payload_digest,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from redis.asyncio import Redis

_KEY_PREFIX = "ng:hitl:"
_INDEX_KEY = f"{_KEY_PREFIX}index"  # zset: member=escalation_id, score=expires_at

# Poll interval for a waiter. Small enough that a human decision feels immediate
# (the caller's timeout is seconds), large enough not to hammer Redis per waiter.
_POLL_INTERVAL_SECONDS = 0.1

# KEYS: record, index.  ARGV: id, tenant, agent, method, tool, rule_id, reason,
# source, digest, preview, created_at, expires_at, ttl_seconds, max_pending, now
_RAISE_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', ARGV[15])
local pending = redis.call('ZCARD', KEYS[2])
if pending >= tonumber(ARGV[14]) then
  return 0
end
redis.call('HSET', KEYS[1],
  'escalation_id', ARGV[1], 'tenant', ARGV[2], 'agent', ARGV[3],
  'method', ARGV[4], 'tool', ARGV[5], 'rule_id', ARGV[6],
  'reason', ARGV[7], 'source', ARGV[8], 'digest', ARGV[9],
  'argument_preview', ARGV[10], 'created_at', ARGV[11],
  'expires_at', ARGV[12], 'status', 'pending')
redis.call('PEXPIRE', KEYS[1], ARGV[13])
redis.call('ZADD', KEYS[2], ARGV[12], ARGV[1])
return 1
"""

# KEYS: record, index.  ARGV: new_status, decided_by, decision_reason, decided_at, id
# Returns 1 decided, 0 already-decided, -1 unknown/expired.
_DECIDE_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then
  return -1
end
local status = redis.call('HGET', KEYS[1], 'status')
if status ~= 'pending' then
  return 0
end
redis.call('HSET', KEYS[1],
  'status', ARGV[1], 'decided_by', ARGV[2],
  'decision_reason', ARGV[3], 'decided_at', ARGV[4])
redis.call('ZREM', KEYS[2], ARGV[5])
return 1
"""

# KEYS: record, index.  ARGV: id
# Returns 1 consumed, 0 not-approved, -1 unknown/expired.
_CONSUME_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then
  return -1
end
local status = redis.call('HGET', KEYS[1], 'status')
if status ~= 'approved' then
  return 0
end
redis.call('HSET', KEYS[1], 'status', 'consumed')
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[2], ARGV[1])
return 1
"""


class HitlStoreError(HitlError):
    """The shared store is unreachable or returned an unusable answer (fail-closed)."""


@dataclass(slots=True)
class _RecordFields:
    """The subset of ``PendingEscalation`` that crosses the process boundary."""

    escalation_id: str
    tenant: str
    agent: str | None
    method: str | None
    tool: str | None
    rule_id: str
    reason: str
    source: str
    digest: str
    argument_preview: str
    created_at: float
    expires_at: float
    status: str
    decided_by: str | None = None
    decision_reason: str | None = None
    decided_at: float | None = None

    def to_escalation(self) -> PendingEscalation:
        return PendingEscalation(
            escalation_id=self.escalation_id,
            tenant=self.tenant,
            agent=self.agent,
            method=self.method,
            tool=self.tool,
            rule_id=self.rule_id,
            reason=self.reason,
            source=self.source,
            digest=self.digest,
            argument_preview=self.argument_preview,
            created_at=self.created_at,
            expires_at=self.expires_at,
            status=self.status,
            decided_by=self.decided_by,
            decision_reason=self.decision_reason,
            decided_at=self.decided_at,
        )


def _record_key(escalation_id: str) -> str:
    return f"{_KEY_PREFIX}rec:{escalation_id}"


def _as_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _from_hash(raw: dict[str, str]) -> _RecordFields:
    return _RecordFields(
        escalation_id=raw.get("escalation_id", ""),
        tenant=raw.get("tenant", ""),
        agent=raw.get("agent") or None,
        method=raw.get("method") or None,
        tool=raw.get("tool") or None,
        rule_id=raw.get("rule_id", ""),
        reason=raw.get("reason", ""),
        source=raw.get("source", ""),
        digest=raw.get("digest", ""),
        argument_preview=raw.get("argument_preview", ""),
        created_at=_as_float(raw.get("created_at")),
        expires_at=_as_float(raw.get("expires_at")),
        status=raw.get("status", STATUS_PENDING),
        decided_by=raw.get("decided_by") or None,
        decision_reason=raw.get("decision_reason") or None,
        decided_at=_as_float(raw["decided_at"]) if raw.get("decided_at") else None,
    )


class RedisHitlStore:
    """Multi-worker HITL registry. Same surface as ``HitlBroker``, shared state.

    Construct with an existing ``redis.asyncio.Redis`` client (the app reuses
    its Redis dependency; tests pass ``fakeredis.FakeAsyncRedis``).
    """

    def __init__(
        self,
        *,
        redis: Redis,
        ttl_seconds: float = 300.0,
        max_pending: int = 256,
        poll_interval_seconds: float = _POLL_INTERVAL_SECONDS,
    ) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._max_pending = max_pending
        self._poll = poll_interval_seconds

    # ── internals ──

    async def _awaitable(self, value: Any) -> Any:
        """Await a redis-py call whose type is ``Awaitable[T] | T``.

        redis-py types every command against one surface that serves both the
        sync and async clients, so the return is a union. Awaiting it is correct
        at runtime for the async client and this keeps mypy honest instead of
        scattering ``type: ignore``.
        """
        if isinstance(value, Awaitable):
            return await value
        return value

    async def _hgetall(self, key: str) -> dict[str, Any]:
        raw = await self._awaitable(self._redis.hgetall(key))
        return cast("dict[str, Any]", raw)

    async def _eval(self, script: str, keys: Sequence[str], args: Sequence[Any]) -> Any:
        try:
            return await self._awaitable(self._redis.eval(script, len(keys), *keys, *args))
        except HitlError:  # pragma: no cover - defensive
            raise
        except Exception as exc:
            raise HitlStoreError(f"shared HITL store unavailable: {exc}") from exc

    # ── lifecycle ──

    async def raise_escalation(
        self,
        *,
        tenant: str,
        agent: str | None,
        method: str | None,
        tool: str | None,
        rule_id: str,
        reason: str,
        source: str,
        payload: Any,
        arguments: Any = None,
    ) -> PendingEscalation:
        """Register a pending call in the shared store (capacity is a refusal)."""
        now = time.time()
        escalation_id = secrets.token_urlsafe(18)
        expires_at = now + self._ttl
        digest = payload_digest(payload)
        preview = _preview(arguments)

        decided = await self._eval(
            _RAISE_LUA,
            [_record_key(escalation_id), _INDEX_KEY],
            [
                escalation_id,
                tenant,
                agent or "",
                method or "",
                tool or "",
                rule_id,
                reason,
                source,
                digest,
                preview,
                now,
                expires_at,
                # PEXPIRE takes whole milliseconds: a sub-second TTL is a test
                # affordance, and flooring it to EXPIRE's 1 s would let an
                # 'expired' record still be approved.
                max(1, int(self._ttl * 1000)),
                self._max_pending,
                now,
            ],
        )
        if int(decided) != 1:
            raise HitlCapacityError(
                f"too many pending escalations ({self._max_pending}); refusing rather than queueing"
            )
        return _RecordFields(
            escalation_id=escalation_id,
            tenant=tenant,
            agent=agent,
            method=method,
            tool=tool,
            rule_id=rule_id,
            reason=reason,
            source=source,
            digest=digest,
            argument_preview=preview,
            created_at=now,
            expires_at=expires_at,
            status=STATUS_PENDING,
        ).to_escalation()

    async def get(self, escalation_id: str) -> PendingEscalation | None:
        """One record, or None when unknown/expired/consumed."""
        try:
            raw = await self._awaitable(self._redis.hgetall(_record_key(escalation_id)))
        except Exception as exc:
            raise HitlStoreError(f"shared HITL store unavailable: {exc}") from exc
        if not raw:
            return None
        return _from_hash(raw).to_escalation()

    async def pending_for_approval(self) -> list[PendingEscalation]:
        """All pending records, oldest first — the approver's whole truth."""
        try:
            ids = await self._awaitable(self._redis.zrange(_INDEX_KEY, 0, -1))
        except Exception as exc:
            raise HitlStoreError(f"shared HITL store unavailable: {exc}") from exc

        out: list[PendingEscalation] = []
        stale: list[str] = []
        for escalation_id in ids:
            raw = await self._awaitable(self._redis.hgetall(_record_key(escalation_id)))
            if not raw:  # TTL expired between the zset read and the hash read
                stale.append(escalation_id)
                continue
            record = _from_hash(raw)
            if record.status == STATUS_PENDING:
                out.append(record.to_escalation())
            elif record.status in {STATUS_APPROVED, STATUS_DENIED, STATUS_CONSUMED}:
                # Decided records leave the pending index on read: the zset is
                # an index of *candidates*, the hash is the truth.
                stale.append(escalation_id)
        if stale:
            with suppress(Exception):  # a failed cleanup must not fail the read
                await self._awaitable(self._redis.zrem(_INDEX_KEY, *stale))
        return sorted(out, key=lambda r: r.created_at)

    async def decide(
        self, escalation_id: str, *, approved: bool, decided_by: str, reason: str | None = None
    ) -> PendingEscalation:
        """Record a human decision. Single-shot, atomic across workers."""
        result = int(
            await self._eval(
                _DECIDE_LUA,
                [_record_key(escalation_id), _INDEX_KEY],
                [
                    STATUS_APPROVED if approved else STATUS_DENIED,
                    decided_by,
                    reason or "",
                    time.time(),
                    escalation_id,
                ],
            )
        )
        if result == -1:
            raise HitlError("unknown or expired escalation")
        if result == 0:
            current = await self.get(escalation_id)
            state = current.status if current is not None else "unknown"
            raise HitlError(f"escalation already {state}")

        record = await self.get(escalation_id)
        if record is None:  # expired in the microseconds after the decision
            raise HitlError("unknown or expired escalation")
        return record

    async def wait(self, escalation_id: str, timeout: float) -> PendingEscalation | None:
        """Poll the shared state, bounded by ``timeout``; None means refusal.

        There is deliberately no path from an error, an expiry or a timeout to an
        approval: every non-decision returns None and the gateway refuses.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                raw = await self._awaitable(self._redis.hgetall(_record_key(escalation_id)))
            except Exception:  # store failure == no decision == refusal
                return None
            if not raw:
                return None  # expired or consumed (or never existed)
            record = _from_hash(raw)
            if record.status in {STATUS_APPROVED, STATUS_DENIED}:
                return record.to_escalation()
            if time.monotonic() >= deadline:
                return None
            await self._sleep()

    async def _sleep(self) -> None:
        import asyncio

        await asyncio.sleep(self._poll)

    async def consume(self, escalation_id: str) -> None:
        """Use an approval exactly once, atomically across workers."""
        result = int(
            await self._eval(
                _CONSUME_LUA, [_record_key(escalation_id), _INDEX_KEY], [escalation_id]
            )
        )
        if result == -1:
            raise HitlError("unknown or expired escalation")
        if result == 0:
            current = await self.get(escalation_id)
            state = current.status if current is not None else "expired"
            raise HitlError(f"escalation is {state}, not approved")

    async def close(self) -> None:
        """Release the client. The app may share the client with other backends."""
        with suppress(Exception):  # best effort on shutdown
            await self._awaitable(self._redis.aclose())

    # ── visibility for the approver API ──

    async def pending_count(self) -> int:
        return len(await self.pending_for_approval())
