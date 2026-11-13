"""NG-15: human-in-the-loop approval for MCP ESCALATE decisions.

Before this module the MCP gateway's ESCALATE verdict *looked* like a workflow
and behaved like a denial: it refused the call, audited it, and left the
operator to discover it in the trail. A policy that says "a human must approve
this" had no way to actually approve it. NG-15 closes the loop:

1. The gateway **raises** an escalation bound to the exact call — tenant, agent,
   tool, method and a SHA-256 digest of the canonical JSON-RPC body.
2. It waits, bounded, for a decision (`hitl_timeout_seconds`, hard-capped at 60).
3. A **tenant-level** credential whose tenant is in ``hitl_approver_tenants``
   (default: the admin tenant) decides via ``POST /v1/mcp/escalations/{id}``.
4. Approved → the request continues through every remaining gate (baseline,
   provenance, credentials) and forwards. Denied, timed out, expired or
   capacity-refused → the call is refused, exactly as before.

Design rules, all fail-closed:

- **No approver configured means no approval path.** The default approver set is
  the admin tenant; if nothing is configured, ``hitl_enabled`` is off and the old
  refuse-and-audit behaviour is unchanged.
- **An agent cannot approve itself.** Decisions require a tenant-level
  credential (no authenticated agent identity). The abuse case this feature
  creates is an agent raising an escalation for itself and approving it; the
  actor rule makes that impossible even when the agent's tenant is an approver.
- **Approval binds to the payload.** The digest covers the canonical body, so an
  approval cannot be replayed against different arguments, and the request is
  single-use (``consume``) once approved.
- **Approval is not authority.** An approved call still faces the argument
  constraints, the signed-catalog baseliner, the provenance gate and the
  credential resolver — a human saying yes cannot make a rug-pull execute.
- **Bounded everything.** Pending escalations are capped; TTLs and wait
  timeouts are capped; expired records are purged and can never be approved
  late.
- **Argument values stay out of the audit trail.** The approver view carries a
  bounded argument preview (that is the review), but audit events carry only the
  digest and ids. The preview never leaves the authenticated approver API.

Stated limit: the broker is **per worker** and in memory. A decision must reach
the worker holding the pending request, so a multi-worker deployment needs
sticky routing for ``/v1/mcp/escalations`` (or a shared store, which this build
does not ship). The same limit is documented for tool-catalog baselines, and it
is stated in the runbook rather than hidden.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

# The approver view is a review surface, not a log sink: bound the argument
# preview so a large payload cannot turn one escalation into unbounded memory.
_MAX_PREVIEW_CHARS = 2000

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_DENIED = "denied"
STATUS_EXPIRED = "expired"
STATUS_CONSUMED = "consumed"


class HitlError(Exception):
    """An escalation operation is invalid for the current state (fail-closed)."""


class HitlCapacityError(HitlError):
    """Too many pending escalations; the call is refused rather than queued."""


def payload_digest(payload: Any) -> str:
    """SHA-256 of the canonical JSON body — what an approval is bound to.

    Sorted keys + no whitespace means the same logical body always digests the
    same, while any change to any argument changes the digest.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _preview(arguments: Any) -> str:
    try:
        rendered = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):  # pragma: no cover - parsed JSON is serializable
        return "[unrenderable]"
    if len(rendered) <= _MAX_PREVIEW_CHARS:
        return rendered
    return rendered[:_MAX_PREVIEW_CHARS] + "…(truncated)"


@dataclass(slots=True)
class PendingEscalation:
    """One call waiting for a human. Mutable: the status is the state machine."""

    escalation_id: str
    tenant: str
    agent: str | None
    method: str | None
    tool: str | None
    rule_id: str
    reason: str
    source: str  # "intent_gate" | "argument_gate"
    digest: str
    argument_preview: str
    created_at: float  # monotonic, for expiry
    expires_at: float
    status: str = STATUS_PENDING
    decided_by: str | None = None
    decision_reason: str | None = None
    decided_at: float | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def is_expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.monotonic()) >= self.expires_at

    def to_view(self) -> dict[str, Any]:
        """The approver-facing view. Ids, digest and a bounded preview — no secrets."""
        return {
            "escalation_id": self.escalation_id,
            "tenant": self.tenant,
            "agent": self.agent,
            "method": self.method,
            "tool": self.tool,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "source": self.source,
            "request_digest": self.digest,
            "argument_preview": self.argument_preview,
            "created_at": self.created_at,
            "expires_in_seconds": max(0.0, round(self.expires_at - time.monotonic(), 3)),
            "status": self.status,
        }


class HitlBroker:
    """Per-worker escalation registry: raise, await, decide, consume.

    Every method is safe to call concurrently from request handlers: mutations
    happen under one ``asyncio.Lock`` and waiters block on a per-record event.
    """

    def __init__(self, ttl_seconds: float = 300.0, max_pending: int = 256) -> None:
        self._ttl = ttl_seconds
        self._max_pending = max_pending
        self._records: dict[str, PendingEscalation] = {}
        self._lock = asyncio.Lock()

    # ── visibility for tests/ops ──

    @property
    def pending_count(self) -> int:
        return sum(1 for r in self._records.values() if r.status == STATUS_PENDING)

    def get(self, escalation_id: str) -> PendingEscalation | None:
        return self._records.get(escalation_id)

    # ── lifecycle ──

    def _purge_expired(self, now: float) -> None:
        # Every record (pending, approved-but-unconsumed, denied) dies at its
        # TTL: a stale approval must never authorize a later call.
        for record in list(self._records.values()):
            if not record.is_expired(now):
                continue
            if record.status == STATUS_PENDING:
                record.status = STATUS_EXPIRED
                record.decided_at = now
                # Wake any waiter so it reports a timeout rather than hanging.
                record.event.set()
            self._records.pop(record.escalation_id, None)

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
        """Register a pending call. Raises ``HitlCapacityError`` when full.

        Capacity is a refusal, never a queue: an unbounded backlog of
        half-approved tool calls is its own denial of service.
        """
        now = time.monotonic()
        async with self._lock:
            self._purge_expired(now)
            if self.pending_count >= self._max_pending:
                raise HitlCapacityError(
                    f"too many pending escalations ({self._max_pending}); refusing rather than queueing"
                )
            record = PendingEscalation(
                escalation_id=secrets.token_urlsafe(18),
                tenant=tenant,
                agent=agent,
                method=method,
                tool=tool,
                rule_id=rule_id,
                reason=reason,
                source=source,
                digest=payload_digest(payload),
                argument_preview=_preview(arguments),
                created_at=now,
                expires_at=now + self._ttl,
            )
            self._records[record.escalation_id] = record
            return record

    def pending_for_approval(self) -> list[PendingEscalation]:
        """All unexpired pending records, oldest first (approver list view).

        Purges expired records first, so the list an approver sees is the whole
        truth: an expired escalation must not be listable, and an
        approved-but-unconsumed record (a client that disconnected after the
        decision) must not linger past its TTL. Safe without the lock — no
        await here, so nothing can interleave on the event loop.
        """
        self._purge_expired(time.monotonic())
        pending = [r for r in self._records.values() if r.status == STATUS_PENDING]
        return sorted(pending, key=lambda r: r.created_at)

    async def decide(
        self, escalation_id: str, *, approved: bool, decided_by: str, reason: str | None = None
    ) -> PendingEscalation:
        """Record a human decision. Single-shot: a second decision is an error."""
        now = time.monotonic()
        async with self._lock:
            record = self._records.get(escalation_id)
            if record is None:
                raise HitlError("unknown or expired escalation")
            if record.status == STATUS_PENDING and record.is_expired(now):
                record.status = STATUS_EXPIRED
                record.decided_at = now
                record.event.set()
                raise HitlError("escalation expired before a decision")
            if record.status != STATUS_PENDING:
                raise HitlError(f"escalation already {record.status}")
            record.status = STATUS_APPROVED if approved else STATUS_DENIED
            record.decided_by = decided_by
            record.decision_reason = reason
            record.decided_at = now
            record.event.set()
            return record

    async def wait(self, escalation_id: str, timeout: float) -> PendingEscalation | None:
        """Wait (bounded) for a decision; None means no decision in time.

        The caller maps None to a refusal. There is no path here that turns a
        timeout into an approval.
        """
        record = self._records.get(escalation_id)
        if record is None:
            return None
        try:
            await asyncio.wait_for(record.event.wait(), timeout=max(0.0, timeout))
        except TimeoutError:
            return None
        if record.status in {STATUS_APPROVED, STATUS_DENIED}:
            return record
        return None

    async def consume(self, escalation_id: str) -> None:
        """Mark an approval used, so it can never authorize a second call."""
        async with self._lock:
            record = self._records.get(escalation_id)
            if record is None:
                raise HitlError("unknown or expired escalation")
            if record.status != STATUS_APPROVED:
                raise HitlError(f"escalation is {record.status}, not approved")
            record.status = STATUS_CONSUMED
            self._records.pop(escalation_id, None)
