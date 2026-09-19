"""SIEM routing + BLOCK-rate spike alerting (P2-7).

Routes NeuralGuard verdict events to security-monitoring infrastructure:

- **Splunk HEC** (native): POSTs to the HTTP Event Collector endpoint with
  the Splunk-formatted envelope (``Authorization: Splunk <token>``).
- **Generic JSON webhook**: POSTs the same structured event for ingestion by
  ELK/Elastic (webhook input), Microsoft Sentinel (Logic Apps connector),
  or any JSON-consuming collector. NeuralGuard does NOT claim native ELK or
  Sentinel connectors — those products consume this webhook through their
  own supported integration points.

**Spike detection**: a bounded sliding window tracks the BLOCK ratio across
recent verdicts. Crossing the threshold fires ONE alert event (edge-triggered,
cooldown-suppressed — no alert storms).

Delivery contract (observability, not an inline control):
- Fire-and-forget from the request path; delivery failures are logged and
  counted, never raised, and NEVER affect the firewall verdict.
- Bounded concurrency: at most ``max_inflight`` deliveries at once; beyond
  that events are DROPPED with a warning log. Bounded memory, no unbounded
  queue to OOM the worker under load.
- Tokens (Splunk HEC / webhook bearer) are held server-side and are never
  logged or echoed.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import httpx
import structlog

if TYPE_CHECKING:
    from neuralguard.config.settings import SiemSettings
    from neuralguard.models.schemas import AuditEvent

logger = structlog.get_logger(__name__)

# Log at most one drop-warning per this many drops (bounded log noise under load).
_DROP_LOG_EVERY = 50

# ECS severity mapping: NeuralGuard verdict → ScarletAI alert severity.
# High-confidence BLOCKs escalate to critical (a confirmed attack, not a
# suspicion); ALLOW maps to info and is filtered by default anyway.
_VERDICT_SEVERITY = {
    "block": "high",
    "quarantine": "critical",
    "escalate": "medium",
    "sanitize": "medium",
    "rate_limit": "low",
    "allow": "info",
}

# Injection-shaped threat categories (the closed T-* vocabulary): a verdict
# carrying one of these ALSO maps into ScarletAI's ai-category
# ``ai_prompt_injection`` companion event — Scarlet's ai_usage doctrine puts
# the mapping AT THE PRODUCER ("a NeuralGuard prompt-injection verdict maps
# into ai_prompt_injection at its producer"), and Scarlet's Sigma compiler
# only selects FLAT columns (raw_data is not selectable), so the companion
# event is what makes single NeuralGuard injection detections alertable.
_INJECTION_CATEGORIES = frozenset({"T-PI-D", "T-PI-I", "T-JB"})

# ScarletAI's per-POST batch cap (api/ingest.py: 413 above 1000 events).
# Configured batch sizes cannot reach this (batch_max_events ≤ 1000, arrays
# ≤ 3 events) — the split below is pure defensive depth.
_SCARLETAI_BATCH_LIMIT = 1000

# ScarletAI IngestEvent.user_name / process_name max_length (256 chars).
_SCARLETAI_ACTOR_CAP = 256


def map_to_scarletai(payload: dict[str, Any], settings: SiemSettings) -> dict[str, Any]:
    """Map a NeuralGuard SIEM envelope into a ScarletAI IngestEvent dict.

    ScarletAI normalizes to ECS (Elastic Common Schema). The mapping keeps
    the tamper-evident chain (event_hash + event_sig) inside ``raw_data`` so
    the SIEM inherits NeuralGuard's non-repudiation, and NL2SQL can query
    findings/scanner_details directly.

    - host_name: configured host or this machine's hostname (required field,
      sanitized by ScarletAI's own validator).
    - event_category: ECS ``intrusion_detection`` — these ARE detection
      events, not host telemetry.
    - event_type: ECS ``info`` (the verdict detail rides in raw_data).
    - event_action: queryable verdict discriminator (verdict_block,
      block_rate_spike, ...).
    - severity: verdict → ECS severity; a spike alert is critical.
    """
    import platform as _platform

    event_type = payload.get("event_type", "neuralguard.verdict")
    inner = payload.get("event", {})
    if event_type == "neuralguard.block_spike":
        verdict = "block"  # a spike IS a block storm
        action = "block_rate_spike"
        severity = "critical"
    else:
        verdict = str(inner.get("verdict", "allow"))
        action = f"verdict_{verdict}"
        severity = _VERDICT_SEVERITY.get(verdict, "info")
        if verdict == "block" and float(inner.get("confidence", 0.0)) >= 0.9:
            severity = "critical"

    host = settings.scarletai_host or _platform.node() or "neuralguard"
    ts = payload.get("time")
    timestamp = (
        datetime.fromtimestamp(ts, UTC).isoformat()
        if isinstance(ts, (int, float))
        else datetime.now(UTC).isoformat()
    )
    return {
        "@timestamp": timestamp,
        "host_name": host[:253],
        "source": "neuralguard",
        "event_category": "intrusion_detection",
        "event_type": "info",
        "event_action": action,
        "severity": severity,
        "raw_data": {"neuralguard": inner},
    }


def map_to_scarletai_batch(payload: dict[str, Any], settings: SiemSettings) -> list[dict[str, Any]]:
    """Map one SIEM envelope into the ScarletAI IngestEvent dicts for its POST.

    Fleet contract (companion events, scarletai sink ONLY — Splunk/webhook
    consumers keep the full audit event, which already carries everything):

    - **Actor slot (A1):** every mapped verdict event sets ``user_name`` to
      the audit event's ``tenant_id`` (ECS-borrowed actor slot, Scarlet's
      own ai_usage convention: user_name is the ACTOR). ``tenant_id`` ALSO
      stays inside ``raw_data.neuralguard`` — Scarlet's
      ``ai_verdict_block_sustained`` correlation detector groups on it.
    - **Injection companion (A2):** a verdict whose ``threat_categories``
      intersects {T-PI-D, T-PI-I, T-JB} adds ONE ``ai``-category
      ``ai_prompt_injection`` event — same severity, ``@timestamp`` and
      ``host_name`` as the parent, marked ``neuralguard_companion`` in
      ``raw_data``. No ``process_name`` (there is no tool in an input scan).
    - **MCP-denial companion (A3):** an MCP-gateway event (``metadata.mcp``)
      with verdict block/escalate adds ONE ``mcp_tool_denied`` companion
      with the denied tool in ``process_name``. A verdict event can yield
      BOTH companions (an injection-shaped MCP gate block is two domain
      lenses on one decision).
    - **Spikes (A4):** ``neuralguard.block_spike`` payloads stay
      single-element — the spike payload is not an AuditEvent dump.

    Companions ride the SAME POST as the parent (they are derived from it,
    so they inherit its routing decision, including the allow-filter).
    Purity/totality: no companion path may raise — a companion-build failure
    logs and degrades to the parent-only delivery (observability never
    affects verdicts, the P2-7 doctrine).
    """
    parent = map_to_scarletai(payload, settings)
    if payload.get("event_type") == "neuralguard.block_spike":
        return [parent]  # A4: spikes stay single-element, no slot mapping

    inner = payload.get("event", {})
    tenant = inner.get("tenant_id")
    if isinstance(tenant, str) and tenant:
        # A1: actor slot, capped to IngestEvent's 256-char field limit.
        parent["user_name"] = tenant[:_SCARLETAI_ACTOR_CAP]

    events: list[dict[str, Any]] = [parent]

    # A2: injection companion — Scarlet's closed ai-category vocabulary.
    try:
        categories = {str(c) for c in (inner.get("threat_categories") or [])}
        if categories & _INJECTION_CATEGORIES:
            events.append(
                {
                    "@timestamp": parent["@timestamp"],
                    "host_name": parent["host_name"],
                    "source": "neuralguard",
                    "event_category": "ai",
                    "event_type": "info",
                    "event_action": "ai_prompt_injection",
                    "user_name": parent.get("user_name"),
                    "severity": parent["severity"],
                    "raw_data": {
                        "neuralguard": inner,
                        "neuralguard_companion": "prompt_injection",
                    },
                }
            )
    except Exception as exc:
        logger.warning(
            "siem_companion_build_failed",
            companion="prompt_injection",
            error=exc.__class__.__name__,
        )

    # A3: MCP-denial companion — the denied tool call as an ai-category event
    # (tool in process_name per Scarlet's ai_usage producer convention).
    try:
        metadata = inner.get("metadata") or {}
        if metadata.get("mcp") and str(inner.get("verdict", "")) in {"block", "escalate"}:
            tool = metadata.get("tool")
            events.append(
                {
                    "@timestamp": parent["@timestamp"],
                    "host_name": parent["host_name"],
                    "source": "neuralguard",
                    "event_category": "ai",
                    "event_type": "info",
                    "event_action": "mcp_tool_denied",
                    "user_name": parent.get("user_name"),
                    "process_name": tool[:_SCARLETAI_ACTOR_CAP]
                    if isinstance(tool, str) and tool
                    else None,
                    "severity": parent["severity"],
                    "raw_data": {
                        "neuralguard": inner,
                        "neuralguard_companion": "mcp_denial",
                    },
                }
            )
    except Exception as exc:
        logger.warning(
            "siem_companion_build_failed",
            companion="mcp_denial",
            error=exc.__class__.__name__,
        )

    return events


class SiemRouter:
    """Fan-out of audit events to SIEM sinks + BLOCK-rate spike alerting.

    Constructed only when ``config.siem.enabled`` AND at least one sink is
    configured (``create_app`` refuses otherwise in production). All methods
    are safe to call from the request path: ``route`` never blocks on network
    I/O and never raises.
    """

    def __init__(
        self,
        settings: SiemSettings,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        # Injectable httpx transport (tests: MockTransport — implements BOTH
        # sync and async transport interfaces). None = real network.
        self._transport = transport
        self._background: set[asyncio.Task[None]] = set()
        self._sinks: list[str] = []
        if settings.splunk_hec_url:
            self._sinks.append("splunk_hec")
        if settings.webhook_url:
            self._sinks.append("webhook")
        if settings.scarletai_url:
            self._sinks.append("scarletai")
        self._semaphore = asyncio.Semaphore(settings.max_inflight)
        # Spike detector: ring of recent verdicts (True = BLOCK) + running count.
        self._recent_blocks: deque[bool] = deque(maxlen=settings.spike_window)
        self._block_count: int = 0
        self._in_spike: bool = False
        # None = never alerted. (0.0 would make a fresh process — CI VMs boot
        # with time.monotonic() near zero — treat itself as in-cooldown and
        # swallow the FIRST spike alert for `cooldown_seconds`.)
        self._last_alert_ts: float | None = None
        self._drops: int = 0
        # Opt-in batch buffering (scarletai sink only): verdict families
        # (parent + companions) accumulate and flush on size/time. Default
        # cap = 1 → never used, byte-identical legacy behavior.
        self._batch_buffer: list[list[dict[str, Any]]] = []
        self._batch_lock = asyncio.Lock()
        self._flusher_task: asyncio.Task[None] | None = None

    # ── Public API (request-path safe) ────────────────────────────────────

    def route(self, event: AuditEvent) -> None:
        """Route one audit event: spike bookkeeping + fire-and-forget delivery.

        Called from ``AuditLogger._persist`` AFTER the chain hash is stamped,
        so SIEM consumers receive the tamper-evident form of every event.

        ALLOW-verdict filtering (scarletai sink only, ``route_allow=False``)
        filters DELIVERY, never the spike detector — the block RATIO needs
        allow verdicts in its denominator.
        """
        if not self._sinks:
            return
        try:
            payload = self._verdict_payload(event)
        except Exception:  # pragma: no cover — serialization of a typed model
            logger.warning("siem_serialize_failed", event_id=event.event_id)
            return

        spiked = self._record_and_detect(event.verdict.value)
        if spiked:
            self._schedule(self._spike_payload())

        # ALLOW filter: scarletai-only deployments default to alert-quality.
        if (
            event.verdict.value == "allow"
            and not self.settings.scarletai_route_allow
            and self._sinks == ["scarletai"]
        ):
            return
        self._schedule(payload)

    # ── Spike detection ───────────────────────────────────────────────────

    def _record_and_detect(self, verdict: str) -> bool:
        """Update the sliding window; return True when an alert must fire.

        Edge-triggered: entering the spike state fires once; the alert
        re-arms only after the cooldown elapses AND the ratio recovers below
        the threshold.
        """
        is_block = verdict == "block"
        if len(self._recent_blocks) == self.settings.spike_window:
            # Window full: deque will evict the oldest element on append.
            self._block_count -= int(self._recent_blocks[0])
        self._recent_blocks.append(is_block)
        self._block_count += int(is_block)

        now = time.monotonic()
        window_full = len(self._recent_blocks) == self.settings.spike_window
        ratio = self._block_count / len(self._recent_blocks) if self._recent_blocks else 0.0
        over = window_full and ratio >= self.settings.spike_block_threshold
        cooled = (
            self._last_alert_ts is None
            or (now - self._last_alert_ts) >= self.settings.spike_cooldown_seconds
        )

        if over and not self._in_spike and cooled:
            self._in_spike = True
            self._last_alert_ts = now
            logger.warning(
                "block_rate_spike",
                window_events=len(self._recent_blocks),
                block_count=self._block_count,
                block_ratio=round(ratio, 3),
                threshold=self.settings.spike_block_threshold,
                msg="Sustained BLOCK-rate spike detected — alert dispatched to SIEM sinks",
            )
            return True
        if not over and self._in_spike:
            # Ratio recovered: re-arm the edge trigger.
            self._in_spike = False
        return False

    def _spike_payload(self) -> dict[str, Any]:
        window = len(self._recent_blocks)
        return self._envelope(
            "neuralguard.block_spike",
            {
                "alert": "block_rate_spike",
                "window_events": window,
                "block_count": self._block_count,
                "block_ratio": round(self._block_count / window, 4) if window else 0.0,
                "threshold": self.settings.spike_block_threshold,
            },
        )

    def _verdict_payload(self, event: AuditEvent) -> dict[str, Any]:
        return self._envelope("neuralguard.verdict", event.model_dump(mode="json"))

    @staticmethod
    def _envelope(event_type: str, event: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_type": event_type,
            "time": time.time(),
            "event": event,
        }

    # ── Delivery (bounded, best-effort) ───────────────────────────────────

    def _schedule(self, payload: dict[str, Any]) -> None:
        """Schedule delivery without ever blocking or raising the caller."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop (CLI / sync test context): bounded sync send.
            self._deliver_sync(payload)
            return
        task = loop.create_task(self._deliver_async(payload))
        # Keep a reference so the task is not garbage-collected mid-flight.
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _on_dropped(self, reason: str) -> None:
        self._drops += 1
        if self._drops % _DROP_LOG_EVERY == 1:
            logger.warning(
                "siem_event_dropped",
                reason=reason,
                total_dropped=self._drops,
                msg="SIEM delivery dropped an event (observability, not inline — verdicts unaffected)",
            )

    async def _deliver_async(self, payload: dict[str, Any]) -> None:
        if self._semaphore.locked():
            self._on_dropped("inflight_cap")
            return
        async with self._semaphore:
            timeout = self.settings.timeout_seconds
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=cast("httpx.AsyncBaseTransport | None", self._transport),
            ) as client:
                if "splunk_hec" in self._sinks:
                    await self._post_splunk(client, payload)
                if "webhook" in self._sinks:
                    await self._post_webhook(client, payload)
                if "scarletai" in self._sinks:
                    await self._post_scarletai(client, payload)

    def _deliver_sync(self, payload: dict[str, Any]) -> None:
        # Sync fallback path (no running loop): single bounded attempt.
        try:
            with httpx.Client(
                timeout=self.settings.timeout_seconds,
                transport=cast("httpx.BaseTransport | None", self._transport),
            ) as client:
                if "splunk_hec" in self._sinks:
                    self._post_splunk_sync(client, payload)
                if "webhook" in self._sinks:
                    self._post_webhook_sync(client, payload)
                if "scarletai" in self._sinks:
                    self._post_scarletai_sync(client, payload)
        except Exception as exc:
            self._on_dropped(f"sync_error: {exc.__class__.__name__}")

    # ── Sink implementations ─────────────────────────────────────────────

    def _post_scarletai_sync(self, client: httpx.Client, payload: dict[str, Any]) -> None:
        scarletai_url = self.settings.scarletai_url
        assert scarletai_url is not None  # "scarletai" in _sinks guarantees this
        headers = (
            {"Authorization": f"Bearer {self.settings.scarletai_token}"}
            if self.settings.scarletai_token
            else {}
        )
        # Sync fallback (no running loop — CLI / sync test context): no
        # buffering. The time-based flush needs an event loop, and immediate
        # delivery is strictly better in a one-shot CLI process.
        response = client.post(
            scarletai_url, json=map_to_scarletai_batch(payload, self.settings), headers=headers
        )
        self._check(response, "scarletai")

    async def _post_scarletai(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
        try:
            events = map_to_scarletai_batch(payload, self.settings)
            spike = payload.get("event_type") == "neuralguard.block_spike"
            if self._batching_enabled() and not spike:
                # Opt-in batching (scarletai sink only): verdict families
                # buffer and flush on size/time. Spike alerts BYPASS the
                # buffer — a critical alert must never wait a flush interval.
                await self._buffer_verdict(client, events)
            else:
                await self._post_scarletai_events(client, events)
        except Exception as exc:
            logger.warning("siem_sink_failed", sink="scarletai", error=str(exc))
            self._on_dropped(f"scarletai: {exc.__class__.__name__}")

    # ── Batch buffering (opt-in, scarletai sink only) ────────────────────

    def _batching_enabled(self) -> bool:
        """True when the buffer is on (cap > 1). Splunk/webhook never buffer."""
        return self.settings.scarletai_batch_max_events > 1

    async def _buffer_verdict(
        self, client: httpx.AsyncClient, events: list[dict[str, Any]]
    ) -> None:
        """Buffer one mapped event-array; flush when the buffer reaches the cap.

        The cap counts EVENTS (parent + companions), not arrays. Called while
        the caller holds the inflight semaphore, so a size-flush POST rides
        the caller's permit — the inflight cap stays the only concurrency
        bound. Between flushes the buffer holds at most the cap (overshoot
        bounded by one array — a family flushes together, never split).
        """
        flush: list[list[dict[str, Any]]] = []
        async with self._batch_lock:
            self._batch_buffer.append(events)
            if (
                sum(len(arr) for arr in self._batch_buffer)
                >= self.settings.scarletai_batch_max_events
            ):
                flush = self._batch_buffer
                self._batch_buffer = []
        self._ensure_flusher()
        if flush:
            await self._post_event_arrays(client, flush)

    def _ensure_flusher(self) -> None:
        """Start the time-based flusher lazily (first buffered verdict).

        Only meaningful in a running loop (the buffered path is async).
        Idempotent: a no-op when the flusher is alive.
        """
        if self._flusher_task is None or self._flusher_task.done():
            self._flusher_task = asyncio.get_running_loop().create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        interval = self.settings.scarletai_batch_flush_seconds
        while True:
            await asyncio.sleep(interval)
            await self._flush_from_loop()

    async def _flush_from_loop(self) -> None:
        """Drain + POST the buffer (flusher ticks / shutdown final flush).

        Never raises. When the inflight semaphore is exhausted the tick is
        SKIPPED (retry next interval) rather than dropping the batch — the
        buffer is bounded, so retrying is safe and no data is lost to
        transient contention. Deliveries keep the drop-on-cap semantics.
        """
        if not self._batch_buffer:  # lock-free fast path; a race is a no-op tick
            return
        if self._semaphore.locked():
            return  # busy: retry next tick, never queue behind a slow sink
        try:
            async with self._semaphore:
                async with self._batch_lock:
                    arrays = self._batch_buffer
                    self._batch_buffer = []
                if not arrays:
                    return  # pragma: no cover — concurrent-drainer race guard
                # (the lock-free check → lock drain sequence has no await
                # between them in-process; a second drainer cannot interleave.
                # Kept as a defensive invariant for future callers.)
                async with httpx.AsyncClient(
                    timeout=self.settings.timeout_seconds,
                    transport=cast("httpx.AsyncBaseTransport | None", self._transport),
                ) as client:
                    await self._post_event_arrays(client, arrays)
        except Exception as exc:
            logger.warning("siem_batch_flush_failed", error=str(exc))
            self._on_dropped(f"scarletai_batch: {exc.__class__.__name__}")

    @staticmethod
    def _chunk_arrays(arrays: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        """Split buffered families into POSTs of ≤ ScarletAI's 1000-event cap.

        Chunking respects ARRAY boundaries — a parent + companions family is
        never split across POSTs. Unreachable at configured sizes (cap ≤ 1000,
        families ≤ 3 events); pure defensive depth against the 413.
        """
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        count = 0
        for arr in arrays:
            if count and count + len(arr) > _SCARLETAI_BATCH_LIMIT:
                chunks.append(current)
                current, count = [], 0
            current.extend(arr)
            count += len(arr)
        if current:
            chunks.append(current)
        return chunks

    async def _post_event_arrays(
        self, client: httpx.AsyncClient, arrays: list[list[dict[str, Any]]]
    ) -> None:
        """POST buffered families (split ≤1000/POST); failures drop + count."""
        for chunk in self._chunk_arrays(arrays):
            await self._post_scarletai_events(client, chunk)

    async def shutdown_flush(self) -> None:
        """Best-effort final flush + flusher teardown (lifespan shutdown).

        Buffered events are LOST on a hard crash (no persistence) — the
        best-effort doctrine is documented on the batch knob. Never raises.
        """
        task = self._flusher_task
        self._flusher_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._flush_from_loop()

    async def _post_scarletai_events(
        self, client: httpx.AsyncClient, events: list[dict[str, Any]]
    ) -> None:
        """POST one mapped event-array (parent + companions) to ScarletAI.

        Companions ride the SAME POST as the parent by construction. The
        companion contract is Scarlet's closed vocabulary, so companions are
        scarletai-only; Splunk/webhook sinks keep the single-event envelope.
        """
        scarletai_url = self.settings.scarletai_url
        assert scarletai_url is not None  # "scarletai" in _sinks guarantees this
        headers = (
            {"Authorization": f"Bearer {self.settings.scarletai_token}"}
            if self.settings.scarletai_token
            else {}
        )
        response = await client.post(scarletai_url, json=events, headers=headers)
        self._check(response, "scarletai")

    # ── Sink implementations ─────────────────────────────────────────────

    def _post_splunk_sync(self, client: httpx.Client, payload: dict[str, Any]) -> None:
        hec_url = self.settings.splunk_hec_url
        assert hec_url is not None  # "splunk_hec" in _sinks guarantees this
        response = client.post(
            f"{hec_url}/services/collector/event",
            json={
                "time": payload["time"],
                "sourcetype": self.settings.splunk_source_type,
                "event": payload,
            },
            headers={"Authorization": f"Splunk {self.settings.splunk_hec_token}"},
        )
        self._check(response, "splunk_hec")

    async def _post_splunk(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
        try:
            hec_url = self.settings.splunk_hec_url
            assert hec_url is not None  # "splunk_hec" in _sinks guarantees this
            response = await client.post(
                f"{hec_url}/services/collector/event",
                json={
                    "time": payload["time"],
                    "sourcetype": self.settings.splunk_source_type,
                    "event": payload,
                },
                headers={"Authorization": f"Splunk {self.settings.splunk_hec_token}"},
            )
            self._check(response, "splunk_hec")
        except Exception as exc:
            logger.warning("siem_sink_failed", sink="splunk_hec", error=str(exc))
            self._on_dropped(f"splunk_hec: {exc.__class__.__name__}")

    def _post_webhook_sync(self, client: httpx.Client, payload: dict[str, Any]) -> None:
        webhook_url = self.settings.webhook_url
        assert webhook_url is not None  # "webhook" in _sinks guarantees this
        headers = (
            {"Authorization": f"Bearer {self.settings.webhook_token}"}
            if self.settings.webhook_token
            else {}
        )
        response = client.post(webhook_url, json=payload, headers=headers)
        self._check(response, "webhook")

    async def _post_webhook(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
        try:
            webhook_url = self.settings.webhook_url
            assert webhook_url is not None  # "webhook" in _sinks guarantees this
            headers = (
                {"Authorization": f"Bearer {self.settings.webhook_token}"}
                if self.settings.webhook_token
                else {}
            )
            response = await client.post(webhook_url, json=payload, headers=headers)
            self._check(response, "webhook")
        except Exception as exc:
            logger.warning("siem_sink_failed", sink="webhook", error=str(exc))
            self._on_dropped(f"webhook: {exc.__class__.__name__}")

    def _check(self, response: httpx.Response, sink: str) -> None:
        if response.status_code >= 400:
            # A rejected event never landed in the SIEM — count it as dropped.
            self._on_dropped(f"{sink}: http_{response.status_code}")
            logger.warning(
                "siem_sink_rejected",
                sink=sink,
                status=response.status_code,
                msg="SIEM sink rejected an event (token/endpoint misconfiguration?)",
            )
