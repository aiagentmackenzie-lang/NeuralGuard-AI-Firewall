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
        response = client.post(
            scarletai_url, json=[map_to_scarletai(payload, self.settings)], headers=headers
        )
        self._check(response, "scarletai")

    async def _post_scarletai(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
        try:
            scarletai_url = self.settings.scarletai_url
            assert scarletai_url is not None  # "scarletai" in _sinks guarantees this
            headers = (
                {"Authorization": f"Bearer {self.settings.scarletai_token}"}
                if self.settings.scarletai_token
                else {}
            )
            response = await client.post(
                scarletai_url,
                json=[map_to_scarletai(payload, self.settings)],
                headers=headers,
            )
            self._check(response, "scarletai")
        except Exception as exc:
            logger.warning("siem_sink_failed", sink="scarletai", error=str(exc))
            self._on_dropped(f"scarletai: {exc.__class__.__name__}")

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
