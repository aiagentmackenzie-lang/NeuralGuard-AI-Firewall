"""NeuralGuard Prometheus metrics.

A single registry with counters/histograms for SOC observability:
- request verdicts (allow/block/sanitize/escalate/quarantine/rate_limit)
- per-scanner latency histograms
- judge timeouts and circuit-breaker state
- audit persistence failures
- body-size rejections and auth rejections

Exposed at GET /v1/metrics (auth-protected in production).
"""

from __future__ import annotations

from typing import Any

# Prometheus client is an optional dependency. If unavailable, metrics are
# no-ops so the service still runs. Install with `pip install prometheus-client`.
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROM_AVAILABLE = False


class Metrics:
    """Singleton metrics holder. No-ops when prometheus_client is absent."""

    def __init__(self) -> None:
        self.available = _PROM_AVAILABLE
        if not _PROM_AVAILABLE:
            return
        # Use a fresh registry so multiple app instances in tests don't clash.
        self.registry: CollectorRegistry = CollectorRegistry()
        self.verdicts: Counter = Counter(
            "neuralguard_verdicts_total",
            "Total verdicts by type",
            labelnames=("verdict",),
            registry=self.registry,
        )
        self.scanner_latency: Histogram = Histogram(
            "neuralguard_scanner_latency_seconds",
            "Per-scanner execution latency",
            labelnames=("layer",),
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )
        self.pipeline_latency: Histogram = Histogram(
            "neuralguard_pipeline_latency_seconds",
            "Total pipeline latency",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )
        self.judge_timeouts: Counter = Counter(
            "neuralguard_judge_timeouts_total",
            "LLM-judge call timeouts",
            registry=self.registry,
        )
        self.judge_calls: Counter = Counter(
            "neuralguard_judge_calls_total",
            "LLM-judge calls invoked",
            registry=self.registry,
        )
        self.circuit_open: Gauge = Gauge(
            "neuralguard_judge_circuit_open",
            "1 if judge circuit breaker is open, else 0",
            registry=self.registry,
        )
        self.audit_failures: Counter = Counter(
            "neuralguard_audit_persist_failures_total",
            "Audit persistence failures",
            labelnames=("backend",),
            registry=self.registry,
        )
        self.auth_rejections: Counter = Counter(
            "neuralguard_auth_rejections_total",
            "Authentication rejections",
            labelnames=("reason",),
            registry=self.registry,
        )
        self.body_rejections: Counter = Counter(
            "neuralguard_body_too_large_total",
            "Request bodies rejected for exceeding size limit",
            registry=self.registry,
        )
        self.rate_limit_hits: Counter = Counter(
            "neuralguard_rate_limit_hits_total",
            "Requests rejected by rate limiter",
            registry=self.registry,
        )

    # No-op-safe recording helpers -------------------------------------------

    def record_verdict(self, verdict: str) -> None:
        if self.available:
            self.verdicts.labels(verdict=verdict).inc()

    def observe_scanner(self, layer: str, seconds: float) -> None:
        if self.available:
            self.scanner_latency.labels(layer=layer).observe(seconds)

    def observe_pipeline(self, seconds: float) -> None:
        if self.available:
            self.pipeline_latency.observe(seconds)

    def record_judge_timeout(self) -> None:
        if self.available:
            self.judge_timeouts.inc()

    def record_judge_call(self) -> None:
        if self.available:
            self.judge_calls.inc()

    def set_circuit_open(self, open_: bool) -> None:
        if self.available:
            self.circuit_open.set(1 if open_ else 0)

    def record_audit_failure(self, backend: str) -> None:
        if self.available:
            self.audit_failures.labels(backend=backend).inc()

    def record_auth_rejection(self, reason: str) -> None:
        if self.available:
            self.auth_rejections.labels(reason=reason).inc()

    def record_body_rejection(self) -> None:
        if self.available:
            self.body_rejections.inc()

    def record_rate_limit_hit(self) -> None:
        if self.available:
            self.rate_limit_hits.inc()

    def expose(self) -> tuple[bytes, str]:
        """Return (payload_bytes, content_type) for the /metrics endpoint."""
        if not self.available:
            return (
                b"# prometheus_client not installed; metrics disabled\n",
                "text/plain; version=0.0.4",
            )
        return generate_latest(self.registry), "text/plain; version=0.0.4; charset=utf-8"


# Module-level singleton. Metrics holds no request state so a single instance
# is fine across the application.
metrics = Metrics()


def record(value: Any, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
    """Placeholder kept for import stability; prefer methods on `metrics`."""
    _ = (value, args, kwargs)
