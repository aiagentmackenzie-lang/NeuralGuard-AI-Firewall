"""Contracts for the Postgres audit-backend init path and its readiness report.

Both were found by re-running the Kubernetes cluster drill against the
**published** v0.3.0 image on 2026-11-17, not by reading the code:

1. **The boot-order race bricked the durable audit path permanently.** The
   neuralguard pods started before the `postgres` Service had endpoints, so the
   lifespan's one-shot ``create_all`` failed with ``[Errno -2] Name or service
   not known``. It logged ``db_init_failed`` once and fell back to JSONL — and
   never retried. Nine evaluate() calls later the connection worked and every
   audit INSERT failed with ``relation "audit_events" does not exist``; only a
   ``rollout restart`` fixed it. A dependency that is briefly late must not
   disable a configured backend for the lifetime of the pod.
2. **Readiness said the audit DB was fine.** ``/v1/ready`` reported
   ``audit_db: "ok"`` and aggregate ``status: "healthy"`` while nothing was
   being persisted, because the probe was ``SELECT 1`` — connectivity, never the
   schema the writes need. A firewall that sells tamper-evident audit must not
   report a healthy audit store it cannot write to.

The contracts are written so a *correct* implementation passes and a wrong one
fails: they count attempts, they assert the schema is what the probe checks, and
they assert the aggregate status is no longer "healthy" in the broken state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]


class _FakeResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar(self) -> Any:
        return self._value


class _FakeConn:
    """Async connection whose ``to_regclass`` answer is scripted."""

    def __init__(self, schema_present: bool) -> None:
        self._schema_present = schema_present
        self.queries: list[str] = []

    async def execute(self, stmt: Any) -> _FakeResult:
        sql = str(getattr(stmt, "text", stmt))
        self.queries.append(sql)
        if "to_regclass" in sql:
            return _FakeResult("audit_events" if self._schema_present else None)
        return _FakeResult(1)

    async def run_sync(self, fn: Any) -> None:
        """create_all() goes through run_sync; the fake schema is always valid."""
        return None

    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeEngine:
    def __init__(self, schema_present: bool = True) -> None:
        self.conn = _FakeConn(schema_present)

    def connect(self) -> _FakeConn:
        return self.conn


class TestInitRetriesALateDependency:
    """A late Postgres must be waited for, not read as 'JSONL forever'."""

    async def test_init_retries_until_the_dependency_resolves(self) -> None:
        from neuralguard.db.init import initialise_postgres

        calls: list[int] = []
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        class _Engine:
            def begin(self) -> Any:
                return _FakeBegin()

        class _FakeBegin:
            async def __aenter__(self) -> _FakeConn:
                return _FakeConn(True)

            async def __aexit__(self, *exc: object) -> None:
                return None

        def factory(url: str) -> Any:
            calls.append(1)
            if len(calls) < 3:
                # What asyncpg actually raised in the drill: DNS not yet resolvable.
                raise OSError("[Errno -2] Name or service not known")
            return _Engine()

        result = await initialise_postgres(
            "postgresql+asyncpg://u:p@postgres:5432/neuralguard",
            engine_factory=factory,
            sleep=fake_sleep,
        )

        assert result.ok is True, "a transient DNS failure at boot must not disable postgres audit"
        assert result.attempts == 3, f"expected 3 attempts, got {result.attempts}"
        assert len(sleeps) == 2, "each retry must wait (no hot loop against a down dependency)"

    async def test_init_gives_up_on_a_budget_and_reports_the_error(self) -> None:
        from neuralguard.db.init import initialise_postgres

        async def fake_sleep(seconds: float) -> None:
            return None

        def always_fails(url: str) -> Any:
            raise OSError("[Errno -2] Name or service not known")

        result = await initialise_postgres(
            "postgresql+asyncpg://u:p@postgres:5432/neuralguard",
            engine_factory=always_fails,
            attempts=4,
            sleep=fake_sleep,
        )

        assert result.ok is False
        assert result.attempts == 4, "the attempt budget must be honoured, not exceeded"
        assert result.error and "Name or service not known" in result.error, (
            "the failure must keep the underlying error; 'init failed' alone is not diagnosable"
        )


class TestReadinessDoesNotClaimABrokenAuditStore:
    async def test_schema_missing_is_reported_as_fail_not_ok(self) -> None:
        from neuralguard.api.readiness import _probe_db
        from neuralguard.config.settings import AuditSettings, NeuralGuardConfig

        cfg = NeuralGuardConfig(
            audit=AuditSettings(backend="postgres", postgres_url="postgresql+asyncpg://x"),
        )
        engine = _FakeEngine(schema_present=False)

        status = await _probe_db(cfg, engine=engine)  # type: ignore[arg-type]

        assert status == "fail", (
            "a configured postgres audit backend whose table is missing cannot persist "
            "anything, so readiness must not call it ok"
        )
        assert any("to_regclass" in q for q in engine.conn.queries), (
            "the probe must check for the schema the writes need, not merely connectivity"
        )

    async def test_schema_present_is_reported_as_ok(self) -> None:
        from neuralguard.api.readiness import _probe_db
        from neuralguard.config.settings import AuditSettings, NeuralGuardConfig

        cfg = NeuralGuardConfig(
            audit=AuditSettings(backend="postgres", postgres_url="postgresql+asyncpg://x"),
        )
        status = await _probe_db(cfg, engine=_FakeEngine(schema_present=True))  # type: ignore[arg-type]
        assert status == "ok"

    async def test_aggregate_status_is_not_healthy_with_a_broken_audit_store(self) -> None:
        from neuralguard.api.readiness import check_readiness
        from neuralguard.config.settings import AuditSettings, NeuralGuardConfig

        # Covered end-to-end by the existing readiness fixtures; the invariant
        # asserted here is the one that mattered in the drill: 'healthy' must be
        # unreachable while a configured backend cannot persist.
        from tests.unit.test_readiness import _pipeline_with_core, _request

        cfg = NeuralGuardConfig(
            audit=AuditSettings(backend="postgres", postgres_url="postgresql+asyncpg://x"),
        )
        req = _request(cfg, _pipeline_with_core())
        r = await check_readiness(req)
        assert r.components["audit_db"] != "ok", (
            "with no engine at all, and a postgres backend configured, the audit store "
            "cannot be reported working"
        )
        assert r.status != "healthy", (
            "aggregate status 'healthy' alongside a non-working audit store is the exact "
            "lie this contract exists to prevent"
        )


def test_readiness_probe_signature_carries_an_injectable_engine() -> None:
    """The probe must be testable without a live Postgres (this is a unit suite)."""
    import inspect

    from neuralguard.api.readiness import _probe_db

    params = inspect.signature(_probe_db).parameters
    assert "engine" in params, (
        "_probe_db(config, engine=...) — an un-injectable global makes the schema check "
        "untestable outside an integration run with a live Postgres"
    )


class TestNoStaleDocsClaimOnInit:
    """The documented fallback behaviour must match the retrying implementation."""

    def test_main_uses_the_retrying_initialiser(self) -> None:
        src = (_REPO / "src" / "neuralguard" / "main.py").read_text(encoding="utf-8")
        assert "initialise_postgres" in src, (
            "the lifespan must call db.init.initialise_postgres (retrying) instead of an "
            "inline one-shot create_all that a late dependency defeats"
        )


class TestAttemptBudgetIsConfigurable:
    """The retry budget must be an operator knob, not a constant in the source.

    A slow StatefulSet/PVC attach and a deliberately-bad URL in a test both go
    through the same path, so the budget has to be settable — the 2026-11-17 fix
    first hardcoded it, and three lifespan tests then each spent 23 s waiting on
    a database that was never going to appear.
    """

    def test_setting_exists_with_a_default_and_is_passed_through(self) -> None:
        from neuralguard.config.settings import AuditSettings

        assert AuditSettings().init_attempts == 6, (
            "the default budget is the production posture; tests and slow clusters "
            "override it explicitly"
        )
        main_src = (_REPO / "src" / "neuralguard" / "main.py").read_text(encoding="utf-8")
        assert "attempts=config.audit.init_attempts" in main_src, (
            "the lifespan must pass the configured budget — a hardcoded value silently "
            "ignores NEURALGUARD_AUDIT_INIT_ATTEMPTS"
        )

    async def test_one_attempt_means_no_sleep_at_all(self) -> None:
        # Async, not asyncio.run(): run() tears down the current event loop, which
        # breaks later sync tests that call asyncio.get_event_loop() (caught when
        # tests/unit/test_db.py started failing only in a full-suite run).
        from neuralguard.db.init import initialise_postgres

        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        def always_fails(url: str) -> object:
            raise OSError("Name or service not known")

        await initialise_postgres(
            "postgresql+asyncpg://u:p@postgres:5432/db",
            engine_factory=always_fails,
            attempts=1,
            sleep=fake_sleep,
        )
        assert slept == [], "attempts=1 must fail fast (the documented 'set 1 to fail fast')"
