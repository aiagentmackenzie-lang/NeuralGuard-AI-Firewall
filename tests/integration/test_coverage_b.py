"""Coverage-focused tests for Phase B hardening paths."""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import (
    AuditSettings,
    AuthSettings,
    NeuralGuardConfig,
    ServerSettings,
)
from neuralguard.logging.audit import AuditLogger
from neuralguard.main import create_app
from neuralguard.middleware.bodysize import BodySizeMiddleware
from neuralguard.models.schemas import EvaluateRequest, Verdict


def _fake_receive(body: bytes):
    sent = {"done": False}

    async def receive():
        if sent["done"]:
            return {"type": "http.disconnect"}
        sent["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


class TestBodySizeChunkedPath:
    """Cover the no-Content-Length / chunked branch of BodySizeMiddleware."""

    @pytest.mark.asyncio
    async def test_no_content_length_under_limit_passes(self):
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse

        async def ok_endpoint(request):
            body = await request.body()
            return PlainTextResponse(f"got:{len(body)}")

        app = Starlette()
        app.add_middleware(BodySizeMiddleware, max_bytes=1024)
        app.add_route("/v1/x", ok_endpoint, methods=["POST"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            # Send body with no Content-Length by using content=bytes
            r = await c.post("/v1/x", content=b"hello world")
            assert r.status_code == 200
            assert "got:11" in r.text

    @pytest.mark.asyncio
    async def test_no_content_length_over_limit_413(self):
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse

        async def ok_endpoint(request):
            await request.body()
            return PlainTextResponse("ok")

        app = Starlette()
        app.add_middleware(BodySizeMiddleware, max_bytes=16)
        app.add_route("/v1/x", ok_endpoint, methods=["POST"])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.post("/v1/x", content=b"A" * 200)
            assert r.status_code == 413


class TestAuditInflightOverflow:
    @pytest.mark.asyncio
    async def test_overflow_falls_back_to_jsonl(self, tmp_path):
        logger = AuditLogger(
            AuditSettings(
                backend="postgres",
                jsonl_path=tmp_path / "audit",
                max_inflight_writes=1,
            )
        )

        async def _drive():
            import uuid as _uuid

            from neuralguard.models.schemas import AuditEvent

            # Build a minimal event and persist twice; second should overflow to JSONL.
            ev = AuditEvent(
                event_id=str(_uuid.uuid4()),
                request_id=str(_uuid.uuid4()),
                tenant_id="t",
                verdict=Verdict.ALLOW,
                findings_count=0,
                threat_categories=[],
                confidence=0.0,
                total_latency_ms=1.0,
            )
            # Force the postgres path; engine is None so it falls back to JSONL
            # before reaching inflight logic. To exercise overflow we patch
            # get_engine to return a sentinel and the ORM insert to a no-op.
            logger._pg_checked = True
            logger._pg_available = True

            # Monkeypatch the inner async insert to never touch a real DB.
            async def _fake_insert(orm_obj):
                await asyncio.sleep(0.01)

            logger._async_insert = staticmethod(_fake_insert)  # type: ignore[assignment]

            # Make get_engine return non-None so we reach the inflight branch.
            import neuralguard.db.engine as eng

            original_get = eng.get_engine
            eng.get_engine = lambda: object()  # truthy sentinel
            try:
                logger._persist_postgres(ev)
                logger._persist_postgres(ev)  # second should overflow
            finally:
                eng.get_engine = original_get
                # the first task may be pending; let it flush
                await asyncio.sleep(0)

        await asyncio.wait_for(_drive(), timeout=5)
        # Overflow counter incremented and JSONL written
        assert logger._dropped_overflow >= 1


class TestMainLifespanBranches:
    @pytest.mark.asyncio
    async def test_production_tls_notice_logged_when_insecure_allowed(self, capsys):
        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=True),
        )
        app = create_app(config)
        async with app.router.lifespan_context(app):
            pass
        out = capsys.readouterr().out
        assert "production_insecure_http_allowed" in out or "production_tls_notice" in out

    @pytest.mark.asyncio
    async def test_production_tls_notice_when_not_allowed(self, capsys):
        config = NeuralGuardConfig(
            environment="production",
            auth=AuthSettings(enabled=True, api_keys=["k|acme"]),
            server=ServerSettings(allow_insecure_http=False),
        )
        app = create_app(config)
        async with app.router.lifespan_context(app):
            pass
        assert "production_tls_notice" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_shutdown_disposes_postgres_engine(self):
        from neuralguard.db.engine import create_engine, dispose_engine, get_engine

        config = NeuralGuardConfig(
            environment="development",
            audit=AuditSettings(
                backend="postgres",
                postgres_url="postgresql+asyncpg://u:p@localhost:5432/db",
            ),
        )
        app = create_app(config)
        # Pre-create engine so the shutdown branch runs.
        create_engine(config.audit.postgres_url or "")
        assert get_engine() is not None
        async with app.router.lifespan_context(app):
            pass
        assert get_engine() is None
        await dispose_engine()  # idempotent


class TestBodySizeDirectASGI:
    """Directly exercise the no-Content-Length branch — pure-ASGI interface (P2-8)."""

    @pytest.mark.asyncio
    async def test_no_cl_over_limit_short_circuits_413(self):
        from starlette.responses import Response

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/x",
            "raw_path": b"/v1/x",
            "query_string": b"",
            "headers": [],  # no content-length, no transfer-encoding
        }
        body = b"A" * 200

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        statuses: list[int] = []

        async def send(message):
            if message["type"] == "http.response.start":
                statuses.append(message["status"])

        forwarded = {"called": False}

        async def app(scope, receive, send):
            forwarded["called"] = True
            resp = Response("ok", status_code=200)
            await resp(scope, receive, send)

        mw = BodySizeMiddleware(app=app, max_bytes=16)
        await mw(scope, receive, send)
        assert statuses == [413]
        assert not forwarded["called"]  # short-circuited before the app

    @pytest.mark.asyncio
    async def test_no_cl_under_limit_forwards(self):
        from starlette.responses import Response

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/x",
            "raw_path": b"/v1/x",
            "query_string": b"",
            "headers": [],
        }
        body = b"hi"

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        statuses: list[int] = []
        bodies: list[bytes] = []

        async def send(message):
            if message["type"] == "http.response.start":
                statuses.append(message["status"])
            elif message["type"] == "http.response.body":
                bodies.append(message.get("body", b""))

        async def app(inner_scope, inner_receive, inner_send):
            # The drained body must be replayed for downstream consumers.
            from starlette.requests import Request

            req = Request(inner_scope, inner_receive)
            resp = Response(await req.body(), status_code=200)
            await resp(inner_scope, inner_receive, inner_send)

        mw = BodySizeMiddleware(app=app, max_bytes=64)
        await mw(scope, receive, send)
        assert statuses == [200]
        assert b"".join(bodies) == b"hi"  # body survived the drain+replay
