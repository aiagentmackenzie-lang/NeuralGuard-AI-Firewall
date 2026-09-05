"""P2-8: the global exception handler genuinely backstops the middleware stack.

Pre-conversion (BaseHTTPMiddleware), an unexpected exception raised INSIDE a
custom middleware bypassed the app-level ``@app.exception_handler(Exception)``
and surfaced as a bare server error without a correlation id — the reason
routes had to self-wrap in ``_internal_error``. With the pure-ASGI conversion,
an exception raised in middleware propagates into the handler and comes back
as the sanitized, correlation-id'd 500.
"""

from __future__ import annotations

import re

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.config.settings import (
    AuthSettings,
    NeuralGuardConfig,
    RateLimitSettings,
    ServerSettings,
)
from neuralguard.main import create_app


@pytest.mark.asyncio
async def test_middleware_exception_hits_global_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """A boom inside AuthMiddleware → sanitized 500 WITH correlation_id."""
    config = NeuralGuardConfig(
        environment="development",
        auth=AuthSettings(enabled=True, api_keys=["ng_test_key|default"]),
        rate_limit=RateLimitSettings(enabled=True, backend="memory"),
        server=ServerSettings(max_request_body_bytes=1048576),
    )
    app = create_app(config)

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("middleware blew up")

    # Break the runtime-state lookup to simulate an unexpected failure INSIDE
    # the middleware (not in a route).
    for mw in app.user_middleware:
        pass  # user_middleware entries are factories; patch the built stack below

    # Walk the built ASGI stack to find the AuthMiddleware instance.
    # FastAPI builds the ASGI stack lazily — force it, then walk the chain.
    app.middleware_stack = app.build_middleware_stack()
    node = app.middleware_stack
    auth_mw = None
    seen = 0
    while node is not None and seen < 20:
        if type(node).__name__ == "AuthMiddleware":
            auth_mw = node
            break
        node = getattr(node, "app", None)
        seen += 1
    assert auth_mw is not None, "AuthMiddleware not found in the built stack"
    monkeypatch.setattr(type(auth_mw._state), "lookup", boom, raising=True)

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/v1/evaluate",
            json={"prompt": "hello", "tenant_id": "default"},
            headers={"Authorization": "Bearer ng_test_key"},
        )
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "internal_error"
    assert "middleware blew up" not in r.text  # no internals leaked
    assert re.fullmatch(r"[0-9a-f-]{36}", body.get("correlation_id", "")), (
        "500 must carry a correlation id"
    )


@pytest.mark.asyncio
async def test_ratelimit_middleware_exception_hits_global_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A boom inside RateLimitMiddleware → sanitized 500 WITH correlation_id."""
    config = NeuralGuardConfig(
        environment="development",
        auth=AuthSettings(enabled=True, api_keys=["ng_test_key|default"]),
        rate_limit=RateLimitSettings(enabled=True, backend="memory"),
        server=ServerSettings(max_request_body_bytes=1048576),
    )
    app = create_app(config)

    app.middleware_stack = app.build_middleware_stack()
    node = app.middleware_stack
    rl_mw = None
    seen = 0
    while node is not None and seen < 20:
        if type(node).__name__ == "RateLimitMiddleware":
            rl_mw = node
            break
        node = getattr(node, "app", None)
        seen += 1
    assert rl_mw is not None, "RateLimitMiddleware not found in the built stack"
    monkeypatch.setattr(type(rl_mw._counter), "check", _boom_async, raising=True)

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/v1/evaluate",
            json={"prompt": "hello", "tenant_id": "default"},
            headers={"Authorization": "Bearer ng_test_key"},
        )
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "internal_error"
    assert re.fullmatch(r"[0-9a-f-]{36}", body.get("correlation_id", ""))


async def _boom_async(*args: object, **kwargs: object) -> tuple[bool, int, int]:
    raise RuntimeError("ratelimit blew up")
