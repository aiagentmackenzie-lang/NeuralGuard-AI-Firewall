"""Shared pure-ASGI helpers (P2-8).

The custom middleware stack (bodysize / auth / ratelimit) is pure ASGI so an
unexpected exception propagates to the app's global ``Exception`` handler
(the documented Starlette issue: exceptions raised inside
``BaseHTTPMiddleware`` bypass it). These helpers support the conversion.
"""

from __future__ import annotations

from typing import Any

from starlette.datastructures import Headers
from starlette.types import Message, Receive, Send

__all__ = ["Message", "Receive", "Send", "content_length", "drain_body", "replay_receive"]


def content_length(scope: dict[str, Any]) -> int | None:
    """Declared Content-Length from the raw ASGI header list, or None.

    Returns None both when the header is absent and when it is unparseable —
    callers distinguish via :func:`raw_content_length` where it matters.
    """
    headers = Headers(scope=scope)
    raw = headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def drain_body(receive: Receive) -> bytes:
    """Drain the ASGI request body from ``receive`` (bounded by the caller)."""
    body = b""
    while True:
        message = await receive()
        if message["type"] == "http.request":
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        elif message["type"] == "http.disconnect":
            break
    return body


def content_length_raw(scope: dict[str, Any]) -> str | None:
    """Raw Content-Length header value (unparseable stays raw), or None."""
    return Headers(scope=scope).get("content-length")


def replay_receive(body: bytes, receive: Receive) -> Receive:
    """Wrap ``receive`` so the drained ``body`` can be read again downstream.

    The first read returns the whole body in a single ``http.request``
    message; subsequent reads fall through to the original ``receive``
    (which, after a full drain, yields disconnects). Each middleware that
    needs the body wraps once — the wrapper chains cleanly.
    """
    sent = {"done": False}

    async def receive_() -> Message:
        if not sent["done"]:
            sent["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return receive_
