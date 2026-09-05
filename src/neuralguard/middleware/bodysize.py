"""Body-size middleware (pure ASGI, P2-8).

Caps the inbound request body (Content-Length based, with a bounded stream
read for chunked bodies). Pure ASGI so an unexpected exception propagates to
the app's global ``Exception`` handler — the BaseHTTPMiddleware conversion
(P2-8) exists precisely so the handler genuinely backstops every layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from neuralguard.metrics import metrics
from neuralguard.middleware.asgi import drain_body, replay_receive

if TYPE_CHECKING:
    from starlette.types import Receive, Send

logger = structlog.get_logger(__name__)


class BodySizeMiddleware:
    """Cap inbound request body size (Content-Length based)."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self._max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http" or self._max_bytes <= 0:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        # Only constrain body-bearing paths. Health/info have no body.
        if method in {"POST", "PUT", "PATCH"} and path.startswith("/v1/"):
            headers = Headers(scope=scope)
            content_length_header = headers.get("content-length")
            if content_length_header is not None:
                try:
                    declared = int(content_length_header)
                except ValueError:
                    await self._too_large(scope, receive, send, "Invalid Content-Length header")
                    return
                if declared > self._max_bytes:
                    metrics.record_body_rejection()
                    logger.warning(
                        "body_too_large",
                        path=path,
                        declared=declared,
                        limit=self._max_bytes,
                    )
                    await self._too_large(
                        scope,
                        receive,
                        send,
                        f"Request body exceeds limit of {self._max_bytes} bytes",
                    )
                    return

            # Chunked transfer encoding has no Content-Length. Cap by reading
            # the stream up to the limit and re-injecting a bounded body so
            # the downstream parser never sees more than max_bytes+1.
            transfer_encoding = headers.get("transfer-encoding", "").lower()
            if "chunked" in transfer_encoding or content_length_header is None:
                body = await drain_body(receive)
                if len(body) > self._max_bytes:
                    metrics.record_body_rejection()
                    await self._too_large(
                        scope,
                        receive,
                        send,
                        f"Request body exceeds limit of {self._max_bytes} bytes",
                    )
                    return

                # Replay the drained body for downstream consumers.
                receive = replay_receive(body, receive)

        await self.app(scope, receive, send)

    async def _too_large(
        self,
        scope: dict[str, Any],
        receive: Receive,
        send: Send,
        detail: str,
    ) -> None:
        response = JSONResponse(
            status_code=413,
            content={"error": "payload_too_large", "message": detail},
        )
        await response(scope, receive, send)
