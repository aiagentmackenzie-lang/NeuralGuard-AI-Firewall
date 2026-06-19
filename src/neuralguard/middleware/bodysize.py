"""Request body size limiting middleware.

Rejects requests whose declared Content-Length (or, for chunked transfers,
accumulated body) exceeds the configured cap BEFORE the JSON parser buffers
the entire payload. This prevents oversized-body memory exhaustion at the
framework layer (the structural scanner's max_input_length only fires after
the body is already parsed).

Returns 413 Payload Too Large with a JSON error body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from neuralguard.metrics import metrics

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

logger = structlog.get_logger(__name__)


class BodySizeMiddleware(BaseHTTPMiddleware):
    """Cap inbound request body size (Content-Length based)."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if self._max_bytes <= 0:
            return await call_next(request)

        # Only constrain body-bearing paths. Health/info have no body.
        if request.method in {"POST", "PUT", "PATCH"} and request.url.path.startswith("/v1/"):
            content_length_header = request.headers.get("content-length")
            if content_length_header is not None:
                try:
                    declared = int(content_length_header)
                except ValueError:
                    return self._too_large("Invalid Content-Length header")
                if declared > self._max_bytes:
                    metrics.record_body_rejection()
                    logger.warning(
                        "body_too_large",
                        path=request.url.path,
                        declared=declared,
                        limit=self._max_bytes,
                    )
                    return self._too_large(f"Request body exceeds limit of {self._max_bytes} bytes")

            # Chunked transfer encoding has no Content-Length. Cap by reading
            # the stream up to the limit and re-injecting a bounded body so the
            # downstream parser never sees more than max_bytes+1.
            transfer_encoding = request.headers.get("transfer-encoding", "").lower()
            if "chunked" in transfer_encoding or content_length_header is None:
                body = await self._read_bounded(request)
                if len(body) > self._max_bytes:
                    metrics.record_body_rejection()
                    return self._too_large(f"Request body exceeds limit of {self._max_bytes} bytes")

                # Re-inject the consumed body for downstream consumers.
                async def receive() -> dict[str, Any]:
                    return {"type": "http.request", "body": body, "more_body": False}

                request._receive = receive  # re-inject bounded body

        return await call_next(request)

    async def _read_bounded(self, request: Request) -> bytes:
        """Read up to max_bytes+1 bytes from the request stream."""
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            chunks.append(chunk)
            if total > self._max_bytes:
                break
        return b"".join(chunks)

    @staticmethod
    def _too_large(detail: str) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={"error": "payload_too_large", "message": detail},
            headers={"Retry-After": "0"},
        )
