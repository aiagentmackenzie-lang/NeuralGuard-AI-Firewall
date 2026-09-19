"""NG-7/NG-8: MCP transport — JSON-RPC passthrough to the upstream MCP server.

The MCP streamable-HTTP transport is a single endpoint that accepts POSTed
JSON-RPC messages and returns JSON responses (SSE streaming exists in the
spec; this gateway is request/response only in v1 — a streaming call is
refused, same fail-closed posture as the chat proxy's SSE hold-back).

The transport holds no state and injects no upstream auth by default (local
MCP servers are typically unauthenticated); an upstream Authorization header
is configured server-side via ``upstream_auth_token`` (Bearer scheme) and is
never logged — and a caller-supplied ``Authorization`` header can never reach
the upstream: the server-side token overwrites it.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


class McpUpstreamError(Exception):
    """The upstream MCP call failed (network, timeout, or non-2xx).

    The route layer converts this to a generic 502 — error details are
    logged, never returned to callers.
    """


class McpTransport:
    """Forwards JSON-RPC payloads to the configured MCP server endpoint."""

    def __init__(self, settings: Any, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=settings.timeout_seconds)

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    async def forward(
        self, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """POST one JSON-RPC message to the upstream MCP endpoint.

        Raises:
            McpUpstreamError: on connection failure, timeout, non-2xx, or
                non-JSON response.
        """
        base = str(self._settings.upstream_url).rstrip("/")
        forward_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if headers:
            forward_headers.update(headers)
        # Server-side upstream auth: the configured token OVERWRITES any
        # caller-supplied Authorization — a gateway client must never be able
        # to smuggle its own credentials upstream. The token is held
        # server-side and never logged (log lines carry url/status/body_len
        # only; httpx exception reprs carry the URL, never headers).
        token = str(getattr(self._settings, "upstream_auth_token", "") or "")
        if token:
            forward_headers["Authorization"] = f"Bearer {token}"

        try:
            response = await self._client.post(base, json=payload, headers=forward_headers)
        except httpx.TimeoutException as exc:
            logger.warning("mcp_upstream_timeout", url=base, error=repr(exc))
            raise McpUpstreamError("MCP upstream timed out") from exc
        except httpx.HTTPError as exc:
            logger.warning("mcp_upstream_error", url=base, error=repr(exc))
            raise McpUpstreamError("MCP upstream unreachable") from exc

        if response.status_code >= 400:
            logger.warning(
                "mcp_upstream_rejected",
                url=base,
                status=response.status_code,
                body_len=len(response.content),
            )
            raise McpUpstreamError(f"MCP upstream returned status {response.status_code}")

        try:
            data: dict[str, Any] = response.json()
            return data
        except ValueError as exc:
            logger.error("mcp_upstream_bad_json", url=base)
            raise McpUpstreamError("MCP upstream returned invalid JSON") from exc

    async def aclose(self) -> None:
        """Release the HTTP client if this transport owns it."""
        if self._owns_client:
            import contextlib

            with contextlib.suppress(Exception):
                await self._client.aclose()
