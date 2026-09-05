"""Upstream forwarder (F9): sends ALLOWed chat payloads to the OpenAI-compatible
upstream and returns the raw response.

The forwarder holds no state; auth is injected per call from ProxySettings
(the operator's upstream key — server-side only). Errors surface as a
generic :class:`UpstreamError` so routes never leak upstream internals to
callers. The client is injectable for hermetic tests.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


class UpstreamError(Exception):
    """The upstream call failed (network, timeout, or non-2xx status).

    The route layer converts this to a generic 502 — the error detail is
    logged, never returned to the caller.
    """


class UpstreamForwarder:
    """Forwards OpenAI-format chat payloads to the configured upstream."""

    def __init__(self, settings: Any, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=settings.timeout_seconds)

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    async def forward_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST the payload to ``{upstream_url}/chat/completions``.

        Raises:
            UpstreamError: on connection failure, timeout, or non-2xx.
        """
        base = str(self._settings.upstream_url).rstrip("/")
        url = f"{base}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = self._settings.upstream_api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = await self._client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning("proxy_upstream_timeout", url=url, error=repr(exc))
            raise UpstreamError("upstream timed out") from exc
        except httpx.HTTPError as exc:
            logger.warning("proxy_upstream_error", url=url, error=repr(exc))
            raise UpstreamError("upstream unreachable") from exc

        if response.status_code >= 400:
            # Upstream rejected the call: log detail server-side, return generic.
            logger.warning(
                "proxy_upstream_rejected",
                url=url,
                status=response.status_code,
                body_len=len(response.content),
            )
            raise UpstreamError(f"upstream returned status {response.status_code}")

        try:
            data: dict[str, Any] = response.json()
            return data
        except ValueError as exc:
            logger.error("proxy_upstream_bad_json", url=url)
            raise UpstreamError("upstream returned invalid JSON") from exc

    async def aclose(self) -> None:
        """Release the HTTP client if this forwarder owns it."""
        if self._owns_client:
            import contextlib

            with contextlib.suppress(Exception):
                await self._client.aclose()
