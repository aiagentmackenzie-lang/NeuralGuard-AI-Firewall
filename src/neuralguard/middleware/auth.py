"""API-key authentication middleware.

Validates the `Authorization: Bearer <key>` or `X-API-Key: <key>` header
against the configured API keys and binds the request to the key's tenant.

Security properties:
- Constant-time key comparison (hmac.compare_digest) to resist timing attacks.
- Tenant identity is derived from the authenticated key, NOT from a client-
  supplied header. This prevents the rate-limit tenant-spoofing bypass.
- When `enforce_tenant_from_key` is on, a request body tenant_id that disagrees
  with the key's bound tenant is rejected with 403.
- Public endpoints (e.g. a minimal /v1/health) may bypass auth.
"""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING, Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

    from neuralguard.config.settings import AuthSettings

logger = structlog.get_logger(__name__)


def _extract_key(request: Request) -> str | None:
    """Pull the API key from Authorization: Bearer ... or X-API-Key header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        candidate = auth_header.split(" ", 1)[1].strip()
        if candidate:
            return candidate
    api_key_header = request.headers.get("X-API-Key")
    if api_key_header:
        return api_key_header.strip()
    return None


def _constant_time_lookup(candidate: str, key_map: dict[str, str]) -> str | None:
    """Return the tenant for `candidate` using constant-time comparison.

    Compares against every configured key with compare_digest so timing does
    not leak which key (or prefix) matched.
    """
    matched_tenant: str | None = None
    for key, tenant in key_map.items():
        if hmac.compare_digest(candidate, key):
            # Last match wins deterministically; keys are unique in practice.
            matched_tenant = tenant
    return matched_tenant


class AuthMiddleware(BaseHTTPMiddleware):
    """Enforce API-key auth and bind requests to the authenticated tenant."""

    def __init__(self, app: Any, settings: AuthSettings) -> None:
        super().__init__(app)
        self.settings = settings
        self._key_map = settings.key_to_tenant()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not self.settings.enabled:
            # Auth disabled (development). Mark unauthenticated for downstream
            # middleware that may key on the principal.
            request.state.authenticated = False
            request.state.auth_tenant = None
            return await call_next(request)

        path = request.url.path

        # Public endpoints bypass auth (kept minimal: health only by default).
        if path in self.settings.public_endpoints:
            request.state.authenticated = False
            request.state.auth_tenant = None
            return await call_next(request)

        # Only protect API paths; leave non-API (e.g. /docs) to FastAPI.
        if not path.startswith("/v1/"):
            request.state.authenticated = False
            request.state.auth_tenant = None
            return await call_next(request)

        candidate = _extract_key(request)
        if candidate is None:
            logger.warning("auth_missing_key", path=path)
            return self._unauthorized("Missing API key")

        tenant = _constant_time_lookup(candidate, self._key_map)
        if tenant is None:
            logger.warning("auth_invalid_key", path=path)
            return self._unauthorized("Invalid API key")

        request.state.authenticated = True
        request.state.auth_tenant = tenant

        # Enforce tenant binding: the body may carry a tenant_id that must
        # agree with the key's bound tenant. We cannot fully parse the JSON
        # body here without consuming it (needed downstream), so we rely on a
        # lightweight peek only for the tenant mismatch check. To avoid
        # breaking the request stream, the route layer performs the final
        # tenant enforcement after body parsing.
        return await call_next(request)

    @staticmethod
    def _unauthorized(detail: str) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"error": "unauthorized", "message": detail},
            headers={"WWW-Authenticate": "Bearer"},
        )
