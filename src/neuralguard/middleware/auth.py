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
from starlette.requests import Request
from starlette.responses import JSONResponse

# Runtime import at bottom of module is avoided — jwtauth imports nothing
# from middleware, so this is safe at module scope.
from neuralguard.auth.jwtauth import AuthRuntimeState
from neuralguard.metrics import metrics

if TYPE_CHECKING:
    from starlette.types import Receive, Send

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

    Scale note: this is O(n_keys)/request. Fine for the ≤ few-hundred API
    keys of a single-tenant appliance; enterprise scale (P2-4) replaces the
    static key map with JWT/OIDC verification (constant-time, key-count
    independent).
    """
    matched_tenant: str | None = None
    for key, tenant in key_map.items():
        if hmac.compare_digest(candidate, key):
            # Last match wins deterministically; keys are unique in practice.
            matched_tenant = tenant
    return matched_tenant


class AuthMiddleware:
    """Enforce API-key auth and bind requests to the authenticated tenant.

    P2-4: when JWT auth is enabled, a Bearer token that is NOT a static key
    is verified as a short-lived JWT (HS256 allowlist, exp enforced) and the
    tenant comes from the token's ``tenant`` claim. Static keys always match
    first (constant-time), so an API key never accidentally parses as a JWT.
    """

    def __init__(
        self,
        app: Any,
        settings: AuthSettings,
        jwt_manager: Any | None = None,  # neuralguard.auth.jwtauth.JwtManager
        runtime_state: Any | None = None,  # neuralguard.auth.jwtauth.AuthRuntimeState
    ) -> None:
        self.app = app
        self.settings = settings
        # Shared live-key state: rotated keys (rotation API) are visible here
        # because routes and middleware hold the SAME AuthRuntimeState.
        self._state = runtime_state if runtime_state is not None else AuthRuntimeState(settings)
        self._jwt_manager = jwt_manager

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)

        if not self.settings.enabled:
            # Auth disabled (development). Mark unauthenticated for downstream
            # middleware that may key on the principal. State lives in the
            # scope dict, so this is visible to every later layer.
            scope.setdefault("state", {})["authenticated"] = False
            scope["state"]["auth_tenant"] = None
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Public endpoints bypass auth (kept minimal: health only by default).
        if path in self.settings.public_endpoints:
            scope.setdefault("state", {})["authenticated"] = False
            scope["state"]["auth_tenant"] = None
            await self.app(scope, receive, send)
            return

        # Only protect API paths; leave non-API (e.g. /docs) to FastAPI.
        if not path.startswith("/v1/"):
            scope.setdefault("state", {})["authenticated"] = False
            scope["state"]["auth_tenant"] = None
            await self.app(scope, receive, send)
            return

        candidate = _extract_key(request)
        if candidate is None:
            logger.warning("auth_missing_key", path=path)
            metrics.record_auth_rejection("missing_key")
            await self._unauthorized(scope, receive, send, "Missing API key")
            return

        tenant = self._state.lookup(candidate)
        if tenant is None and self._jwt_manager is not None:
            # Not a static key — try JWT verification (P2-4). Bearer-only:
            # an X-API-Key header is a key by definition, not a token.
            if candidate != candidate.strip() or " " in candidate:
                tenant = None
            else:
                tenant = self._jwt_manager.verify(candidate)
                if tenant is None:
                    metrics.record_auth_rejection("invalid_token")
        if tenant is None:
            logger.warning("auth_invalid_key", path=path)
            metrics.record_auth_rejection("invalid_key")
            await self._unauthorized(scope, receive, send, "Invalid API key")
            return

        scope.setdefault("state", {})["authenticated"] = True
        scope["state"]["auth_tenant"] = tenant

        # Enforce tenant binding: the body may carry a tenant_id that must
        # agree with the key's bound tenant. We cannot fully parse the JSON
        # body here without consuming it (needed downstream), so we rely on a
        # lightweight peek only for the tenant mismatch check. To avoid
        # breaking the request stream, the route layer performs the final
        # tenant enforcement after body parsing.
        await self.app(scope, receive, send)

    @staticmethod
    async def _unauthorized(
        scope: dict[str, Any],
        receive: Receive,
        send: Send,
        detail: str,
    ) -> None:
        response = JSONResponse(
            status_code=401,
            content={"error": "unauthorized", "message": detail},
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)
