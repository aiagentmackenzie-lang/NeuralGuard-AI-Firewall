"""P2-4 auth management routes: JWT issuance + runtime key rotation.

Mounted only when at least one of the features is enabled:
- ``POST /v1/auth/token`` — exchange a valid credential (static key or
  already-valid JWT) for a short-lived JWT bound to the caller's tenant.
- ``POST /v1/auth/keys/rotate`` — ADMIN-tenant only. Issues a new key
  (generated or supplied), optionally revokes the old one, persists to the
  runtime key store when ``keys_file`` is configured, and updates the auth
  middleware's live key map.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["auth"])


class TokenRequest(BaseModel):
    """Body is optional — the credential is the Authorization header itself."""

    ttl_minutes: int | None = Field(default=None, ge=1, le=1440)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    tenant: str


class RotateRequest(BaseModel):
    tenant: str = Field(description="Tenant the new key is bound to")
    new_key: str | None = Field(
        default=None,
        description="Supply a specific key, or omit to have the server generate one",
    )
    revoke_caller_key: bool = Field(
        default=False,
        description="Also revoke the credential used for this call (admin key rollover)",
    )


class RotateResponse(BaseModel):
    key: str
    tenant: str
    persisted: bool
    revoked_caller_key: bool


@router.post("/v1/auth/token", response_model=TokenResponse)
async def issue_token(token_request: TokenRequest, request: Request) -> Any:
    """Exchange a valid credential for a short-lived JWT (P2-4).

    The caller is already authenticated (middleware). The token is bound to
    the caller's tenant — privilege cannot be widened by exchanging.
    """
    jwt_manager = getattr(request.app.state, "jwt_manager", None)
    tenant = getattr(request.state, "auth_tenant", None)
    if jwt_manager is None:
        return JSONResponse(
            status_code=404,
            content={"error": "not_available", "message": "JWT auth is not enabled"},
        )
    assert tenant is not None  # middleware guarantees for authenticated /v1/* calls
    token = jwt_manager.issue(tenant, ttl_minutes=token_request.ttl_minutes)
    ttl = token_request.ttl_minutes or request.app.state.config.auth.jwt_ttl_minutes
    return TokenResponse(access_token=token, expires_in_minutes=ttl, tenant=tenant)


@router.post("/v1/auth/keys/rotate", response_model=RotateResponse)
async def rotate_key(rotate_request: RotateRequest, request: Request) -> Any:
    """Rotate API keys at runtime (P2-4). ADMIN tenant only.

    Durable when ``NEURALGUARD_AUTH_KEYS_FILE`` is set (atomic 0600 write,
    reloaded by every worker boot). Without a keys file: runtime-only —
    refused in production (a rotation that evaporates on restart is a
    footgun), allowed with a loud warning in development.
    """
    mw = request.app.state.auth_state
    tenant = getattr(request.state, "auth_tenant", None)
    config = request.app.state.config
    if tenant != config.auth.admin_tenant:
        return JSONResponse(
            status_code=403,
            content={
                "error": "forbidden",
                "message": "key rotation requires an admin-tenant credential",
            },
        )

    key_store = getattr(request.app.state, "key_store", None)
    if key_store is None and config.environment == "production":
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_persisted",
                "message": (
                    "runtime-only rotation refused in production: set "
                    "NEURALGUARD_AUTH_KEYS_FILE for durable rotation"
                ),
            },
        )

    new_key = rotate_request.new_key
    if new_key is None:
        # Generating a key that is never persisted is a footgun (it dies with
        # the process) — refused in production, allowed with warning in dev.
        if config.environment == "production":
            return JSONResponse(
                status_code=409,
                content={
                    "error": "not_persisted",
                    "message": (
                        "runtime-only key generation requires NEURALGUARD_AUTH_KEYS_FILE "
                        "in production (a generated key must be persisted to be usable)"
                    ),
                },
            )
        from neuralguard.auth.jwtauth import RuntimeKeyStore

        new_key = RuntimeKeyStore.generate_key()
    mw.add_key(new_key, rotate_request.tenant)
    if key_store is not None:
        key_store.add(new_key, rotate_request.tenant)

    revoked = False
    caller_header = request.headers.get("Authorization", "")
    caller_key = (
        caller_header.split(" ", 1)[1].strip()
        if caller_header.lower().startswith("bearer ")
        else None
    )
    if rotate_request.revoke_caller_key and caller_key:
        revoked = bool(mw.revoke_key(caller_key))
    return RotateResponse(
        key=new_key,
        tenant=rotate_request.tenant,
        persisted=key_store is not None,
        revoked_caller_key=revoked,
    )
