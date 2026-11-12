"""JWT bearer auth + runtime key rotation (P2-4).

Two auth methods coexist:
- **Static API keys** (existing): `<key>|<tenant>` config entries — machine
  credentials, constant-time compared.
- **Short-lived JWTs** (new, opt-in): issued by ``POST /v1/auth/token`` in
  exchange for a valid key, verified on every request. The static key can
  then live in the deployment config only; callers present 15-minute
  bearer tokens.

Honest scope (no vapor):
- HS256 with a server-side secret (``NEURALGUARD_AUTH_JWT_SECRET``, ≥32
  chars, required when jwt_enabled). RS256/OIDC discovery needs JWKS
  fetch infrastructure and is a documented follow-up, not claimed here.
- ``alg`` is allowlisted to HS256 — the alg-confusion / ``alg:none`` class
  is structurally rejected (PyJWT enforces the allowlist on decode).
- ``exp`` is mandatory-by-generation and enforced on verify.
- Runtime key rotation persists to ``NEURALGUARD_AUTH_KEYS_FILE`` (atomic
  write, 0600). Without a keys file, rotation is runtime-only and is
  REFUSED in production (a rotation that silently evaporates on restart
  is a footgun, not a feature). Multi-worker note: each worker process
  reloads the keys file on its own rotation call; until every worker has
  rotated there is a divergence window — the runbook recommends redeploy
  (env keys) for multi-worker rotation.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import tempfile
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import jwt as pyjwt
import structlog

from neuralguard.auth.principal import Principal, normalize_agent

if TYPE_CHECKING:
    from pathlib import Path

    from neuralguard.config.settings import AuthSettings

logger = structlog.get_logger(__name__)

# The allowlist is the security boundary: anything not in it (none, RS*,
# HS384/512, symmetric-confusion variants) is rejected by decode.
_ALLOWED_ALGS = ["HS256"]

_TOKEN_ISSUER = "neuralguard"


class AuthRuntimeState:
    """Shared, mutable live-key state for middleware AND rotation routes.

    Constructed once at boot, passed to AuthMiddleware (lookup) and stashed
    on app.state (routes call add_key/revoke_key). Static env keys are the
    seed; rotated keys are added at runtime and persisted by RuntimeKeyStore.
    """

    def __init__(self, settings: AuthSettings) -> None:
        self.settings = settings
        self._key_map: dict[str, Principal] = settings.key_to_principal()

    def resolve(self, candidate: str) -> Principal | None:
        """Constant-time principal lookup (same guarantee as the old helper).

        Every key is compared with ``compare_digest`` (no early exit on match)
        so timing does not reveal which key matched.
        """
        import hmac

        matched: Principal | None = None
        for key, principal in self._key_map.items():
            if hmac.compare_digest(candidate, key):
                matched = principal
        return matched

    def lookup(self, candidate: str) -> str | None:
        """Tenant-only view of :meth:`resolve` (kept for existing callers)."""
        principal = self.resolve(candidate)
        return principal.tenant if principal is not None else None

    def add_key(self, key: str, tenant: str, agent: str | None = None) -> None:
        self._key_map[key] = Principal(tenant=tenant.lower(), agent=normalize_agent(agent))

    def revoke_key(self, key: str) -> bool:
        return self._key_map.pop(key, None) is not None


class JwtManager:
    """Issue and verify short-lived HS256 bearer tokens."""

    def __init__(self, settings: AuthSettings) -> None:
        # Settings validator enforces: jwt_enabled ⇒ secret set and ≥32 chars.
        self.settings = settings

    def issue(self, tenant: str, ttl_minutes: int | None = None, agent: str | None = None) -> str:
        """Mint a token bound to ``tenant`` (and optionally one workload agent).

        The agent travels as a claim, so per-agent policy survives token
        exchange without trusting anything the caller sends in a header.
        """
        now = datetime.now(UTC)
        ttl = ttl_minutes if ttl_minutes is not None else self.settings.jwt_ttl_minutes
        normalized_agent = normalize_agent(agent)
        payload = {
            "iss": _TOKEN_ISSUER,
            "sub": f"agent:{tenant}/{normalized_agent}" if normalized_agent else f"tenant:{tenant}",
            "iat": now,
            "exp": now + timedelta(minutes=ttl),
            "tenant": tenant,
        }
        if normalized_agent:
            payload["agent"] = normalized_agent
        secret = self.settings.jwt_secret
        assert secret is not None  # validator guarantees when jwt_enabled
        return pyjwt.encode(payload, secret, algorithm="HS256")

    def verify_principal(self, token: str) -> Principal | None:
        """Return the bound principal for a valid token, else None.

        Signature, allowlisted algorithm, and expiry are all enforced by
        decode; any failure is a None (→ 401 upstream), never a raise. A
        malformed ``agent`` claim invalidates the token rather than degrading
        it to a tenant-wide credential.
        """
        secret = self.settings.jwt_secret
        if secret is None:
            return None
        try:
            payload: dict[str, Any] = pyjwt.decode(
                token, secret, algorithms=_ALLOWED_ALGS, issuer=_TOKEN_ISSUER
            )
        except pyjwt.InvalidTokenError:
            return None
        tenant = payload.get("tenant")
        if not isinstance(tenant, str) or not tenant:
            return None
        raw_agent = payload.get("agent")
        if raw_agent is not None and not isinstance(raw_agent, str):
            return None
        try:
            agent = normalize_agent(raw_agent)
        except ValueError:
            return None
        return Principal(tenant=tenant, agent=agent)

    def verify(self, token: str) -> str | None:
        """Return the bound tenant for a valid token, else None."""
        principal = self.verify_principal(token)
        return principal.tenant if principal is not None else None


class RuntimeKeyStore:
    """Durable runtime key store backing the rotation API.

    File shape (JSON): {"keys": [{"key", "tenant", "added_at",
    "rotated_from"?}, ...]}. Loaded at boot and merged with env keys (env
    keys remain the bootstrap/admin path). Writes are atomic (temp file +
    rename) with 0600 permissions.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._keys: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for entry in data.get("keys", []):
                key, tenant = str(entry.get("key", "")), str(entry.get("tenant", ""))
                if key and tenant:
                    self._keys[key] = tenant.lower()
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("keys_file_unreadable", error=str(exc), path=str(self.path))

    def all_keys(self) -> dict[str, str]:
        """{key: tenant} for every runtime-rotated key."""
        return dict(self._keys)

    def add(self, key: str, tenant: str, rotated_from: str | None = None) -> None:
        self._keys[key] = tenant.lower()
        self._persist(rotated_from=rotated_from)

    def remove(self, key: str) -> bool:
        if key not in self._keys:
            return False
        del self._keys[key]
        self._persist()
        return True

    @staticmethod
    def generate_key() -> str:
        return f"ng_{secrets.token_urlsafe(32)}"

    def _persist(self, rotated_from: str | None = None) -> None:
        payload = {
            "keys": [
                {
                    "key": k,
                    "tenant": t,
                    "rotated_from": rotated_from,
                }
                for k, t in self._keys.items()
            ]
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".keys-", suffix=".json")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
