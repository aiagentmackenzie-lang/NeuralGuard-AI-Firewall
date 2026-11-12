"""NG-12: per-tool upstream credentials, resolved server-side.

Before this, one ``NEURALGUARD_MCP_UPSTREAM_AUTH_TOKEN`` authenticated *every*
tool call. A gateway in front of a server whose ``read_file`` and
``delete_database`` tools share one token cannot express "this call carries the
read-only credential" — the upstream sees the same principal either way.

Design rules, all fail-closed:

- **References, not secrets, in config.** A tool maps to ``env:VAR_NAME`` or
  ``file:/path/to/secret``. Tokens never live in YAML, are never accepted from
  a caller header (the transport already discards those), and never appear in
  an audit event or an error message — only the *reference* does.
- **A mapped-but-unavailable credential denies; it never falls back to the
  default.** Falling back would silently grant the broad default token to a
  call the operator explicitly scoped — a privilege escalation that looks like
  a config typo.
- **``require_tool_credential`` closes the other direction.** With it set, a
  tool with no mapping is refused instead of inheriting the default. That is
  the switch an operator flips once every tool has its own credential.
- **Resolution is lazy.** Secrets injected by an orchestrator after boot (a
  mounted file, a projected env var) work without a restart. The cost is that
  a missing secret surfaces as a refused call rather than a failed boot; the
  boot log lists unresolved references by name so it is still visible early.

Stated limit: ``file:`` references read whatever the process can read. This
module checks the secret is non-empty and plausibly a token; it does not police
filesystem permissions — put the file on a 0600 mount and let the OS be the
control it is good at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "env:"
FILE_PREFIX = "file:"

# A secret longer than this is far more likely to be a mis-pointed file (a PEM
# bundle, a keyring dump) than a bearer token. Refusing it keeps a wrong path
# from turning into an unbounded header.
_MAX_SECRET_BYTES = 4096


class CredentialError(Exception):
    """A credential reference is malformed or cannot be satisfied (fail-closed)."""


@dataclass(frozen=True, slots=True)
class ResolvedCredential:
    """The token for one call plus the operator-facing source label.

    ``source`` is the *reference* (``env:READONLY_TOKEN``), never the secret —
    safe for audit events, findings and logs.
    """

    token: str
    source: str


def validate_reference(reference: str) -> str:
    """Return the normalized reference, or raise ``ValueError`` (used at boot).

    Called from the settings layer so a typo fails at configuration load rather
    than on the first call that needs the credential.
    """
    ref = reference.strip()
    if ref.startswith(ENV_PREFIX):
        name = ref[len(ENV_PREFIX) :]
        if not name or any(ch in name for ch in " \t\n\0="):
            raise ValueError(f"invalid credential reference {reference!r}: bad env var name")
        return ENV_PREFIX + name
    if ref.startswith(FILE_PREFIX):
        path = ref[len(FILE_PREFIX) :]
        if not path or "\0" in path:
            raise ValueError(f"invalid credential reference {reference!r}: empty path")
        return FILE_PREFIX + path
    raise ValueError(
        f"invalid credential reference {reference!r}: expected 'env:NAME' or 'file:/path'"
    )


def _read_secret(reference: str) -> str:
    if reference.startswith(ENV_PREFIX):
        name = reference[len(ENV_PREFIX) :]
        raw = os.environ.get(name, "")
    elif reference.startswith(FILE_PREFIX):
        path = Path(reference[len(FILE_PREFIX) :])
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            # Path in the message, never file contents.
            raise CredentialError(f"credential file unreadable: {path}") from exc
    else:  # pragma: no cover - validate_reference rejects these at boot
        raise CredentialError(f"unsupported credential reference: {reference}")

    secret = raw.strip()
    if not secret:
        raise CredentialError(f"credential reference resolved to an empty value: {reference}")
    if len(secret.encode("utf-8")) > _MAX_SECRET_BYTES:
        raise CredentialError(f"credential reference resolved to an oversized value: {reference}")
    return secret


class ToolCredentialResolver:
    """Maps a tool call to the upstream credential it should carry."""

    def __init__(
        self,
        tool_credentials: dict[str, str],
        default_token: str = "",
        require_tool_credential: bool = False,
    ) -> None:
        self._mapping = {tool: validate_reference(ref) for tool, ref in tool_credentials.items()}
        self._default = default_token.strip()
        self._require_mapped = require_tool_credential

    def source_for(self, tool: str) -> str:
        """The reference label for this tool (safe for audit), or 'default'."""
        return self._mapping.get(tool, "default" if self._default else "none")

    def is_mapped(self, tool: str) -> bool:
        return tool in self._mapping

    def unresolved(self) -> list[str]:
        """References that currently resolve to nothing (names only, for boot logs)."""
        missing: list[str] = []
        for reference in sorted(set(self._mapping.values())):
            try:
                _read_secret(reference)
            except CredentialError:
                missing.append(reference)
        return missing

    def for_tool(self, tool: str) -> ResolvedCredential:
        """Resolve the credential for one tool call.

        Raises ``CredentialError`` when a mapped reference cannot be satisfied
        (never falls back) or when a mapping is required and absent.
        """
        reference = self._mapping.get(tool)
        if reference is not None:
            return ResolvedCredential(token=_read_secret(reference), source=reference)
        if self._require_mapped:
            raise CredentialError(f"no upstream credential mapped for tool {tool!r}")
        # Empty default = no Authorization header (legacy local-MCP posture).
        return ResolvedCredential(
            token=self._default, source="default" if self._default else "none"
        )
