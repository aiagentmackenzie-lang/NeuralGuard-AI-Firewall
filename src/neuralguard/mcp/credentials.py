"""NG-12 / NG-14: upstream credentials selected server-side, by tool and agent.

NG-12 gave a tool its own credential. That closed "one token for every tool",
but left a second, narrower confusion: two agents permitted the same tool
shared that tool's credential, so the upstream could not say *which agent*
made the call. NG-14 keys the mapping by ``(agent, tool)`` as well, so an
upstream that logs a call can attribute it to the authority that made it.

Mapping keys (``NEURALGUARD_MCP_TOOL_CREDENTIALS``):

- ``"read_file": "env:READONLY_TOKEN"`` — tool-wide: any caller (tenant-level
  or any agent) permitted the tool carries this credential.
- ``"billing-bot|refund": "file:/run/secrets/refund-billing"`` — agent-scoped:
  only that workload agent (NG-11 identity from the authenticated credential)
  carries it.

Resolution order is most-specific-first: ``(agent, tool)`` -> tool-wide ->
default token. Every step fails closed:

- **References, not secrets, in config.** A tool maps to ``env:VAR_NAME`` or
  ``file:/path/to/secret``. Tokens never live in YAML, are never accepted from
  a caller header (the transport already discards those), and never appear in
  an audit event or an error message — only the *reference* does.
- **A mapped-but-unavailable credential denies; it never falls back to the
  default.** Falling back would silently grant the broad default token to a
  call the operator explicitly scoped — a privilege escalation that looks like
  a config typo.
- **An agent-scoped tool does not degrade to the default.** Once *any* agent
  mapping exists for a tool, a caller with no matching mapping is refused —
  including a tenant-level (agent-less) caller. Otherwise configuring
  ``billing-bot|refund`` would hand every other agent the broad default for
  ``refund``, which is precisely the sharing this feature removes. Add a
  tool-wide key when that fallback is genuinely intended.
- **``require_tool_credential`` closes the other direction.** With it set, a
  tool with no mapping at all is refused instead of inheriting the default.
  That is the switch an operator flips once every tool has its own credential.
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

from neuralguard.auth.principal import normalize_agent

ENV_PREFIX = "env:"
FILE_PREFIX = "file:"

# Same separator NG-11 uses for ``key|tenant|agent`` bindings, so operator
# config reads consistently: ``agent|tool``.
AGENT_SEPARATOR = "|"

# A secret longer than this is far more likely to be a mis-pointed file (a PEM
# bundle, a keyring dump) than a bearer token. Refusing it keeps a wrong path
# from turning into an unbounded header.
_MAX_SECRET_BYTES = 4096

# Characters that cannot appear in a tool name we are willing to key on. MCP
# tool names are free-form text, but whitespace and NUL in a config key are
# always a mistake and would make the mapping impossible to match reliably.
_BAD_TOOL_CHARS = frozenset(" \t\n\r\0")


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


def _validate_tool(tool: str) -> str:
    if not tool:
        raise ValueError("invalid credential mapping key: empty tool name")
    if _BAD_TOOL_CHARS & set(tool):
        raise ValueError(f"invalid credential mapping key {tool!r}: whitespace/control in tool")
    return tool


def split_mapping_key(key: str) -> tuple[str | None, str]:
    """Split ``tool`` / ``agent|tool`` into ``(agent, tool)``.

    Raises ``ValueError`` on anything malformed (used at configuration load, so
    a typo fails at boot rather than on the first call that needs it). The
    agent id goes through NG-11's normalizer: it lands in the audit trail and
    in operator findings, so it is lowercased and charset-restricted, and an
    empty agent half is an error rather than an accidental tool-wide mapping.
    """
    raw = key.strip()
    if not raw:
        raise ValueError("invalid credential mapping key: empty key")
    if AGENT_SEPARATOR not in raw:
        return None, _validate_tool(raw)

    parts = raw.split(AGENT_SEPARATOR)
    if len(parts) != 2:
        raise ValueError(f"invalid credential mapping key {key!r}: expected 'tool' or 'agent|tool'")
    agent_part, tool_part = parts
    agent = normalize_agent(agent_part)  # raises InvalidPrincipalError (a ValueError)
    if agent is None:
        raise ValueError(
            f"invalid credential mapping key {key!r}: empty agent id "
            "(omit the '|' for a tool-wide credential)"
        )
    tool = _validate_tool(tool_part.strip())
    return agent, tool


def validate_mapping_key(key: str) -> str:
    """Return the normalized mapping key, or raise ``ValueError`` (used at boot)."""
    agent, tool = split_mapping_key(key)
    return f"{agent}{AGENT_SEPARATOR}{tool}" if agent else tool


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
    """Maps an ``(agent, tool)`` call to the upstream credential it should carry."""

    def __init__(
        self,
        tool_credentials: dict[str, str],
        default_token: str = "",
        require_tool_credential: bool = False,
    ) -> None:
        self._agent_scoped: dict[tuple[str, str], str] = {}
        self._tool_wide: dict[str, str] = {}
        for key, reference in tool_credentials.items():
            agent, tool = split_mapping_key(key)
            ref = validate_reference(reference)
            if agent is None:
                if tool in self._tool_wide:
                    raise ValueError(f"duplicate credential mapping for tool {tool!r}")
                self._tool_wide[tool] = ref
            else:
                scoped = (agent, tool)
                if scoped in self._agent_scoped:
                    raise ValueError(
                        f"duplicate credential mapping for agent {agent!r} + tool {tool!r}"
                    )
                self._agent_scoped[scoped] = ref
        self._scoped_tools = {tool for (_agent, tool) in self._agent_scoped}
        self._default = default_token.strip()
        self._require_mapped = require_tool_credential

    # ── introspection (boot visibility; labels only, never secrets) ──

    def agent_scoped_tools(self) -> list[str]:
        """Tools that have at least one agent-specific credential."""
        return sorted(self._scoped_tools)

    def is_agent_scoped(self, tool: str) -> bool:
        """True when any agent mapping exists for this tool (no default fallback)."""
        return tool in self._scoped_tools

    def source_for(self, tool: str, agent: str | None = None) -> str:
        """The reference label for this call (safe for audit), or a scope marker."""
        reference = self._reference_for(tool, agent)
        if reference is not None:
            return reference
        if self.is_agent_scoped(tool):
            # Deliberate marker: an unmapped agent asking for an agent-scoped
            # tool is denied, not handed the default.
            return "agent-scoped-unmapped"
        return "default" if self._default else "none"

    def is_mapped(self, tool: str, agent: str | None = None) -> bool:
        return self._reference_for(tool, agent) is not None

    def unresolved(self) -> list[str]:
        """References that currently resolve to nothing (names only, for boot logs)."""
        missing: list[str] = []
        references = {*self._tool_wide.values(), *self._agent_scoped.values()}
        for reference in sorted(references):
            try:
                _read_secret(reference)
            except CredentialError:
                missing.append(reference)
        return missing

    # ── resolution ──

    def _reference_for(self, tool: str, agent: str | None) -> str | None:
        """Most-specific-first: (agent, tool) -> tool-wide -> None."""
        if agent:
            scoped = self._agent_scoped.get((agent, tool))
            if scoped is not None:
                return scoped
        return self._tool_wide.get(tool)

    def for_tool(self, tool: str, agent: str | None = None) -> ResolvedCredential:
        """Resolve the credential for one tool call.

        ``agent`` is the authenticated (NG-11-normalized) workload identity —
        never a caller-supplied header value.

        Raises ``CredentialError`` when a mapped reference cannot be satisfied
        (never falls back), when an agent-scoped tool has no mapping for this
        caller (never degrades to the default), or when a mapping is required
        and absent.
        """
        reference = self._reference_for(tool, agent)
        if reference is not None:
            return ResolvedCredential(token=_read_secret(reference), source=reference)
        if self.is_agent_scoped(tool):
            caller = f"agent {agent!r}" if agent else "a tenant-level credential"
            raise CredentialError(
                f"tool {tool!r} is agent-scoped and has no credential mapped for {caller}"
            )
        if self._require_mapped:
            raise CredentialError(f"no upstream credential mapped for tool {tool!r}")
        # Empty default = no Authorization header (legacy local-MCP posture).
        return ResolvedCredential(
            token=self._default, source="default" if self._default else "none"
        )
