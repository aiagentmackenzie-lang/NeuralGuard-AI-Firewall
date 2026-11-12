"""NG-11: the authenticated principal (tenant + optional workload agent).

Identity is derived from the **credential**, never from a request header. A
caller cannot claim to be an agent; its key or JWT is bound to one at issue
time (``key|tenant|agent`` in config, or an ``agent`` claim on a token minted
in exchange for such a key). A header could be spoofed and would make every
per-agent rule decorative.

``agent`` is deliberately a narrow identifier, not free text: it lands in the
hash-chained audit trail and in operator-facing findings, so it is lowercased
and restricted to a conservative charset. An invalid binding fails loudly at
config-load time (boot) rather than silently degrading to a tenant-wide
identity — a misconfigured agent id that quietly becomes "no agent" would turn
per-agent policy off without anyone noticing."""

from __future__ import annotations

import re
from dataclasses import dataclass

_AGENT_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


class InvalidPrincipalError(ValueError):
    """A credential binding names an agent id we refuse to carry."""


def normalize_agent(agent: str | None) -> str | None:
    """Validate and normalize an agent id. Raises on anything malformed.

    None means "no agent identity" (a tenant-level credential) and stays None.
    """
    if agent is None:
        return None
    candidate = agent.strip().lower()
    if not candidate:
        return None
    if not _AGENT_ID_RE.match(candidate):
        raise InvalidPrincipalError(
            "agent id must be 1-64 chars of [a-z0-9._-] starting alphanumeric"
        )
    return candidate


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is calling: a tenant, optionally one workload agent inside it."""

    tenant: str
    agent: str | None = None

    def describe(self) -> str:
        """Log-safe rendering; never used for policy decisions."""
        return f"{self.tenant}/{self.agent}" if self.agent else self.tenant
