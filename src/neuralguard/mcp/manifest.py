"""NG-7: MCP tool-catalog manifest hashing + signed baselines (pure core).

The rug pull (MDPI FI 18(5):243, MCPSecBench / Trustworthy MCP Registry):
a benign MCP server mutates its tool list mid-session (``tools/list_changed``),
swapping in a poisoned tool or a description carrying an indirect prompt
injection. Standard clients cannot detect the temporal drift. The research
answer — hash the tool catalog, sign manifest mutations (Ed25519), verify
per update — is what this module provides, reusing P2-10's live-fire-proven
Ed25519 signing.

Canonicalization: a catalog is hashed over a DETERMINISTIC encoding — tools
sorted by name, each reduced to ``{name, description, inputSchema}``, serialized
as compact sorted-key JSON. Both sides of a comparison run the same function,
so any semantic change in the defended surface (names, descriptions, schemas)
changes the hash; metadata noise (titles, annotations ordering) must NOT.

The module is pure: no I/O, no state. State lives in ``baseliner.py``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

# ── Catalog canonicalization ───────────────────────────────────────────────

# Fields that constitute the defended surface of a tool. Everything else in
# a tools/list entry (title, annotations, _meta) is metadata — useful to the
# agent, irrelevant to the security shape of the call.
_DEFENDED_FIELDS = ("name", "description", "inputSchema")


def _canonical_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Reduce a tools/list entry to its defended surface, canonical key order."""
    out: dict[str, Any] = {}
    for field in _DEFENDED_FIELDS:
        value = tool.get(field)
        if value is not None:
            out[field] = value
    return out


def canonical_catalog(tools: list[dict[str, Any]]) -> str:
    """Deterministic canonical encoding of a tools/list catalog.

    Tools are sorted by name (duplicates: first occurrence wins — a server
    sending duplicate names is itself a drift signal the baseliner reports);
    each tool is reduced to its defended fields; the result is compact JSON
    with sorted keys. The same catalog ALWAYS yields the same string.
    """
    by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        reduced = _canonical_tool(tool)
        name = str(reduced.get("name", ""))
        if name and name not in by_name:
            by_name[name] = reduced
    ordered = [by_name[n] for n in sorted(by_name)]
    return json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def catalog_hash(tools: list[dict[str, Any]]) -> str:
    """SHA-256 hex of the canonical catalog encoding."""
    return hashlib.sha256(canonical_catalog(tools).encode("utf-8")).hexdigest()


# ── Signed baseline (P2-10 crypto reuse) ───────────────────────────────────


class ManifestBaseline(BaseModel):
    """A signed record of one tool catalog's integrity.

    ``catalog_hash`` is the SHA-256 of the canonical encoding. When a signing
    seed is configured, ``signature`` is the Ed25519 signature (hex) over
    ``catalog_hash`` (the same sign-the-hash discipline as P2-10 audit events)
    and ``pubkey_fingerprint`` identifies the key without storing it.
    """

    server_id: str = Field(
        description="Logical id of the MCP server (upstream URL hash or given name)."
    )
    catalog_hash: str
    tool_count: int
    tool_names: list[str]
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    signature: str | None = Field(
        default=None,
        description="Ed25519 signature (hex) over catalog_hash; None = unsigned baseline.",
    )
    pubkey_fingerprint: str | None = Field(
        default=None,
        description="SHA-256 prefix of the verifying public key (hex); None = unsigned.",
    )
    restart_rebaseline: bool = Field(
        default=False,
        description=(
            "True when this baseline was created after a worker restart rather "
            "than a first sighting — an honest, loud marker in the audit trail "
            "(in-memory baselines do not survive restarts; the audit chain "
            "carries every historical hash)."
        ),
    )

    def signing_payload(self) -> str:
        """The exact string signed/verified (catalog_hash only — metadata is
        informative, the hash is the integrity claim)."""
        return self.catalog_hash


def build_baseline(
    tools: list[dict[str, Any]],
    server_id: str,
    signing_seed_hex: str | None = None,
    restart_rebaseline: bool = False,
) -> ManifestBaseline:
    """Hash a catalog into a baseline, signing it when a seed is configured.

    Raises SigningKeyError on an unusable seed (fail-closed: a configured
    signing key that cannot sign must stop the deployment, not downgrade to
    unsigned baselines silently).
    """
    catalog_h = catalog_hash(tools)
    names = sorted({str(t.get("name", "")) for t in tools if t.get("name")})
    if signing_seed_hex:
        from neuralguard.logging.signing import (
            SigningKeyError,
            public_key_from_seed,
            sign_event_hash,
        )

        try:
            signature = sign_event_hash(catalog_h, signing_seed_hex)
            fingerprint = hashlib.sha256(
                public_key_from_seed(signing_seed_hex).encode()
            ).hexdigest()[:16]
        except Exception as exc:
            raise SigningKeyError(f"MCP signing seed unusable: {exc!r}") from exc
        return ManifestBaseline(
            server_id=server_id,
            catalog_hash=catalog_h,
            tool_count=len(names),
            tool_names=names,
            signature=signature,
            pubkey_fingerprint=fingerprint,
            restart_rebaseline=restart_rebaseline,
        )
    return ManifestBaseline(
        server_id=server_id,
        catalog_hash=catalog_h,
        tool_count=len(names),
        tool_names=names,
        restart_rebaseline=restart_rebaseline,
    )


def verify_baseline_signature(baseline: ManifestBaseline, pubkey_hex: str) -> bool:
    """Verify a signed baseline against a known public key (P2-10 verify)."""
    from neuralguard.logging.signing import verify_event_signature

    if not baseline.signature:
        return False
    return verify_event_signature(baseline.signing_payload(), baseline.signature, pubkey_hex)


# ── Drift reporting (what lands in the audit trail / SIEM) ─────────────────


def drift_report(baseline: ManifestBaseline, new_tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Human-auditable diff between the last-good baseline and a new catalog.

    Shape (JSON-safe, SIEM-ready): added/removed tool names, changed tool
    names (same name set but the canonical hash moved — description/schema
    drift), and the old/new catalog hashes.
    """
    old_names = set(baseline.tool_names)

    new_canonical: dict[str, str] = {}
    for tool in new_tools:
        reduced = _canonical_tool(tool)
        name = str(reduced.get("name", ""))
        if name and name not in new_canonical:
            new_canonical[name] = hashlib.sha256(
                json.dumps(
                    reduced, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
            ).hexdigest()

    # Hash the baseline-side tools the same way for a per-tool diff. The
    # baseline stores names only (the full old catalog is reconstructable from
    # the audit chain when forensic depth is needed), so per-tool "changed"
    # detection compares name presence; description/schema changes surface as
    # catalog-level drift with BOTH hashes recorded — the pair is the proof.
    new_names = set(new_canonical)
    return {
        "old_catalog_hash": baseline.catalog_hash,
        "new_catalog_hash": catalog_hash(new_tools),
        "old_tool_count": baseline.tool_count,
        "new_tool_count": len(new_names),
        "added": sorted(new_names - old_names),
        "removed": sorted(old_names - new_names),
        # Same name set but different catalog hash => description/schema drift.
        "changed": sorted(old_names & new_names)
        if (
            set(baseline.tool_names) == set(new_names)
            and baseline.catalog_hash != catalog_hash(new_tools)
        )
        else [],
        "signature": baseline.signature,
        "recorded_at": baseline.recorded_at.isoformat(),
    }
