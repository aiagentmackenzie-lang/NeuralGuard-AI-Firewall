"""NG-7: stateful MCP tool-inventory baseliner — the rug-pull detector.

Lifecycle per MCP server (upstream):

1. **First sighting** of ``tools/list`` -> baseline created (signed when a
   signing seed is configured), anchored into the audit trail.
2. **Match** on subsequent listings -> nothing to do.
3. **Drift** — the catalog hash changed:
   - *strict* mode (default, fail-closed): the changed catalog is WITHHELD
     from the caller (403 + drift report), the baseliner enters a poisoned
     state, and every ``tools/call`` is refused until an explicit
     re-baseline. A call naming a tool the last-good baseline never
     contained is a rug-pull *execute* attempt — refused even in advisory
     mode (that is not "drift observation", that is the payload running).
   - *advisory* mode: the changed catalog passes through with the drift
     report attached (audit event + response header) — for canary
     deployments that measure before they enforce.
4. **Recovery from drift** (the ONLY paths, by construction):
   - *Signature-verified*: the upstream/registry ships an Ed25519 signature
     over the NEW catalog hash (transported next to the response, e.g. an
     ``X-MCP-Catalog-Signature`` header) and the operator has pinned a
     verification key. ``check()`` verifies and auto-rebaselines.
   - *Operator re-baseline*: ``rebaseline()`` — an explicit administrative
     act (config change + restart, or an admin call), never implicit.
   - With ``require_signature_on_change=true``, ONLY the signature path
     resolves drift — an unsigned catalog change can never pass, though the
     operator may still explicitly force a rebaseline for planned migrations
     (that call is logged as such).

Honest posture notes (documented, not hidden):
- Baselines are IN-MEMORY per worker. A worker restart re-baselines on
  first sight with a loud ``BASELINE_RECREATED_RESTART`` audit marker — the
  audit chain carries every historical hash pair, so drift evidence
  survives restarts even though the in-memory state does not.
- Every state transition is a returned ``BaselineState`` the route logs and
  audits; the decision core has no I/O side effects (testability first).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from neuralguard.logging.signing import verify_event_signature
from neuralguard.mcp.manifest import (
    ManifestBaseline,
    build_baseline,
    catalog_hash,
    drift_report,
)

BaselineMode = Literal["strict", "advisory"]


class BaselineState(BaseModel):
    """Immutable snapshot of a decision — the route logs/audits exactly this."""

    outcome: Literal[
        "BASELINE_CREATED",
        "BASELINE_RECREATED_RESTART",
        "MATCH",
        "DRIFT_BLOCKED",
        "DRIFT_PASSED_ADVISORY",
        "REBASELINED",
        "NO_BASELINE",
    ]
    decision: Literal["allow", "block", "alert_allow"]
    old_catalog_hash: str | None = None
    new_catalog_hash: str | None = None
    report: dict[str, Any] | None = Field(
        default=None, description="drift_report payload on drift; None otherwise."
    )
    reason: str


class McpBaseliner:
    """Per-server tool-catalog integrity state (NG-7).

    One instance per MCP server per worker. Concurrency: all decision
    methods are synchronous CPU-only work on the event loop's single thread
    (the route awaits the transport, then decides) — same discipline as the
    Agent Guardian session window. No locks needed by construction.
    """

    def __init__(
        self,
        server_id: str,
        mode: BaselineMode = "strict",
        signing_seed_hex: str | None = None,
        verify_pubkey_hex: str | None = None,
        require_signature_on_change: bool = False,
    ) -> None:
        self.server_id = server_id
        self.mode: BaselineMode = mode
        self.signing_seed_hex = signing_seed_hex or None
        self.verify_pubkey_hex = verify_pubkey_hex or None
        self.require_signature_on_change = require_signature_on_change
        self._baseline: ManifestBaseline | None = None
        self._poisoned = False  # strict-mode drift latches until recovery

    # ── Queries (tools/call path) ──────────────────────────────────────

    @property
    def poisoned(self) -> bool:
        """True after an unresolved strict-mode drift."""
        return self._poisoned

    @property
    def baseline(self) -> ManifestBaseline | None:
        return self._baseline

    def evaluate_call(self, tool_name: str) -> BaselineState:
        """Gate one ``tools/call`` against the last-good baseline.

        - No baseline yet (the server never listed tools): strict refuses —
          a gateway that never saw the catalog cannot vouch for any tool;
          advisory allows with an alert.
        - Poisoned (unresolved drift): refuse EVERYTHING.
        - Unknown tool (not in the last-good catalog): rug-pull execute —
          refused in BOTH modes.
        """
        if self._baseline is None:
            if self.mode == "strict":
                return BaselineState(
                    outcome="NO_BASELINE",
                    decision="block",
                    reason=(
                        "No tool-catalog baseline for this MCP server — strict "
                        "mode refuses tool calls until a tools/list is baselined."
                    ),
                )
            return BaselineState(
                outcome="NO_BASELINE",
                decision="alert_allow",
                reason=(
                    "No tool-catalog baseline (advisory mode) — allowing the "
                    "call but alerting: the gateway cannot vouch for this tool."
                ),
            )
        if self._poisoned:
            return BaselineState(
                outcome="DRIFT_BLOCKED",
                decision="block",
                old_catalog_hash=self._baseline.catalog_hash,
                reason=(
                    "Tool-catalog drift is unresolved (strict mode) — tool "
                    "executions are refused until an explicit re-baseline."
                ),
            )
        if tool_name not in self._baseline.tool_names:
            return BaselineState(
                outcome="DRIFT_BLOCKED",
                decision="block",
                old_catalog_hash=self._baseline.catalog_hash,
                reason=(
                    f"Tool {tool_name!r} is not in the last-good baseline — a tool "
                    "the catalog never contained is executing (rug-pull shape), "
                    "refused in every mode."
                ),
            )
        return BaselineState(
            outcome="MATCH",
            decision="allow",
            old_catalog_hash=self._baseline.catalog_hash,
            reason=f"Tool {tool_name!r} is in the last-good baseline.",
        )

    # ── Transitions (tools/list path) ──────────────────────────────────

    def check(
        self,
        tools: list[dict[str, Any]],
        change_signature_hex: str | None = None,
    ) -> BaselineState:
        """Check a freshly fetched ``tools/list`` catalog against the baseline.

        ``change_signature_hex``: an Ed25519 signature over the NEW catalog
        hash, transported by the trusted registry alongside the response
        (header). Verified only when ``verify_pubkey_hex`` is pinned.
        """
        new_hash = catalog_hash(tools)

        if self._baseline is None:
            baseline = self._record(tools, restart_rebaseline=False)
            return BaselineState(
                outcome="BASELINE_CREATED",
                decision="allow",
                new_catalog_hash=new_hash,
                reason=(
                    f"First sighting: baseline recorded "
                    f"({baseline.tool_count} tools, "
                    f"signed={baseline.signature is not None})."
                ),
            )

        old = self._baseline
        if old.catalog_hash == new_hash and not self._poisoned:
            return BaselineState(
                outcome="MATCH",
                decision="allow",
                old_catalog_hash=old.catalog_hash,
                new_catalog_hash=new_hash,
                reason="Catalog unchanged since the last-good baseline.",
            )

        # ── DRIFT ──
        report = drift_report(old, tools)

        # Recovery path 1: signature-verified change.
        if (
            self.verify_pubkey_hex
            and change_signature_hex
            and verify_event_signature(new_hash, change_signature_hex, self.verify_pubkey_hex)
        ):
            return self.rebaseline(tools, via_signature=True)

        if self.mode == "advisory":
            return BaselineState(
                outcome="DRIFT_PASSED_ADVISORY",
                decision="alert_allow",
                old_catalog_hash=old.catalog_hash,
                new_catalog_hash=new_hash,
                report=report,
                reason=(
                    "Tool-catalog drift detected (advisory mode) — catalog "
                    "passed through with the drift report attached."
                ),
            )

        # strict mode: withhold the changed catalog, latch poisoned.
        self._poisoned = True
        sig_note = (
            " Signature verification unavailable/failed and "
            "require_signature_on_change is set — only a signed change or an "
            "explicit operator re-baseline resolves this."
            if self.require_signature_on_change
            else " Explicit operator re-baseline (or a verified registry signature) resolves it."
        )
        return BaselineState(
            outcome="DRIFT_BLOCKED",
            decision="block",
            old_catalog_hash=old.catalog_hash,
            new_catalog_hash=new_hash,
            report=report,
            reason=(
                "Tool-catalog drift detected (strict mode): the changed catalog "
                "is withheld and tool calls are refused until an explicit "
                "re-baseline." + sig_note
            ),
        )

    def rebaseline(
        self, tools: list[dict[str, Any]], *, via_signature: bool = False
    ) -> BaselineState:
        """Explicit re-baseline (operator action / verified signature).

        The ONLY path out of the poisoned state — never called implicitly
        on drift.
        """
        self._record(tools, restart_rebaseline=False)
        self._poisoned = False
        return BaselineState(
            outcome="REBASELINED",
            decision="allow",
            new_catalog_hash=catalog_hash(tools),
            reason=(
                "Explicit re-baseline recorded"
                + (" via signature-verified catalog" if via_signature else "")
                + " — poisoned state cleared."
            ),
        )

    def restart_rebaseline(self, tools: list[dict[str, Any]]) -> BaselineState:
        """Worker-restart re-baseline: same trust decision, loud marker."""
        self._record(tools, restart_rebaseline=True)
        self._poisoned = False
        return BaselineState(
            outcome="BASELINE_RECREATED_RESTART",
            decision="allow",
            new_catalog_hash=catalog_hash(tools),
            reason=(
                "Worker restart: baseline recreated from first sight — "
                "marked restart_rebaseline in the audit trail."
            ),
        )

    # ── Internals ──────────────────────────────────────────────────────

    def _record(self, tools: list[dict[str, Any]], *, restart_rebaseline: bool) -> ManifestBaseline:
        self._baseline = build_baseline(
            tools,
            self.server_id,
            signing_seed_hex=self.signing_seed_hex,
            restart_rebaseline=restart_rebaseline,
        )
        return self._baseline
