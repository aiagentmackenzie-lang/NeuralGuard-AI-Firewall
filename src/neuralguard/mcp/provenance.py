"""NG-9: provenance-lite egress binding (pure core).

CaMeL (arXiv:2503.18813) defeats prompt injection by construction with
capability-based information flow. That is a system architecture, not a
middleware feature — this module deliberately does NOT ship a mini-CaMeL.
The practical slice (the 20% that stops 80% of exfiltration):

- Tool results passing through the gateway are TAINTED into a per-session
  window (keyed on the MCP ``Mcp-Session-Id`` header).
- Tenant policy classifies which tools are EGRESS-CAPABLE (send email,
  HTTP post, upload, ...). A call to an egress tool has its arguments
  checked for tainted content BEFORE forwarding.
- Modes: ``off`` (default — the control is opt-in), ``warn`` (alert + allow,
  audit event), ``block`` (refuse the call — fail-closed when enabled).

Honest detection boundary (documented, not hidden): matching is over
NORMALIZED exact copies (case/whitespace/zero-width-normalized substring
containment). Paraphrase, summarization, and encoding evasion are
CaMeL-class problems — out of scope here by design. What this stops is the
dominant real shape: retrieved content (a planted instruction, a secret, a
canary) being passed verbatim to an outbound tool.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Literal

from pydantic import BaseModel

ProvenanceMode = Literal["off", "warn", "block"]

# Zero-width / bidi control characters stripped before matching (an attacker
# can hide a copy from naive matching by interleaving them; the fold makes
# the comparison robust without touching the delivered payload).
_INVISIBLES = {
    ch
    for rng in (
        (0x200B, 0x200F),
        (0x202A, 0x202E),
        (0x2060, 0x2064),
        (0xFEFF, 0xFEFF),
        (0x00AD, 0x00AD),
    )
    for ch in map(chr, range(rng[0], rng[1] + 1))
}
_INVISIBLE_TABLE = {ord(ch): None for ch in _INVISIBLES}


def normalize(text: str) -> str:
    """Deterministic comparison form: strip invisibles, collapse whitespace, lowercase."""
    return " ".join(text.translate(_INVISIBLE_TABLE).split()).lower()


def extract_text_blocks(jsonrpc_result: Any) -> list[str]:
    """Extract text content from a tools/call JSON-RPC result, defensively.

    MCP tool results carry ``content`` as a list of blocks; text blocks look
    like ``{"type": "text", "text": "..."}``. Malformed shapes yield [] —
    this helper never raises (the forward path must not die on a weird
    upstream result; unextractable content is simply untracked, and the
    route logs the extraction count honestly).
    """
    if not isinstance(jsonrpc_result, dict):
        return []
    content = jsonrpc_result.get("content")
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return texts


def extract_argument_strings(arguments: Any) -> list[str]:
    """Collect string values from a tools/call ``params.arguments`` tree.

    Bounded walk (depth 4, total 64 strings) so a hostile argument tree
    cannot turn the check into a DoS.
    """
    out: list[str] = []

    def _walk(node: Any, depth: int) -> None:
        if len(out) >= 64 or depth > 4:
            return
        if isinstance(node, str):
            if node.strip():
                out.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                _walk(value, depth + 1)
        elif isinstance(node, list):
            for item in node:
                _walk(item, depth + 1)

    _walk(arguments, 0)
    return out


class ProvenanceDecision(BaseModel):
    """The outcome of one egress check (route logs/audits exactly this)."""

    outcome: Literal["unchecked", "clean", "tainted"]
    decision: Literal["allow", "alert_allow", "block"]
    rule_id: str = ""
    reason: str = ""
    tool: str | None = None
    session_id: str | None = None


_UNCHECKED = ProvenanceDecision(
    outcome="unchecked",
    decision="allow",
    reason="Provenance control off for this call (mode=off or tool not egress-capable).",
)


class TaintStore:
    """Bounded per-session window of tainted-content fingerprints (NG-9).

    Fingerprinting: sliding 4-word shingles over the NORMALIZED tool-result
    text. Whole-text substring matching is asymmetric — an argument quoting
    PART of a tainted result evades it — while a shingle set catches any
    partial quote of >= 4 contiguous words. Very short results (< 4 words)
    are stored whole. Long single tokens (base64 blobs, tokens) get a
    prefix fingerprint so huge-payload exfil is not skipped by the shingle
    length cap.

    In-memory per worker, mirroring the baseliner/AG posture: TTL'd, LRU
    cap on sessions, per-session cap on shingles. Bounded memory by
    construction; matching is bounded on the candidate side (first 512
    shingles) so a hostile argument tree cannot turn the check into a DoS.
    """

    _SHINGLE_WORDS = 4

    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        max_sessions: int = 1000,
        max_shingles_per_session: int = 512,
        max_shingle_chars: int = 200,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_shingles_per_session = max_shingles_per_session
        self.max_shingle_chars = max_shingle_chars
        # session_id -> (expires_at, OrderedSet-of-fingerprints)
        self._sessions: OrderedDict[str, tuple[float, OrderedDict[str, None]]] = OrderedDict()

    def _prune_expired(self, now: float) -> None:
        for sid in [sid for sid, (expires, _) in list(self._sessions.items()) if expires <= now]:
            del self._sessions[sid]

    def _fingerprint(self, normalized: str) -> list[str]:
        """Shingles (4-word windows) + long-token prefix fingerprints."""
        words = normalized.split()
        out: list[str] = []
        k = self._SHINGLE_WORDS
        if len(words) < k:
            if normalized:
                out.append(normalized)
        else:
            for i in range(len(words) - k + 1):
                shingle = " ".join(words[i : i + k])
                if len(shingle) <= self.max_shingle_chars:
                    out.append(shingle)
        # Long-token fingerprints: a base64/hex blob quoted verbatim (the
        # prefix is bounded AND distinctive — 32 chars of a blob identify it).
        for word in words:
            if len(word) >= 32:
                out.append("\x00tok:" + word[:32])
        return out

    def taint_result(self, session_id: str, texts: list[str]) -> int:
        """Fingerprint tool-result content into the session window.
        Returns fingerprints stored."""
        if not texts:
            return 0
        now = time.monotonic()
        self._prune_expired(now)
        entry = self._sessions.get(session_id)
        if entry is None:
            if len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)  # evict LRU session
            entry = (now + self.ttl_seconds, OrderedDict())
            self._sessions[session_id] = entry
        else:
            entry = (now + self.ttl_seconds, entry[1])
            self._sessions[session_id] = entry
            self._sessions.move_to_end(session_id)
        fingerprints = entry[1]
        stored = 0
        for text in texts:
            normalized = normalize(text)
            if not normalized:
                continue
            for fp in self._fingerprint(normalized):
                if fp not in fingerprints:
                    fingerprints[fp] = None
                    while len(fingerprints) > self.max_shingles_per_session:
                        fingerprints.popitem(last=False)  # evict oldest
                    stored += 1
        return stored

    def contains_tainted(self, session_id: str, candidate_text: str) -> bool:
        """Whether the candidate's fingerprints intersect the session window."""
        entry = self._sessions.get(session_id)
        if entry is None:
            return False
        if entry[0] <= time.monotonic():
            del self._sessions[session_id]
            return False
        self._sessions.move_to_end(session_id)
        normalized = normalize(candidate_text)
        if not normalized:
            return False
        # Candidates are bounded too: only the first 512 shingles are checked
        # (a hostile 100k-word argument cannot make this loop unbounded).
        candidate_fps = self._fingerprint(normalized)[:512]
        fingerprints = entry[1]
        return any(fp in fingerprints for fp in candidate_fps)


class ProvenanceGate:
    """NG-9 decision core: taint recording + egress binding for one gateway."""

    def __init__(
        self,
        mode: ProvenanceMode = "off",
        ttl_seconds: float = 3600.0,
        max_sessions: int = 1000,
        require_session: bool = False,
    ) -> None:
        self.mode: ProvenanceMode = mode
        self.require_session = require_session
        self._store = TaintStore(ttl_seconds=ttl_seconds, max_sessions=max_sessions)

    def record_tool_result(self, session_id: str | None, jsonrpc_result: Any) -> int:
        """Taint a tools/call response's text content into the session window.

        Returns fragments stored (0 when mode=off or nothing extractable) —
        the route logs the count; a None session records into the shared
        tenant bucket (documented cross-talk: session-less clients share a
        window; require_session exists for strict deployments).
        """
        if self.mode == "off":
            return 0
        texts = extract_text_blocks(jsonrpc_result)
        if not texts:
            return 0
        return self._store.taint_result(session_id or "__tenant__", texts)

    def evaluate_egress(
        self,
        *,
        session_id: str | None,
        tool: str,
        arguments: Any,
        egress_tools: set[str],
    ) -> ProvenanceDecision:
        """Gate one tools/call against the provenance policy.

        - mode=off or a non-egress tool: unchecked (allow) — the control is
          opt-in and scoped to the tools the tenant classified.
        - mode!=off, egress tool, no session id, require_session=True:
          refused — a strict deployment does not let un-attributable calls
          bypass taint attribution.
        - tainted arguments: warn -> alert_allow; block -> block
          (MCP-PROV-001, fail-closed).
        """
        if self.mode == "off" or tool not in egress_tools:
            return _UNCHECKED
        if session_id is None and self.require_session:
            return ProvenanceDecision(
                outcome="unchecked",
                decision="block",
                rule_id="MCP-PROV-002",
                reason=(
                    "No Mcp-Session-Id on an egress tool call with provenance "
                    "enforcement on and require_session=true — the gateway "
                    "cannot attribute taint, so the call is refused "
                    "(fail-closed)."
                ),
                tool=tool,
            )
        strings = extract_argument_strings(arguments)
        if not strings:
            return ProvenanceDecision(
                outcome="clean",
                decision="allow",
                rule_id="MCP-PROV-003",
                reason="Egress call carries no string arguments to check.",
                tool=tool,
                session_id=session_id,
            )
        sid = session_id or "__tenant__"
        for text in strings:
            if self._store.contains_tainted(sid, text):
                decision: Literal["alert_allow", "block"] = (
                    "alert_allow" if self.mode == "warn" else "block"
                )
                return ProvenanceDecision(
                    outcome="tainted",
                    decision=decision,
                    rule_id="MCP-PROV-001",
                    reason=(
                        "Egress tool arguments contain content that passed "
                        "through this gateway's tool results (provenance "
                        f"taint, mode={self.mode}) — "
                        + ("alerted; call allowed." if self.mode == "warn" else "call refused.")
                    ),
                    tool=tool,
                    session_id=session_id,
                )
        return ProvenanceDecision(
            outcome="clean",
            decision="allow",
            rule_id="MCP-PROV-004",
            reason="Egress arguments carry no tainted content.",
            tool=tool,
            session_id=session_id,
        )
