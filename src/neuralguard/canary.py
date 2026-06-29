"""Canary token manager — per-session system-prompt exfiltration canaries (B3).

A canary token is an unpredictable, per-session string the operator injects
into the LLM system prompt before serving a turn. If the token later appears
in the model's output, the system prompt has been exfiltrated (the model was
induced to repeat its own confidential instructions). NeuralGuard mints the
token, and the ``/v1/scan/output`` endpoint detects a leak.

Design
------
- **Deterministic + unguessable.** The token is HMAC-SHA256(``session_id`` ||
  ``label``) keyed by a server-side secret, base32-encoded. Mint and detect
  re-derive the same token for the same session, so the operator does not have
  to store the token between the mint call and the output scan — only the
  ``session_id`` is the join key. An attacker who does not know the secret
  cannot predict the token, so they cannot smuggle it out without the model
  actually leaking it.
- **No storage.** Because derivation is deterministic, no server-side state is
  kept. This is a detection signal, not a session store. (Contrast with Agent
  Guardian, which IS stateful.)
- **Fail-closed on misconfiguration.** Minting without a configured secret
  raises; a misconfigured manager is never silently available. Detection
  (``check_leak``) is fail-*safe*: a misconfigured/disabled manager reports
  no leak rather than crashing the output scan, because the canary is an
  additive signal — the output scan still runs its PII / extraction patterns.
- **Bounded label space.** A small set of labels (``A``..``H``) lets the
  operator place up to 8 distinct canaries in a single system prompt (e.g.
  at the top, middle, and bottom) without minting an unbounded number of
  tokens. Detection recomputes the same bounded set, so it is O(labels) HMACs
  per output scan — cheap.
- **The secret is never logged or echoed.** Mint returns only the derived
  token, never the secret. Partial-token evidence in findings is redacted.

Non-goals
---------
- Not a forensics-grade watermark; a determined adversary who exfiltrates the
  prompt character-by-character can evade a static canary. This raises the
  bar and catches the common "repeat your instructions" exfiltration, which is
  the documented threat (OWASP LLM07 / ASI07).
- No revocation: rotating the server secret invalidates all outstanding
  canaries (intentional — rotate on suspected leak).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralguard.config.settings import CanarySettings

# 10 bytes of HMAC output -> 80 bits of entropy, base32-encoded to 16 chars.
# Enough that guessing is infeasible and the token is short enough to embed
# in a system prompt without distortion.
_HMAC_BYTES = 10
# Bounded label space; mint and detect both iterate this set.
_MAX_LABELS = 8
_LABELS = [chr(ord("A") + i) for i in range(_MAX_LABELS)]
_TOKEN_PREFIX = "NGCANARY-"


class CanaryError(Exception):
    """Base error for canary misconfiguration."""


class CanaryDisabledError(CanaryError):
    """Raised when mint is called but the canary feature is disabled."""


class CanaryMisconfiguredError(CanaryError):
    """Raised when the canary feature is enabled but the secret is missing."""


class CanaryManager:
    """Mint and verify per-session canary tokens (B3).

    Construct once per process and store on ``app.state`` (see ``main.py``).
    The manager is stateless beyond its settings; it is safe to share across
    requests and workers.
    """

    def __init__(self, settings: CanarySettings) -> None:
        self._settings = settings
        self._secret: bytes = (settings.secret or "").encode("utf-8")

    @property
    def enabled(self) -> bool:
        """True if the canary feature is enabled AND a secret is configured."""
        return self._settings.enabled and bool(self._secret)

    @staticmethod
    def _encode(mac_bytes: bytes) -> str:
        """Base32-encode the HMAC bytes, strip padding, upper-case."""
        return base64.b32encode(mac_bytes).decode("ascii").rstrip("=").upper()

    def _derive(self, session_id: str, label: str) -> str:
        """Derive one canary token for (session_id, label).

        Raises if the manager is not configured (no secret). Callers that
        want safe behaviour (detection) should guard with ``enabled`` first.
        """
        if not self._secret:
            raise CanaryMisconfiguredError(
                "Canary feature is enabled but NEURALGUARD_CANARY_SECRET is not set."
            )
        mac = hmac.new(self._secret, f"{session_id}|{label}".encode(), hashlib.sha256).digest()
        return f"{_TOKEN_PREFIX}{self._encode(mac[:_HMAC_BYTES])}"

    def mint(self, session_id: str, count: int | None = None) -> list[str]:
        """Mint one or more canary tokens for a session.

        Args:
            session_id: The session to bind the canaries to. Must be non-empty.
            count: Number of distinct canaries (labels A..). Defaults to the
                configured ``token_count``; clamped to ``[1, _MAX_LABELS]``.

        Returns:
            List of canary token strings, in label order (A, B, ...).

        Raises:
            CanaryDisabledError: feature disabled.
            CanaryMisconfiguredError: enabled but no secret.
            ValueError: session_id empty/whitespace.
        """
        if not self._settings.enabled:
            raise CanaryDisabledError("Canary feature is disabled.")
        if not self._secret:
            raise CanaryMisconfiguredError(
                "Canary feature is enabled but NEURALGUARD_CANARY_SECRET is not set."
            )
        sid = (session_id or "").strip()
        if not sid:
            raise ValueError("session_id must not be empty.")
        n = self._settings.token_count if count is None else count
        n = max(1, min(int(n), _MAX_LABELS))
        return [self._derive(sid, _LABELS[i]) for i in range(n)]

    def check_leak(self, session_id: str, output: str) -> str | None:
        """Return the leaked canary token if one appears in ``output``, else None.

        Safe-by-default: returns ``None`` (no leak detected) when the manager
        is disabled or misconfigured, or when the session_id is empty — the
        canary is an additive detection signal, not a gate. Never raises.

        Recomputes the bounded label set (A..H) for the session and substring-
        searches the output. O(_MAX_LABELS) HMACs + substring scans per call.
        """
        if not self._settings.enabled or not self._secret:
            return None
        sid = (session_id or "").strip()
        if not sid:
            return None
        for label in _LABELS:
            try:
                tok = self._derive(sid, label)
            except CanaryError:
                # Misconfiguration surfaced by _derive — treat as no leak
                # (safe-by-default) and let the caller's own startup gate
                # refuse to serve in production.
                return None
            if tok and tok in output:
                return tok
        return None
