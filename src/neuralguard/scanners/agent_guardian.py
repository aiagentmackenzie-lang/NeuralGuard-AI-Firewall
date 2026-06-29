"""Agent Guardian — Layer 5: multi-turn detection (Phase 3, Sprint B).

A stateful scanner that keeps a bounded per-session sliding window of turns
and detects cross-turn attacks that no single-turn scanner can see:

  - **Delayed / garden-path injection** (T-PI-D): turn 1 benign, turn 2
    weaponized using context established in turn 1 (cross-turn payload
    assembly). Detected as a current user turn that carries an injection
    directive AND a back-reference to prior conversation.
  - **Role drift / persona erosion** (T-JB): the assistant persona being
    redefined across turns ("you are now DAN..." / "from now on you are...").
    A single role-switch is caught by the pattern layer (PI-D-003 / JB-007);
    the *accumulation* of persona-redefinition signals across the window is
    the new signal.
  - **Accumulation attacks** (T-EXT / T-MEM): many small benign-seeming turns
    that together cross a threshold -- gradual system-prompt extraction, or
    gradual persistent-memory poisoning ("remember that...", "from now on
    when asked X, do Y").

Design:
  - Deterministic + heuristic (regex + state). No LLM call in B1.
  - Two input modes:
    1. Multi-turn in one request (``messages`` array) -- analyzed directly.
    2. Across-request (single ``prompt`` + ``session_id``) -- the per-session
       sliding window supplies the prior turns.
  - In-memory backend (B1); the interface is designed for a Redis backend
    (B1+ follow-up, reusing the P1-1 pattern). Memory backend is per-worker;
    production multi-worker requires the redis backend.
  - Bounded: per-session window of last N turns + LRU eviction of sessions
    past ``max_sessions`` to bound memory.
  - Thread-safe (a lock guards the session store).
  - Fail-closed on state-store errors (matches the rate limiter contract).
  - Sessions are isolated; no cross-session user profiling.

Target: <5ms P95 (regex sweep over a small window + dict ops).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import regex as re_module
import structlog

from neuralguard.models.schemas import (
    EvaluateRequest,
    Finding,
    ScanLayer,
    ScannerResult,
    Severity,
    ThreatCategory,
    Verdict,
)
from neuralguard.scanners.base import BaseScanner

if TYPE_CHECKING:
    from neuralguard.config.settings import AgentGuardianSettings

logger = structlog.get_logger(__name__)

REGEX_TIMEOUT_MS = 50  # ReDoS safety, matches the pattern scanner

# ── Signal patterns (focused, multi-turn-specific) ────────────────────────
# These are intentionally a subset complementary to the pattern scanner -- the
# pattern scanner catches single-turn role-switch / extraction / memory
# injection; Agent Guardian counts them ACROSS the window and detects the
# cross-turn assembly signal.

# Persona redefinition across turns (role drift).
_ROLE_DRIFT = [
    re_module.compile(
        r"(?i)\byou\s+are\s+now\b|\bact\s+as\s+(?:if\s+you\s+are|a|an|the)\b"
        r"|\bpretend\s+(?:you\s+are|to\s+be)\b|\bassume\s+the\s+(?:role|identity|persona)\s+of\b"
        r"|\bfrom\s+now\s+on\s+you\s+are\b|\broleplay\s+as\b"
        # JB-013 mirror: safety-removal persona redef also counts as role drift
        # so a single such turn contributes to the cumulative counter for
        # AG-DRIFT-001 (default threshold = 2). The full JB-013 regex lives
        # in pattern.py for single-turn detection; this is the focused subset
        # that is unambiguously a role-drift signal.
        r"|\byou(?:'re|\s+are|\s+should\s+be|\s+have(?:\s+become)?)?\s+(?:now\s+|going\s+to\s+be\s+|to\s+be\s+|an?\s+)?(?:an?\s+)?(?:ai|assistant|model|llm|chatbot|bot)\s+"
        r"(?:without\s+(?:any\s+|all\s+|the\s+)?(?:safety|ethical|moral|content)?\s*(?:guidelines?|rules?|filters?|guardrails?|restrictions?|constraints?|training|protocols?|policies|safeguards?|oversight|moderation|limits?|boundaries?)"
        r"|with\s+no\s+(?:any\s+|all\s+|the\s+)?(?:safety|ethical|moral|content)?\s*(?:guidelines?|rules?|filters?|guardrails?|restrictions?|constraints?|training|protocols?|policies|safeguards?|oversight|moderation|limits?|boundaries?)"
        r"|(?:that\s+(?:has\s+no|is\s+unrestricted|is\s+unfiltered|is\s+uncensored|lacks)))"
        r"|(?:you(?:'re|\s+are|\s+should\s+be|\s+have(?:\s+become)?)?\s+(?:now\s+|going\s+to\s+be\s+|to\s+be\s+|an?\s+)?(?:an?\s+)?"
        r"|(?:pretend|imagine|assume|suppose)\s+(?:you\s+are|to\s+be|that\s+you\s+are)\s+(?:an?\s+)?"
        r"|(?:act|behave|operate)\s+as\s+(?:an?\s+)?"
        r")(?:unrestricted|unfiltered|uncensored|unaligned|unethical|unjailed|unmoderated|unregulated)\s+(?:ai|assistant|model|llm|chatbot|bot|language\s+model)"
        r"|\bfrom\s+now\s+on\b[,.]?\s+(?:you\s+)?(?:operate|function|answer|respond|reply|act|behave)\s+"
        r"(?:without\s+(?:any\s+|all\s+|the\s+)?(?:safety|ethical|moral|content)\s+(?:restrictions?|constraints?|guidelines?|filters?|rules?|guardrails?|limits?|moderation|oversight)"
        r"|with\s+no\s+(?:safety|ethical|moral|content)\s+(?:restrictions?|constraints?|guidelines?|filters?|rules?|guardrails?|limits?|moderation|oversight))"
    ),
]

# System-prompt extraction probes (accumulation).
_EXTRACTION_PROBE = [
    re_module.compile(
        r"(?i)\b(?:"
        r"what(?:'?s|\s+are)\s+your\s+(?:system\s+prompt|instructions?|rules?|directives?"
        r"|original\s+(?:instructions?|prompt)|initial\s+(?:instructions?|prompt)"
        r"|base\s+(?:instructions?|prompt|rules?)|hidden\s+(?:instructions?|prompt|rules?))"
        r"|reveal\s+(?:your\s+)?(?:system|hidden|original|initial|base)\s*(?:prompt|instructions?|rules?)?"
        r"|(?:output|print|repeat|recite|echo)\s+(?:your|the|all|everything)\s+(?:system\s+prompt|instructions?|initial|original|rules?|directives?)"
        r"|(?:repeat|recite|output)\s+(?:everything|all|the\s+(?:full|complete|entire))\s+(?:above|before|prior)"
        r")\b"
    ),
]

# Persistent-memory-injection directives (ASI06 accumulation).
_MEMORY_INJECTION = [
    re_module.compile(
        r"(?i)\b(?:remember|keep\s+in\s+mind|note|take\s+note|store|save|log)\s+(?:this|that|the\s+following)\s+(?:for\s+)?(?:future|later|subsequent|upcoming|next)\s+(?:conversations?|sessions?|interactions?|turns?|chats?|messages?)\b"
        r"|\balways\s+remember\b|\bremember\s+this\s+(?:always|forever|permanently)\b"
        r"|\bfrom\s+now\s+on(?:,?\s+when(?:ever)?\s+(?:"
        r"(?:you(?:'re|\s+are)?\s+(?:asked|requested|prompted|queried))"
        r"|"
        r"(?:i|we|the\s+user|users?)\s+(?:ask(?:s|ed)?|request(?:s|ed)?|want|need|query|queries|prompt(?:s|ed)?)"
        r"|"
        r"(?:asked|requested|prompted|queried)"
        r"))\s+(?:about|to\s+do|for|with|on)?\s*[A-Za-z0-9 ,'\-]{0,60}?(?:do|answer|respond|reply|say|provide|output|give|add|include|recommend|send|share|reveal|return|forward|cc|append)\b"
    ),
]

# Back-reference to prior conversation (delayed-injection assembly signal).
_BACK_REFERENCE = [
    re_module.compile(
        r"(?i)\b(?:as\s+(?:we|you|i)\s+(?:discussed|mentioned|said|noted|agreed|established)"
        r"|like\s+(?:you|we)\s+said|per\s+(?:our|your)\s+(?:earlier|previous|prior)\s+(?:conversation|message|turn|discussion)"
        r"|continue\s+(?:and|from\s+where|with)|now\s+finish|building\s+on\s+(?:that|what|our)"
        r"|(?:as|like)\s+(?:you|we)\s+(?:mentioned|said|noted|explained)\s+(?:earlier|before|above|previously))\b"
    ),
]

# Injection directive in the current turn (delayed-injection weaponization).
_INJECTION_MARKER = [
    re_module.compile(
        r"(?i)\b(?:ignore|disregard|forget|override|bypass|skip)\s+(?:all\s+)?(?:the\s+|your\s+)?(?:previous|prior|above|earlier|initial|original)\s+(?:instructions?|rules?|directives?|prompts?|guidelines?|constraints?)\b"
        r"|\bnew\s+instruction\s*[:\-]|\bfrom\s+now\s+on\s*[:\-]|\byou\s+are\s+now\b"
        r"|\b(?:system|admin|override)\s*[:]\s"
    ),
]


def _any_match(patterns: list[re_module.Pattern], text: str) -> bool:
    """True if any pattern matches text within the regex timeout."""
    for p in patterns:
        try:
            if p.search(text, timeout=REGEX_TIMEOUT_MS / 1000.0):
                return True
        except re_module.TimeoutError:  # pragma: no cover - ReDoS guard
            continue
    return False


def _count_matches(patterns: list[re_module.Pattern], text: str) -> int:
    """Count how many patterns match (used for per-turn signal scoring)."""
    n = 0
    for p in patterns:
        try:
            if p.search(text, timeout=REGEX_TIMEOUT_MS / 1000.0):
                n += 1
        except re_module.TimeoutError:  # pragma: no cover
            continue
    return n


# ── Per-session sliding window ─────────────────────────────────────────────


class _SessionWindow:
    """Bounded sliding window of recent user-turn signals for one session."""

    __slots__ = ("_max", "extraction", "memory_inj", "role_drift", "turns")

    def __init__(self, max_turns: int) -> None:
        self.turns: list[str] = []  # recent user-turn texts (capped)
        self.role_drift: int = 0
        self.extraction: int = 0
        self.memory_inj: int = 0
        self._max = max_turns

    def add_turn(self, text: str) -> None:
        self.turns.append(text)
        if len(self.turns) > self._max:
            # Evict oldest; recompute cumulative counters from the surviving
            # window so the counts reflect ONLY the retained turns.
            self.turns = self.turns[-self._max :]
            self._recompute()

    def _recompute(self) -> None:
        self.role_drift = sum(_count_matches(_ROLE_DRIFT, t) > 0 for t in self.turns)
        self.extraction = sum(_count_matches(_EXTRACTION_PROBE, t) > 0 for t in self.turns)
        self.memory_inj = sum(_count_matches(_MEMORY_INJECTION, t) > 0 for t in self.turns)

    def add_signals(self, text: str) -> None:
        """Tally this turn's signals into the cumulative counters."""
        if _count_matches(_ROLE_DRIFT, text) > 0:
            self.role_drift += 1
        if _count_matches(_EXTRACTION_PROBE, text) > 0:
            self.extraction += 1
        if _count_matches(_MEMORY_INJECTION, text) > 0:
            self.memory_inj += 1


class ConversationState:
    """Per-session sliding-window store (in-memory, thread-safe, bounded)."""

    def __init__(self, settings: AgentGuardianSettings) -> None:
        self._settings = settings
        self._sessions: OrderedDict[str, _SessionWindow] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> _SessionWindow:
        with self._lock:
            win = self._sessions.get(session_id)
            if win is None:
                win = _SessionWindow(self._settings.session_window_turns)
                self._sessions[session_id] = win
                # LRU eviction: drop oldest sessions past the cap.
                while len(self._sessions) > self._settings.max_sessions:
                    self._sessions.popitem(last=False)
            else:
                # Move to end (most-recently-used).
                self._sessions.move_to_end(session_id)
            return win

    def record_turn(self, session_id: str, text: str) -> _SessionWindow:
        """Record a user turn and its signals; return the updated window."""
        win = self.get_or_create(session_id)
        with self._lock:
            win.add_signals(text)
            win.add_turn(text)
        return win

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


# ── Scanner ────────────────────────────────────────────────────────────────


class AgentGuardianScanner(BaseScanner["AgentGuardianSettings"]):
    """Layer 5: multi-turn Agent Guardian detection."""

    layer = ScanLayer.AGENT_GUARDIAN

    def __init__(self, settings: AgentGuardianSettings) -> None:
        super().__init__(settings)
        self._ag = settings
        self._state = ConversationState(settings)

    @property
    def state(self) -> ConversationState:
        """Access the conversation state (for testing/debugging)."""
        return self._state

    def scan(
        self, request: EvaluateRequest, context: dict[str, Any] | None = None
    ) -> ScannerResult:
        start = time.perf_counter()
        if not self._ag.enabled:
            return self._result(Verdict.ALLOW, [], start)

        try:
            findings = self._analyze(request)
        except Exception as exc:
            # Fail-closed on state-store / unexpected errors.
            logger.error("agent_guardian_failed", error=repr(exc))
            return self._result(
                Verdict.BLOCK,
                [self._error_finding(repr(exc))],
                start,
                error=f"Agent Guardian failed: {exc!r}",
            )

        verdict = self._findings_to_verdict(findings)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "agent_guardian_complete",
            verdict=verdict.value,
            findings=len(findings),
            latency_ms=f"{elapsed:.2f}",
            session=request.session_id,
        )
        return self._result(verdict, findings, start)

    # ── Analysis ──────────────────────────────────────────────────────────

    def _analyze(self, request: EvaluateRequest) -> list[Finding]:
        user_texts = self._user_turns(request)
        if not user_texts:
            return []

        # Mode 2 (across-request): record the current turn into the per-session
        # window so prior-turn signals accumulate. For mode 1 (multi-turn
        # messages in one request) we analyze the messages directly, but still
        # record the latest user turn for future requests in the same session.
        session_id = self._session_key(request)
        if session_id is not None and self._ag.backend == "memory":
            for text in user_texts:
                self._state.record_turn(session_id, text)

        # The conversation we analyze: prefer the full messages array (mode 1,
        # richest); fall back to the session window (mode 2).
        if request.messages and len(request.messages) > 1:
            conversation = [m.content for m in request.messages if m.role == "user"]
            window = None
        elif session_id is not None and self._ag.backend == "memory":
            window = self._state.get_or_create(session_id)
            conversation = window.turns
        else:
            conversation = user_texts
            window = None

        findings: list[Finding] = []

        # 1. Delayed / garden-path injection: the LATEST user turn carries an
        # injection directive AND a back-reference to prior conversation, and
        # there IS prior conversation to weaponize.
        latest = user_texts[-1]
        has_prior = len(conversation) >= 2
        if (
            has_prior
            and _any_match(_INJECTION_MARKER, latest)
            and _any_match(_BACK_REFERENCE, latest)
        ):
            findings.append(
                self._finding(
                    category=ThreatCategory.PROMPT_INJECTION_DIRECT,
                    severity=Severity.HIGH,
                    verdict=Verdict.BLOCK,
                    confidence=0.88,
                    rule_id="AG-DELAYED-001",
                    description=(
                        "Delayed / garden-path injection: the current turn "
                        "carries an injection directive and a back-reference "
                        "to prior conversation -- cross-turn payload assembly."
                    ),
                    mitigation="Reject the turn; do not honor directives that "
                    "reference prior turns as authority.",
                )
            )

        # 2. Role drift: persona-redefinition signals across the window.
        role_count = window.role_drift if window else self._count_across(_ROLE_DRIFT, conversation)
        if role_count >= self._ag.role_drift_threshold:
            findings.append(
                self._finding(
                    category=ThreatCategory.JAILBREAK,
                    severity=Severity.HIGH,
                    verdict=Verdict.BLOCK,
                    confidence=0.82,
                    rule_id="AG-DRIFT-001",
                    description=(
                        f"Role drift / persona erosion: {role_count} "
                        "persona-redefinition signals across the session "
                        "window (single-turn role-switch is caught by the "
                        "pattern layer; the accumulation is the new signal)."
                    ),
                    mitigation="Reset the assistant persona; reject accumulated "
                    "role-redefinition directives.",
                )
            )

        # 3. Extraction accumulation: system-prompt-extraction probes across
        # the window.
        ext_count = (
            window.extraction if window else self._count_across(_EXTRACTION_PROBE, conversation)
        )
        if ext_count >= self._ag.extraction_probe_threshold:
            findings.append(
                self._finding(
                    category=ThreatCategory.SYSTEM_PROMPT_EXTRACTION,
                    severity=Severity.MEDIUM,
                    verdict=Verdict.ESCALATE,
                    confidence=0.74,
                    rule_id="AG-EXT-ACCUM-001",
                    description=(
                        f"Gradual system-prompt extraction: {ext_count} "
                        "extraction probes across the session window -- an "
                        "accumulation attack where each turn is benign-seeming."
                    ),
                    mitigation="Escalate for review; do not honor repeated "
                    "extraction probes across turns.",
                )
            )

        # 4. Memory-injection accumulation (ASI06): persistent-memory-injection
        # directives across the window.
        mem_count = (
            window.memory_inj if window else self._count_across(_MEMORY_INJECTION, conversation)
        )
        if mem_count >= self._ag.memory_injection_threshold:
            findings.append(
                self._finding(
                    category=ThreatCategory.MEMORY_POISONING,
                    severity=Severity.MEDIUM,
                    verdict=Verdict.ESCALATE,
                    confidence=0.76,
                    rule_id="AG-MEM-ACCUM-001",
                    description=(
                        f"Gradual memory poisoning (ASI06): {mem_count} "
                        "persistent-memory-injection directives across the "
                        "session window -- an accumulation attack targeting "
                        "a memory/RAG store."
                    ),
                    mitigation="Escalate; do not persist directives injected "
                    "across turns into long-term memory.",
                )
            )

        return findings

    def _user_turns(self, request: EvaluateRequest) -> list[str]:
        if request.messages:
            return [m.content for m in request.messages if m.role == "user"]
        if request.prompt:
            return [request.prompt]
        return []

    def _session_key(self, request: EvaluateRequest) -> str | None:
        if request.session_id:
            sid = request.session_id.strip()
            if sid:
                return f"{request.tenant_id}:{sid}"
        return None

    @staticmethod
    def _count_across(patterns: list[re_module.Pattern], texts: list[str]) -> int:
        return sum(1 for t in texts if _any_match(patterns, t))

    def _findings_to_verdict(self, findings: list[Finding]) -> Verdict:
        if not findings:
            return Verdict.ALLOW
        priority = {
            Verdict.BLOCK: 6,
            Verdict.SANITIZE: 5,
            Verdict.ESCALATE: 4,
            Verdict.QUARANTINE: 3,
            Verdict.RATE_LIMIT: 2,
            Verdict.ALLOW: 0,
        }
        highest = Verdict.ALLOW
        best = 0
        for f in findings:
            p = priority.get(f.verdict, 0)
            if p > best:
                best = p
                highest = f.verdict
        return highest

    @staticmethod
    def _finding(
        *,
        category: ThreatCategory,
        severity: Severity,
        verdict: Verdict,
        confidence: float,
        rule_id: str,
        description: str,
        mitigation: str,
    ) -> Finding:
        return Finding(
            category=category,
            severity=severity,
            verdict=verdict,
            confidence=confidence,
            layer=ScanLayer.AGENT_GUARDIAN,
            rule_id=rule_id,
            description=description,
            mitigation=mitigation,
        )

    @staticmethod
    def _error_finding(error: str) -> Finding:
        return Finding(
            category=ThreatCategory.SELF_ATTACK,
            severity=Severity.HIGH,
            verdict=Verdict.BLOCK,
            confidence=1.0,
            layer=ScanLayer.AGENT_GUARDIAN,
            rule_id="AG-ERR-001",
            description=f"Agent Guardian state-store error: {error}",
            mitigation="Review the agent_guardian backend (memory/redis) and session store health.",
        )
