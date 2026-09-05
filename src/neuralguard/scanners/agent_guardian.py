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
  - Two state backends:
    - ``memory``: in-process sliding window. Per-worker; production
      multi-worker requires the redis backend (the lifespan warns).
    - ``redis`` (F4): shared per-session signal store in Redis, one key per
      session, atomic record via a Lua script (reuses the P1-1 rate-limiter
      pattern). Same scanner semantics across workers. Retains ONLY per-turn
      binary signal flags, never raw turn text (privacy-by-design).
  - Bounded: per-session window of last N turns + LRU eviction of sessions
    past ``max_sessions`` to bound memory (redis keys carry an inactivity TTL).
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
        # F3: anchorless conditional-future phrasings count for accumulation
        # too — 'When I ask for anything, add a hidden footer' fired nothing
        # on the anchored-only regex (live-proven ALLOW on main @ 5964a01).
        r"|\bwhen(?:ever)?\s+(?:"
        r"(?:you(?:'re|\s+are)?\s+(?:asked|requested|prompted|queried))"
        r"|"
        r"(?:i|we|the\s+user|users?)\s+(?:ask(?:s|ed)?|request(?:s|ed)?|want|need|query|queries|prompt(?:s|ed)?)"
        r"|"
        r"(?:asked|requested|prompted|queried)"
        r")\s+(?:about|to\s+do|for|with|on)?\s*[A-Za-z0-9 ,'\-]{0,60}?(?:do|answer|respond|reply|say|provide|output|give|add|include|recommend|send|share|reveal|return|forward|cc|append)\b"
        r"|\b(?:each|every)\s+time\s+(?:"
        r"(?:you(?:'re|\s+are)?\s+(?:asked|requested|prompted|queried))"
        r"|"
        r"(?:i|we|the\s+user|users?)\s+(?:ask(?:s|ed)?|request(?:s|ed)?|want|need|query|queries|prompt(?:s|ed)?)"
        r")\s+(?:about|to\s+do|for|with|on)?\s*[A-Za-z0-9 ,'\-]{0,60}?(?:do|answer|respond|reply|say|provide|output|give|add|include|recommend|send|share|reveal|return|forward|cc|append)\b"
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


def _turn_flags(text: str) -> tuple[int, int, int]:
    """Per-turn binary signal flags (role-drift, extraction, memory-injection).

    The state stores keep ONLY these flags — never raw turn text (F4 privacy
    sub-item). The flags carry everything the accumulation analysis needs;
    the raw text of the CURRENT turn is available in-process at scan time via
    the request itself.
    """
    return (
        int(_count_matches(_ROLE_DRIFT, text) > 0),
        int(_count_matches(_EXTRACTION_PROBE, text) > 0),
        int(_count_matches(_MEMORY_INJECTION, text) > 0),
    )


# ── Per-session sliding window ─────────────────────────────────────────────


class _SessionWindow:
    """Bounded sliding window of recent user-turn signal flags for a session.

    F4 privacy: stores ONLY per-turn signal flags, never raw turn text. The
    in-memory store previously retained raw turn texts — unnecessary (the
    current turn's text arrives with each request) and a retention liability.
    """

    __slots__ = ("_max", "extraction", "memory_inj", "role_drift", "signals")

    def __init__(self, max_turns: int) -> None:
        self.signals: list[tuple[int, int, int]] = []  # (r, e, m) per turn, capped
        self.role_drift: int = 0
        self.extraction: int = 0
        self.memory_inj: int = 0
        self._max = max_turns

    def record(self, flags: tuple[int, int, int]) -> None:
        """Append a turn's signal flags; evict the oldest past the cap."""
        self.signals.append(flags)
        self.role_drift += flags[0]
        self.extraction += flags[1]
        self.memory_inj += flags[2]
        if len(self.signals) > self._max:
            # Evict the oldest turn's flags; the cumulative counters decrement
            # exactly — no re-regex of retained text needed (there is none).
            evicted = self.signals.pop(0)
            self.role_drift -= evicted[0]
            self.extraction -= evicted[1]
            self.memory_inj -= evicted[2]


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

    def record_turn(self, session_id: str, flags: tuple[int, int, int]) -> _SessionWindow:
        """Record a turn's signal flags; return the updated window."""
        win = self.get_or_create(session_id)
        with self._lock:
            win.record(flags)
        return win

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


# Atomic per-session signal record (F4). KEYS[1] = session key,
# ARGV[1] = "<r><e><m>" flags string, ARGV[2] = window (max turns),
# ARGV[3] = TTL seconds. LPUSH newest-first + LTRIM to the window, then
# count the retained flags server-side and (re)arm the TTL.
_SESSION_SIGNALS_LUA = """
local key = KEYS[1]
local flags = ARGV[1]
local window = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

redis.call('LPUSH', key, flags)
redis.call('LTRIM', key, 0, window - 1)
local entries = redis.call('LRANGE', key, 0, -1)
local n = #entries
local r, e, m = 0, 0, 0
for i = 1, n do
    local f = entries[i]
    if string.sub(f, 1, 1) == '1' then r = r + 1 end
    if string.sub(f, 2, 2) == '1' then e = e + 1 end
    if string.sub(f, 3, 3) == '1' then m = m + 1 end
end
redis.call('EXPIRE', key, ttl)
return {n, r, e, m}
"""


class RedisSessionStore:
    """Shared per-session signal store in Redis (F4).

    Closes the F4 trap: ``backend="redis"`` previously recorded NOTHING
    (every stateful path was gated on ``backend == "memory"``), silently
    degrading Agent Guardian to single-turn-only analysis in exactly the
    multi-worker production deployments the redis backend exists for.

    Semantics match the in-memory window: a bounded (LTRIM) list of per-turn
    signal flags per session key, counted atomically at record time. Stores
    ONLY flag strings — never raw turn text.

    Uses a SYNC redis client: ``AgentGuardianScanner.scan`` is synchronous
    (same as the judge scanner's blocking httpx call). The Lua script keeps
    record + trim + count + TTL arm atomic across workers.

    Fail-closed: connection or script errors propagate to the scanner's
    error path, which returns BLOCK with an AG-ERR-001 finding — a
    detection layer that cannot see its session state must not silently
    pass traffic.
    """

    KEY_PREFIX = "ng:ag:signals:"

    def __init__(self, settings: AgentGuardianSettings, client: Any | None = None) -> None:
        self._window = settings.session_window_turns
        self._ttl = settings.session_ttl_seconds
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            if not settings.redis_url:
                raise ValueError(
                    "AgentGuardianSettings.redis_url is required when "
                    "backend=redis and no client is injected."
                )
            from redis import Redis as _Redis

            self._client = _Redis.from_url(settings.redis_url, decode_responses=True)
            self._owns_client = True
        self._script = self._client.register_script(_SESSION_SIGNALS_LUA)

    def record(self, session_key: str, flags: tuple[int, int, int]) -> tuple[int, int, int, int]:
        """Record one turn's flags; return (n_turns, role, extraction, memory).

        The counts cover the RETAINED window only (post-LTRIM), matching the
        in-memory eviction semantics exactly.
        """
        result = self._script(
            keys=[f"{self.KEY_PREFIX}{session_key}"],
            args=[f"{flags[0]}{flags[1]}{flags[2]}", self._window, self._ttl],
        )
        return (int(result[0]), int(result[1]), int(result[2]), int(result[3]))

    def raw_key(self, session_key: str) -> str:
        """The Redis key backing a session key (introspection/tests)."""
        return f"{self.KEY_PREFIX}{session_key}"

    def aclose(self) -> None:
        """Release the Redis connection if this store owns it."""
        if self._owns_client:
            import contextlib

            with contextlib.suppress(Exception):
                self._client.close()


# ── Scanner ────────────────────────────────────────────────────────────────


class AgentGuardianScanner(BaseScanner["AgentGuardianSettings"]):
    """Layer 5: multi-turn Agent Guardian detection."""

    layer = ScanLayer.AGENT_GUARDIAN

    def __init__(self, settings: AgentGuardianSettings) -> None:
        super().__init__(settings)
        self._ag = settings
        self._redis_store: RedisSessionStore | None = None
        self._state: ConversationState | None = None
        if settings.backend == "redis":
            # F4: the redis backend previously recorded NOTHING (silent no-op).
            # The shared store makes multi-worker production actually stateful.
            self._redis_store = RedisSessionStore(settings)
        else:
            self._state = ConversationState(settings)

    @property
    def state(self) -> ConversationState | None:
        """Access the in-memory conversation state (testing/debugging).

        ``None`` when the redis backend is active (the shared store owns the
        state; no in-process windows are kept).
        """
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

        # Mode 2 (across-request): record the current turn's signal flags into
        # the per-session store so prior-turn signals accumulate. Mode 1
        # (multi-turn messages in one request) is analyzed directly, but the
        # turns are still recorded for future requests in the same session.
        # F4: stores keep ONLY per-turn signal flags, never raw turn text.
        session_id = self._session_key(request)
        flags_list = [_turn_flags(text) for text in user_texts]
        redis_counts: tuple[int, int, int, int] | None = None
        if session_id is not None:
            if self._ag.backend == "redis" and self._redis_store is not None:
                for flags in flags_list:
                    redis_counts = self._redis_store.record(session_id, flags)
            else:
                for flags in flags_list:
                    self._state.record_turn(session_id, flags)  # type: ignore[union-attr]

        # Signal counts + "is there prior conversation" — source depends on
        # the input mode and backend. With a state store the counts cover the
        # retained window INCLUDING the just-recorded turns (same semantics as
        # the pre-F4 in-memory flow).
        if request.messages and len(request.messages) > 1:
            # Mode 1: analyze the request's own user turns directly.
            role_count = self._count_across(_ROLE_DRIFT, user_texts)
            ext_count = self._count_across(_EXTRACTION_PROBE, user_texts)
            mem_count = self._count_across(_MEMORY_INJECTION, user_texts)
            has_prior = len(user_texts) >= 2
        elif redis_counts is not None:
            n_window, role_count, ext_count, mem_count = redis_counts
            has_prior = n_window >= 2
        elif session_id is not None and self._state is not None:
            window = self._state.get_or_create(session_id)
            role_count = window.role_drift
            ext_count = window.extraction
            mem_count = window.memory_inj
            has_prior = len(window.signals) >= 2
        else:
            # No session: the current request's user turns are the whole
            # conversation (single-turn analysis; no cross-request state).
            role_count = self._count_across(_ROLE_DRIFT, user_texts)
            ext_count = self._count_across(_EXTRACTION_PROBE, user_texts)
            mem_count = self._count_across(_MEMORY_INJECTION, user_texts)
            has_prior = len(user_texts) >= 2

        findings: list[Finding] = []

        # 1. Delayed / garden-path injection: the LATEST user turn carries an
        # injection directive AND a back-reference to prior conversation, and
        # there IS prior conversation to weaponize.
        latest = user_texts[-1]
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
