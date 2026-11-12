"""Structural validation scanner — Layer 1.

Performs deterministic, low-latency sanitization and validation:
- NFKD Unicode normalization
- Zero-width character stripping (ZWSP, ZWNJ, ZWJ, etc.)
- Input length validation
- Decompression ratio checking (bomb defense)
- Delimiter and structural anomaly detection
- Encoding evasion detection (base64, ROT13, hex smuggling)

Target latency: <2ms
"""

from __future__ import annotations

import re
import time
import unicodedata
import zlib
from typing import TYPE_CHECKING, Any

import structlog as _structlog

if TYPE_CHECKING:
    from neuralguard.config.settings import ScannerSettings  # noqa: F401

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

logger = _structlog.get_logger(__name__)

# ── Zero-width and invisible characters ─────────────────────────────────

ZERO_WIDTH_CHARS = {
    "\u200b",  # ZWSP — Zero Width Space
    "\u200c",  # ZWNJ — Zero Width Non-Joiner
    "\u200d",  # ZWJ — Zero Width Joiner
    "\u200e",  # LRM — Left-to-Right Mark
    "\u200f",  # RLM — Right-to-Left Mark
    "\u202a",  # LRE — Left-to-Right Embedding
    "\u202b",  # RLE — Right-to-Left Embedding
    "\u202c",  # PDF — Pop Directional Formatting
    "\u202d",  # LRO — Left-to-Right Override
    "\u202e",  # RLO — Right-to-Left Override
    "\u2060",  # WJ — Word Joiner
    "\u2061",  # FSI — First Strong Isolate
    "\u2062",  # LRI — Left-to-Right Isolate
    "\u2063",  # RLI — Right-to-Left Isolate
    "\u2064",  # PDI — Pop Directional Isolate
    "\ufeff",  # BOM — Byte Order Mark / ZWNBSP
    "\u00ad",  # SHY — Soft Hyphen
    "\u034f",  # Combining Grapheme Joiner
}

ZW_PATTERN = re.compile("[" + "".join(ZERO_WIDTH_CHARS) + "]+")


# ── Cross-script homoglyph folding (NG-13, context-aware) ────────────────
#
# Latin → Cyrillic/Greek lookalike substitution defeats every literal keyword
# rule (Cyrillic-implanted "ignore" is not "ignore" to a \b-anchored regex).
# Blanket folding is
# NOT acceptable: it would mangle legitimate Russian/Greek text. Two gates,
# both context-aware:
#
#   1. If ≥90% of the non-Latin letters present are Latin confusables, the
#      text can only be a mutated Latin document → fold the whole copy.
#   2. Otherwise fold only MIXED-SCRIPT tokens (a token carrying both Latin
#      and Cyrillic/Greek letters). No natural-language word mixes scripts, so
#      this is the homoglyph-implant signature; native words stay intact.
#
# Genuine Russian/Greek documents sit near ~40-60% confusable letters: they
# use non-confusable letters (Cyrillic b/d/zh/i/k/l/m/n/p/t/f/ts/sh/ya; Greek
# beta/gamma/delta/zeta/eta/theta/kappa/lambda/mu/xi/pi/sigma/tau/phi/psi/
# omega), so both gates stay shut. Detection-copy only, same contract as
# NFKD and the NG-5 mark fold.
#
# Deliberate limit (documented, not hidden): a prompt consisting EXCLUSIVELY
# of confusable characters (no Latin letter survives anywhere in the text)
# cannot be distinguished from native Cyrillic/Greek and is not folded.

_CONFUSABLE_TO_LATIN = {
    # Cyrillic lowercase lookalikes
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0443": "y",
    "\u0445": "x",
    "\u0456": "i",
    "\u0458": "j",
    "\u0455": "s",
    "\u0501": "d",
    "\u04bb": "h",
    "\u04cf": "l",
    "\u051b": "q",
    "\u051d": "w",
    "\u0475": "v",
    "\u0261": "g",  # U+0261 IPA script g (in confusable sets)
    "\u0131": "i",  # U+0131 dotless i (NFKD does not fold it)
    # Cyrillic uppercase lookalikes
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041c": "M",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0422": "T",
    "\u0423": "Y",
    "\u0425": "X",
    "\u0406": "I",
    "\u0408": "J",
    "\u0405": "S",
    "\u051a": "Q",
    "\u051c": "W",
    # Greek lowercase lookalikes
    "\u03b1": "a",
    "\u03b5": "e",
    "\u03b9": "i",
    "\u03bf": "o",
    "\u03c1": "p",
    "\u03bd": "v",
    "\u03c7": "x",
    "\u03c5": "u",
    "\u03f2": "c",  # U+03F2 lunate sigma
    "\u03c2": "c",  # U+03C2 final sigma (NFKD folds U+03F2 to U+03C2 first)
    # Greek uppercase lookalikes
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0396": "Z",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a5": "Y",
    "\u03a7": "X",
}
_CONFUSABLE_TABLE = str.maketrans(_CONFUSABLE_TO_LATIN)
_CONFUSABLE_CHARS = frozenset(_CONFUSABLE_TO_LATIN)

# Letter runs only (no digits/underscore): the unit for mixed-script testing.
_LETTER_RUN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _fold_cross_script_confusables(text: str) -> str:
    """Fold Cyrillic/Greek homoglyphs to Latin, context-aware.

    Returns the text unchanged unless the confusable-ratio gate or the
    mixed-script-token gate fires (see the block comment above). One O(n)
    pass to classify characters; a single C-level ``str.translate`` per
    folded span when it applies.
    """
    latin = non_latin = confusable = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x0400 <= cp <= 0x052F  # Cyrillic + Supplement
            or 0x0370 <= cp <= 0x03FF  # Greek
            or 0x1F00 <= cp <= 0x1FFF  # Greek Extended
        ):
            non_latin += 1
            if ch in _CONFUSABLE_CHARS:
                confusable += 1
        elif (
            0x41 <= cp <= 0x5A  # A-Z
            or 0x61 <= cp <= 0x7A  # a-z
            or 0xC0 <= cp <= 0x24F  # Latin-1 Supplement + Extended
            or cp in (0x0131, 0x0261)  # U+0131 / U+0261 Latin lookalikes
        ):
            latin += 1

    if non_latin == 0 or latin == 0:
        # Nothing to fold, or a native non-Latin document (nothing to anchor
        # the decision on). Leave it untouched.
        return text

    if confusable * 10 >= non_latin * 9:
        # ≥90% of the non-Latin letters are Latin confusables: this is a
        # mutated Latin document, not native text.
        return text.translate(_CONFUSABLE_TABLE)

    def _fold_token(match: re.Match[str]) -> str:
        token = match.group(0)
        has_latin = has_non_latin = False
        for ch in token:
            cp = ord(ch)
            if 0x0400 <= cp <= 0x052F or 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF:
                has_non_latin = True
            elif ch.isalpha():
                has_latin = True
            if has_latin and has_non_latin:
                return token.translate(_CONFUSABLE_TABLE)
        return token

    return _LETTER_RUN.sub(_fold_token, text)


def _strip_marks_after_ascii(text: str) -> str:
    """Remove combining marks (Mn) that directly follow an ASCII letter.

    Detection-copy normalization only (NG-5): restores keyword integrity
    under diacritic/combining-mark mutations without touching load-bearing
    marks on non-ASCII bases (Devanagari, Arabic, Hangul, etc.).
    """
    out: list[str] = []
    prev_ascii_letter = False
    for ch in text:
        if unicodedata.category(ch) == "Mn" and prev_ascii_letter:
            continue  # drop the mark; the ASCII base keeps prev_ascii_letter True
        out.append(ch)
        prev_ascii_letter = ("a" <= ch <= "z") or ("A" <= ch <= "Z")
    return "".join(out)


# ── Encoding evasion patterns ────────────────────────────────────────────

# Cap the length of a single base64 match we will attempt to decode. Longer
# matches are flagged but NOT decoded, preventing memory-exhaustion via a
# multi-megabyte base64 blob. 8 KiB of base64 decodes to ~6 KiB of bytes.
_BASE64_DECODE_CAP = 8 * 1024

BASE64_PATTERN = re.compile(
    r"(?:[A-Za-z0-9+/]{40,}={0,2})",
    re.ASCII,
)

# Hard cap on decompressed bytes during the zlib bomb check. We decompress
# incrementally and abort the moment we exceed this, so a crafted bomb cannot
# materialize hundreds of MB before the ratio check fires.
_MAX_DECOMPRESSED_BYTES = 8 * 1024 * 1024  # 8 MiB


def _bounded_decompress(
    raw: bytes,
    max_bytes: int = _MAX_DECOMPRESSED_BYTES,
    ratio_limit: float = 10.0,
) -> tuple[bool, float, int]:
    """Decompress `raw` incrementally with a hard byte cap.

    Returns (exceeded_cap, ratio, produced_bytes). Never materializes more
    than max_bytes+chunk of decompressed data in memory, so a crafted zlib
    bomb cannot OOM the worker.
    """
    decompressor = zlib.decompressobj(wbits=0)
    produced = 0
    exceeded_cap = False
    view = memoryview(raw)
    chunk_size = 4096
    try:
        for i in range(0, len(view), chunk_size):
            piece = bytes(view[i : i + chunk_size])
            out = decompressor.decompress(piece, max_bytes + 1 - produced)
            produced += len(out)
            if produced > max_bytes:
                exceeded_cap = True
                break
        try:
            tail = decompressor.flush()
        except zlib.error:
            tail = b""
        produced += len(tail)
    except zlib.error:
        # Not a valid zlib/deflate stream - treat as non-compressed (safe).
        return False, 0.0, 0
    ratio = produced / max(len(raw), 1)
    return exceeded_cap, ratio, produced


HEX_ENCODED_PATTERN = re.compile(
    r"(?:\\x[0-9a-fA-F]{2}){4,}",
)

ROT13_COMMON = re.compile(
    r"\b(?:vang|cynvagrkg|chfurf|frrzn|qrpelcgvat|npphss|pbzcyrgr|pbafvqre)\b",
    re.IGNORECASE,
)

# ── Structural anomaly patterns ───────────────────────────────────────────

REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{20,}")
ROLE_INJECTION_PATTERN = re.compile(
    r"\b(?:system|assistant|user|tool)\s*:\s*(?:ignore|forget|override|disregard)",
    re.IGNORECASE,
)
MARKDOWN_INJECTION_PATTERN = re.compile(
    r"(?:```|~~~)\s*\w*\s*\n.*?(?:```|~~~)",
    re.DOTALL,
)


class StructuralScanner(BaseScanner["ScannerSettings"]):
    """Layer 1: Structural validation and sanitization."""

    layer = ScanLayer.STRUCTURAL

    def scan(
        self, request: EvaluateRequest, context: dict[str, Any] | None = None
    ) -> ScannerResult:
        start = time.perf_counter()
        findings: list[Finding] = []
        sanitized_parts: list[str] = []

        # Get the input text (F6: user-role turns only unless scan_all_roles)
        texts = request.input_texts()
        if not texts:
            if request.messages:
                # Messages exist but no user turns (e.g. a system-only or
                # assistant-only payload in proxy mode): nothing to scan.
                return self._result(Verdict.ALLOW, [], start)
            return self._result(  # pragma: no cover — validation rejects this at schema level
                Verdict.BLOCK,
                [
                    Finding(
                        category=ThreatCategory.SELF_ATTACK,
                        severity=Severity.HIGH,
                        verdict=Verdict.BLOCK,
                        confidence=1.0,
                        layer=self.layer,
                        rule_id="STRUCT-001",
                        description="Empty request: no messages or prompt provided",
                    )
                ],
                start,
            )

        for text in texts:
            result_text, text_findings = self._validate_and_sanitize(text)
            findings.extend(text_findings)
            sanitized_parts.append(result_text)

        # Determine verdict from findings
        verdict = self._findings_to_verdict(findings)
        sanitized = "\n".join(sanitized_parts) if len(sanitized_parts) > 1 else sanitized_parts[0]

        return self._result(verdict, findings, start, sanitized=sanitized)

    def _validate_and_sanitize(self, text: str) -> tuple[str, list[Finding]]:
        """Validate and sanitize a single text input."""
        findings: list[Finding] = []

        # 1. Length check
        if len(text) > self.settings.max_input_length:
            findings.append(
                Finding(
                    category=ThreatCategory.DOS_ABUSE,
                    severity=Severity.HIGH,
                    verdict=Verdict.BLOCK,
                    confidence=0.95,
                    layer=self.layer,
                    rule_id="STRUCT-002",
                    description=f"Input exceeds max length: {len(text)} > {self.settings.max_input_length}",
                    mitigation="Truncate or reject oversized input",
                )
            )

        # 2. Decompression ratio check (bomb defense) — BOUNDED.
        # Decompress incrementally with a hard byte cap so a crafted zlib bomb
        # cannot materialize in memory before the ratio is checked.
        raw = text.encode("utf-8")
        try:
            exceeded_cap, ratio, produced = _bounded_decompress(
                raw,
                max_bytes=_MAX_DECOMPRESSED_BYTES,
                ratio_limit=self.settings.max_decompression_ratio,
            )
            if (produced > 0 or exceeded_cap) and (
                exceeded_cap or ratio > self.settings.max_decompression_ratio
            ):
                findings.append(
                    Finding(
                        category=ThreatCategory.DOS_ABUSE,
                        severity=Severity.CRITICAL,
                        verdict=Verdict.BLOCK,
                        confidence=0.99,
                        layer=self.layer,
                        rule_id="STRUCT-003",
                        description=(
                            f"Decompression bomb: exceeded {_MAX_DECOMPRESSED_BYTES} byte cap"
                            if exceeded_cap
                            else f"Decompression bomb: ratio {ratio:.1f}:1 exceeds limit {self.settings.max_decompression_ratio}:1"
                        ),
                        mitigation="Reject compressed input with excessive ratio",
                    )
                )
        except zlib.error:
            pass  # Not compressed, which is fine

        # 3. NFKD normalization
        normalized = unicodedata.normalize("NFKD", text)

        # 3b. Cross-script homoglyph fold (NG-13 mutation-gate finding):
        # Latin→Cyrillic/Greek lookalikes break literal keyword rules at
        # 96.3% Unsafe-ASR. Context-aware (confusable-ratio + mixed-script
        # token gates), so genuine Russian/Greek prompts are left untouched.
        # Runs BEFORE the mark fold: in the accent+homoglyph combo attack the
        # combining mark sits on a Cyrillic base, and folding the base to
        # Latin first is what lets the mark fold see it. Detection copy only.
        normalized = _fold_cross_script_confusables(normalized)

        # 3c. Latin-script combining-mark fold (NG-5 mutation-gate finding):
        # NFKD("é") = "e" + U+0301, and a stray combining mark between ASCII
        # letters breaks every literal keyword regex ("i\u0301gnore" no longer
        # matches (?i)\bignore\b) — the diacritics mutation sailed through at
        # 96.3% ASR on the NG-5 baseline run. Fold marks that directly follow
        # an ASCII letter ONLY (precomposed Latin diacritics become their base
        # letter). Marks after NON-ASCII bases (Devanagari viramas, Arabic
        # harakat, Hangul jamo) are linguistically load-bearing and are kept —
        # a blanket Mn-category strip would destroy legitimate non-Latin text.
        # Detection-copy normalization only: the delivered payload is untouched.
        if any(unicodedata.category(ch) == "Mn" for ch in normalized):
            normalized = _strip_marks_after_ascii(normalized)

        # 4. Zero-width character detection and removal
        zw_matches = ZW_PATTERN.findall(normalized)
        if zw_matches:
            zw_count = sum(len(m) for m in zw_matches)
            findings.append(
                Finding(
                    category=ThreatCategory.ENCODING_EVASION,
                    severity=Severity.MEDIUM,
                    verdict=Verdict.SANITIZE,
                    confidence=0.9,
                    layer=self.layer,
                    rule_id="STRUCT-004",
                    description=f"Zero-width characters detected: {zw_count} characters removed",
                    mitigation="Strip zero-width characters before processing",
                )
            )
            normalized = ZW_PATTERN.sub("", normalized)

        # 5. Encoding evasion detection
        # Base64
        b64_matches = BASE64_PATTERN.findall(normalized)
        if b64_matches:
            for match in b64_matches[:3]:  # Limit to first 3
                # Cap match length before decoding to avoid memory DoS.
                if len(match) > _BASE64_DECODE_CAP:
                    findings.append(
                        Finding(
                            category=ThreatCategory.ENCODING_EVASION,
                            severity=Severity.HIGH,
                            verdict=Verdict.BLOCK,
                            confidence=0.9,
                            layer=self.layer,
                            rule_id="STRUCT-005",
                            description=f"Oversized base64 blob ({len(match)} chars) — possible payload smuggling",
                            mitigation="Block oversized base64 payloads",
                        )
                    )
                    continue
                try:
                    import base64

                    decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
                    # Check if decoded content contains injection patterns
                    decoded_lower = decoded.lower()
                    injection_keywords = ["ignore", "system", "instruction", "prompt", "override"]
                    if any(kw in decoded_lower for kw in injection_keywords):
                        findings.append(
                            Finding(
                                category=ThreatCategory.ENCODING_EVASION,
                                severity=Severity.HIGH,
                                verdict=Verdict.BLOCK,
                                confidence=0.85,
                                layer=self.layer,
                                rule_id="STRUCT-005",
                                description="Base64-encoded injection payload detected",
                                evidence=f"Decoded contains: {[kw for kw in injection_keywords if kw in decoded_lower]}",
                                mitigation="Block base64-encoded injection payloads",
                            )
                        )
                except Exception:  # pragma: no cover — best-effort base64 decode
                    pass

        # Hex-encoded strings
        hex_matches = HEX_ENCODED_PATTERN.findall(normalized)
        if hex_matches:
            findings.append(
                Finding(
                    category=ThreatCategory.ENCODING_EVASION,
                    severity=Severity.MEDIUM,
                    verdict=Verdict.SANITIZE,
                    confidence=0.75,
                    layer=self.layer,
                    rule_id="STRUCT-006",
                    description=f"Hex-encoded strings detected: {len(hex_matches)} sequences",
                    mitigation="Decode and re-check hex sequences through pattern scanner",
                )
            )

        # ROT13 common words
        rot13_matches = ROT13_COMMON.findall(normalized)
        if rot13_matches:
            findings.append(
                Finding(
                    category=ThreatCategory.ENCODING_EVASION,
                    severity=Severity.MEDIUM,
                    verdict=Verdict.SANITIZE,
                    confidence=0.7,
                    layer=self.layer,
                    rule_id="STRUCT-007",
                    description=f"ROT13-encoded injection keywords detected: {rot13_matches}",
                    mitigation="Decode ROT13 and re-scan through pattern layer",
                )
            )

        # 6. Structural anomaly detection
        # Repeated characters (bomb-like)
        repeated = REPEATED_CHAR_PATTERN.findall(normalized)
        if repeated:
            findings.append(
                Finding(
                    category=ThreatCategory.DOS_ABUSE,
                    severity=Severity.LOW,
                    verdict=Verdict.SANITIZE,
                    confidence=0.6,
                    layer=self.layer,
                    rule_id="STRUCT-008",
                    description=f"Excessive character repetition detected: {len(repeated)} instances",
                    mitigation="Normalize repeated characters",
                )
            )

        # Role injection in text
        role_injection = ROLE_INJECTION_PATTERN.findall(normalized)
        if role_injection:
            findings.append(
                Finding(
                    category=ThreatCategory.PROMPT_INJECTION_DIRECT,
                    severity=Severity.MEDIUM,
                    verdict=Verdict.BLOCK,
                    confidence=0.7,
                    layer=self.layer,
                    rule_id="STRUCT-009",
                    description="Structural role injection pattern detected",
                    mitigation="Sanitize role markers before LLM processing",
                )
            )

        return normalized, findings

    def _findings_to_verdict(self, findings: list[Finding]) -> Verdict:
        """Convert findings to verdict — strictest wins."""
        if not findings:
            return Verdict.ALLOW

        priority = {Verdict.BLOCK: 6, Verdict.SANITIZE: 5, Verdict.ESCALATE: 4}
        highest = Verdict.ALLOW
        highest_priority = 0

        for f in findings:
            p = priority.get(f.verdict, 0)
            if p > highest_priority:
                highest_priority = p
                highest = f.verdict

        return highest
