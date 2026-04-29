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

import structlog as _structlog

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

# ── Encoding evasion patterns ────────────────────────────────────────────

BASE64_PATTERN = re.compile(
    r"(?:[A-Za-z0-9+/]{40,}={0,2})",
    re.ASCII,
)

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


class StructuralScanner(BaseScanner):
    """Layer 1: Structural validation and sanitization."""

    layer = ScanLayer.STRUCTURAL

    def scan(self, request: EvaluateRequest, context: dict | None = None) -> ScannerResult:
        start = time.perf_counter()
        findings: list[Finding] = []
        sanitized_parts: list[str] = []

        # Get the input text
        if request.messages:
            texts = [m.content for m in request.messages]
        elif request.prompt:
            texts = [request.prompt]
        else:
            return self._result(
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

        # 2. Decompression ratio check (bomb defense)
        try:
            decompressed = zlib.decompress(text.encode("utf-8"), wbits=0)
            ratio = len(decompressed) / max(len(text.encode("utf-8")), 1)
            if ratio > self.settings.max_decompression_ratio:
                findings.append(
                    Finding(
                        category=ThreatCategory.DOS_ABUSE,
                        severity=Severity.CRITICAL,
                        verdict=Verdict.BLOCK,
                        confidence=0.99,
                        layer=self.layer,
                        rule_id="STRUCT-003",
                        description=f"Decompression bomb: ratio {ratio:.1f}:1 exceeds limit {self.settings.max_decompression_ratio}:1",
                        mitigation="Reject compressed input with excessive ratio",
                    )
                )
        except zlib.error:
            pass  # Not compressed, which is fine

        # 3. NFKD normalization
        normalized = unicodedata.normalize("NFKD", text)

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
                except Exception:
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
