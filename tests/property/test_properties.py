"""Property-based tests (hypothesis) for NeuralGuard security invariants.

These verify structural properties that must hold for ANY input, not just
hand-picked examples — the kind of guarantee a firewall needs.
"""

from __future__ import annotations

import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from neuralguard.config.settings import ScannerSettings
from neuralguard.models.schemas import (
    EvaluateRequest,
    ScannerResult,
    Verdict,
)
from neuralguard.scanners.pipeline import _VERDICT_PRIORITY, ScannerPipeline
from neuralguard.scanners.structural import (
    ZERO_WIDTH_CHARS,
    ZW_PATTERN,
    StructuralScanner,
)

# A bounded text strategy: printable ASCII + a sprinkle of unicode + zero-width
# chars, kept small so tests stay fast.
text_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        max_codepoint=0x9FFF,
    ),
    max_size=200,
)

nonempty_text_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        max_codepoint=0x9FFF,
    ),
    min_size=1,
    max_size=200,
)

zw_text_strategy = st.builds(
    lambda base, zw: base + zw + base,
    text_strategy,
    st.sampled_from(sorted(ZERO_WIDTH_CHARS)),
)


def _scanner() -> StructuralScanner:
    return StructuralScanner(ScannerSettings())


class TestStructuralProperties:
    @given(text=text_strategy)
    @settings(max_examples=75, deadline=2000)
    def test_normalization_is_idempotent(self, text):
        # NFKD normalize twice == normalize once (the scanner applies NFKD).
        once = unicodedata.normalize("NFKD", text)
        twice = unicodedata.normalize("NFKD", once)
        assert once == twice

    @given(text=zw_text_strategy)
    @settings(max_examples=75, deadline=2000)
    def test_zero_width_chars_stripped_from_sanitized(self, text):
        scanner = _scanner()
        result = scanner.safe_scan(EvaluateRequest(prompt=text, tenant_id="t"))
        if result.sanitized_output is not None:
            assert not ZW_PATTERN.search(result.sanitized_output), (
                "Zero-width characters survived sanitization"
            )

    @given(text=nonempty_text_strategy)
    @settings(max_examples=75, deadline=3000)
    def test_scanner_never_raises_and_returns_valid_result(self, text):
        # The firewall contract: scan() must never raise for valid input; it
        # must return a valid ScannerResult even on pathological input.
        scanner = _scanner()
        result = scanner.safe_scan(EvaluateRequest(prompt=text, tenant_id="t"))
        assert isinstance(result, ScannerResult)
        assert result.verdict in tuple(Verdict)
        assert result.latency_ms >= 0.0

    @given(text=text_strategy)
    @settings(max_examples=50, deadline=3000)
    def test_bounded_decompress_safe_for_any_input(self, text):
        # Any input fed through the bounded decompressor must return within
        # bounded time and never raise (zlib errors are swallowed).
        from neuralguard.scanners.structural import _bounded_decompress

        exceeded, _ratio, produced = _bounded_decompress(text.encode("utf-8"))
        assert isinstance(exceeded, bool)
        assert isinstance(produced, int)
        assert produced >= 0
        # If it claimed to decompress, produced must be <= cap+chunk when exceeded.
        if exceeded:
            assert produced <= 8 * 1024 * 1024 + 4096


class TestPipelineArbitration:
    def test_strictest_verdict_wins_regardless_of_order(self):
        # Property: the arbitration verdict equals the max-priority verdict
        # across all scanner results, independent of insertion order.
        from neuralguard.models.schemas import ScanLayer

        def _results(verdicts):
            return [
                ScannerResult(
                    layer=layer,
                    verdict=v,
                    findings=[],
                    latency_ms=0.1,
                )
                for layer, v in zip(
                    [ScanLayer.STRUCTURAL, ScanLayer.PATTERN, ScanLayer.SEMANTIC],
                    verdicts,
                    strict=False,
                )
                if v is not None
            ]

        config = ScannerSettings()
        # Use a pipeline with no registered scanners; _arbitrate is pure.
        pipeline = ScannerPipeline.__new__(ScannerPipeline)  # bypass __init__
        pipeline.config = type(
            "C", (), {"scanner": config, "action": type("A", (), {"fail_closed": True})()}
        )()

        cases = [
            ([Verdict.ALLOW, Verdict.SANITIZE, Verdict.BLOCK], Verdict.BLOCK),
            ([Verdict.SANITIZE, Verdict.ALLOW, Verdict.ALLOW], Verdict.SANITIZE),
            ([Verdict.ALLOW, Verdict.ALLOW, Verdict.ALLOW], Verdict.ALLOW),
            ([Verdict.ESCALATE, Verdict.BLOCK, Verdict.ALLOW], Verdict.BLOCK),
        ]
        for verdicts, expected in cases:
            results = _results(verdicts)
            verdict, _ = pipeline._arbitrate(results)
            assert verdict == expected, f"{verdicts} -> {verdict}, expected {expected}"

    def test_priority_ordering_is_total(self):
        # BLOCK > SANITIZE > ESCALATE > QUARANTINE > RATE_LIMIT > ALLOW
        priorities = [
            (_VERDICT_PRIORITY[v], v)
            for v in (
                Verdict.ALLOW,
                Verdict.RATE_LIMIT,
                Verdict.QUARANTINE,
                Verdict.ESCALATE,
                Verdict.SANITIZE,
                Verdict.BLOCK,
            )
        ]
        values = [p for p, _ in priorities]
        assert values == sorted(values), "Verdict priority is not monotonic"
        assert len(set(values)) == len(values), "Verdict priorities not unique"
