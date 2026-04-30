"""Unit tests for data models."""

import pytest
from pydantic import ValidationError

from neuralguard.models.schemas import (
    EvaluateRequest,
    Finding,
    LayerArbitrationResult,
    Message,
    ScanLayer,
    ScannerResult,
    Severity,
    ThreatCategory,
    Verdict,
)


class TestMessage:
    """Tests for Message model."""

    def test_valid_message(self):
        m = Message(role="user", content="Hello")
        assert m.role == "user"
        assert m.content == "Hello"

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="user", content="   ")

    def test_all_roles(self):
        for role in ["system", "user", "assistant", "tool"]:
            m = Message(role=role, content="test")
            assert m.role == role

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            Message(role="admin", content="test")


class TestEvaluateRequest:
    """Tests for EvaluateRequest model."""

    def test_prompt_mode(self):
        r = EvaluateRequest(prompt="Hello")
        assert r.prompt == "Hello"
        assert r.messages is None
        assert r.tenant_id == "default"
        assert r.use_case == "chat"

    def test_messages_mode(self):
        r = EvaluateRequest(
            messages=[
                Message(role="system", content="You are helpful"),
                Message(role="user", content="Hello"),
            ]
        )
        assert len(r.messages) == 2

    def test_tenant_id_normalized(self):
        r = EvaluateRequest(prompt="test", tenant_id="  MyTenant  ")
        assert r.tenant_id == "mytenant"

    def test_tenant_id_empty_rejected(self):
        with pytest.raises(ValidationError):
            EvaluateRequest(prompt="test", tenant_id="")

    def test_tenant_id_too_long(self):
        with pytest.raises(ValidationError):
            EvaluateRequest(prompt="test", tenant_id="x" * 65)

    def test_scanner_override(self):
        r = EvaluateRequest(
            prompt="test",
            scanners=[ScanLayer.STRUCTURAL, ScanLayer.PATTERN],
        )
        assert len(r.scanners) == 2

    def test_use_case_values(self):
        for uc in ["chat", "agent", "rag", "tool", "completion"]:
            r = EvaluateRequest(prompt="test", use_case=uc)
            assert r.use_case == uc

    def test_empty_request_rejected(self):
        """Empty requests (no prompt, no messages) must be rejected at validation."""
        with pytest.raises(ValidationError, match="At least one"):
            EvaluateRequest(prompt=None, messages=None)

    def test_prompt_with_empty_messages_allowed(self):
        """Messages array can be empty — validation only requires prompt OR messages."""
        r = EvaluateRequest(messages=[], prompt=None)
        # Note: empty messages list passes validation but structural scanner
        # will handle it. This test ensures the model_validator allows it.
        assert r.messages == []


class TestFinding:
    """Tests for Finding model."""

    def test_minimal_finding(self):
        f = Finding(
            category=ThreatCategory.PROMPT_INJECTION_DIRECT,
            severity=Severity.HIGH,
            verdict=Verdict.BLOCK,
            confidence=0.95,
            layer=ScanLayer.PATTERN,
            rule_id="PI-D-001",
            description="Direct prompt injection detected",
        )
        assert f.category == ThreatCategory.PROMPT_INJECTION_DIRECT
        assert f.confidence == 0.95

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Finding(
                category=ThreatCategory.PROMPT_INJECTION_DIRECT,
                severity=Severity.HIGH,
                verdict=Verdict.BLOCK,
                confidence=1.5,  # Out of bounds
                layer=ScanLayer.PATTERN,
                rule_id="PI-D-001",
                description="test",
            )

    def test_all_threat_categories(self):
        """Verify all SRD threat categories are defined."""
        expected = [
            "T-PI-D",
            "T-PI-I",
            "T-JB",
            "T-EXT",
            "T-EXF",
            "T-TOOL",
            "T-AGT",
            "T-ENC",
            "T-DOS",
            "T-OUT",
            "T-MEM",
            "T-CASC",
            "T-NG",
        ]
        actual = [c.value for c in ThreatCategory]
        assert sorted(actual) == sorted(expected)


class TestScannerResult:
    """Tests for ScannerResult model."""

    def test_basic_result(self):
        r = ScannerResult(
            layer=ScanLayer.STRUCTURAL,
            verdict=Verdict.ALLOW,
            findings=[],
            latency_ms=1.5,
        )
        assert r.verdict == Verdict.ALLOW
        assert r.latency_ms == 1.5

    def test_result_with_error(self):
        r = ScannerResult(
            layer=ScanLayer.SEMANTIC,
            verdict=Verdict.BLOCK,
            findings=[],
            latency_ms=50.0,
            error="Model loading failed",
        )
        assert r.error == "Model loading failed"


class TestLayerArbitration:
    """Tests for Layer Arbitration result."""

    def test_arbitration_result(self):
        result = LayerArbitrationResult(
            verdict=Verdict.BLOCK,
            findings=[],
            scanner_results=[],
            total_latency_ms=15.0,
            arbitration_reason="Strictest verdict: block from pattern layer",
        )
        assert result.verdict == Verdict.BLOCK
