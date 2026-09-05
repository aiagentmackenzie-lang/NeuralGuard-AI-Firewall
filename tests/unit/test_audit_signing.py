"""P2-10: Ed25519 audit-event signing.

Closes the documented chain gap: a file-write attacker could previously
forge an internally-consistent new chain. Signing binds every event's chain
hash to the holder of the private seed — forging now requires the key.

Covers: keygen roundtrip, tamper rejection, AuditLogger stamping, the
verify_audit_files pubkey mode (missing/foreign signatures = BROKEN), the
settings validator, and both CLI commands (exit codes included).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from neuralguard.config.settings import AuditSettings
from neuralguard.logging.audit import AuditLogger
from neuralguard.logging.signing import (
    SigningKeyError,
    generate_signing_keypair,
    public_key_from_seed,
    sign_event_hash,
    verify_event_signature,
)
from neuralguard.logging.verify import verify_audit_files
from neuralguard.models.schemas import (
    AuditEvent,
    EvaluateRequest,
    EvaluateResponse,
    LayerArbitrationResult,
    Verdict,
)


@pytest.fixture(scope="module")
def keypair() -> tuple[str, str]:
    return generate_signing_keypair()


class TestSigningPrimitives:
    def test_sign_verify_roundtrip(self, keypair: tuple[str, str]) -> None:
        seed, pubkey = keypair
        event_hash = "ab" * 32
        sig = sign_event_hash(event_hash, seed)
        assert verify_event_signature(event_hash, sig, pubkey)

    def test_tampered_hash_fails_verification(self, keypair: tuple[str, str]) -> None:
        seed, pubkey = keypair
        sig = sign_event_hash("ab" * 32, seed)
        assert not verify_event_signature("cd" * 32, sig, pubkey)

    def test_foreign_pubkey_fails(self, keypair: tuple[str, str]) -> None:
        seed, _ = keypair
        _, other_pub = generate_signing_keypair()
        sig = sign_event_hash("ab" * 32, seed)
        assert not verify_event_signature("ab" * 32, sig, other_pub)

    def test_garbage_inputs_fail_closed(self, keypair: tuple[str, str]) -> None:
        _, pubkey = keypair
        assert not verify_event_signature("nothex", "nothex", pubkey)
        assert not verify_event_signature("ab" * 32, "nothex", pubkey)
        assert not verify_event_signature("ab" * 32, "ab" * 32, "ee" * 31)

    def test_bad_seed_materials_raise(self) -> None:
        with pytest.raises(SigningKeyError, match="not valid hex"):
            sign_event_hash("ab" * 32, "zz" * 32)
        with pytest.raises(SigningKeyError, match="32 bytes"):
            sign_event_hash("ab" * 32, "ab" * 16)

    def test_public_key_derivation_matches_keypair(self) -> None:
        seed, pubkey = generate_signing_keypair()
        assert public_key_from_seed(seed) == pubkey


def _log_one(settings: AuditSettings) -> AuditEvent:
    logger = AuditLogger(settings)
    request = EvaluateRequest(prompt="hello", tenant_id="default")
    response = EvaluateResponse(
        request_id="r1",
        tenant_id="default",
        verdict=Verdict.ALLOW,
        findings=[],
        confidence=0.0,
        total_latency_ms=1.0,
        scan_layers_used=[],
    )
    arbitration = LayerArbitrationResult(
        verdict=Verdict.ALLOW,
        findings=[],
        scanner_results=[],
        total_latency_ms=1.0,
        arbitration_reason="test",
    )
    return logger.log_evaluation(request, response, arbitration)


class TestAuditLoggerSigning:
    def test_unsigned_by_default(self, tmp_path: Any) -> None:
        event = _log_one(AuditSettings(jsonl_path=tmp_path / "a"))
        assert event.event_sig is None

    def test_signed_when_configured(self, tmp_path: Any, keypair: tuple[str, str]) -> None:
        seed, pubkey = keypair
        event = _log_one(AuditSettings(jsonl_path=tmp_path / "b", signing_key=seed))
        assert event.event_sig is not None
        assert event.event_hash is not None
        assert verify_event_signature(event.event_hash, event.event_sig, pubkey)

    def test_invalid_signing_key_refused_at_config(self, tmp_path: Any) -> None:
        with pytest.raises(Exception, match="signing_key invalid"):
            AuditSettings(jsonl_path=tmp_path / "c", signing_key="zz" * 32)
        with pytest.raises(Exception, match="signing_key invalid"):
            AuditSettings(jsonl_path=tmp_path / "c", signing_key="ab" * 16)


def _build_audit_dir(
    tmp_path: Any, keypair: tuple[str, str], sign: bool, tamper: bool = False
) -> Any:
    seed, _ = keypair
    settings = (
        AuditSettings(jsonl_path=tmp_path / "audit", signing_key=seed)
        if sign
        else AuditSettings(jsonl_path=tmp_path / "audit")
    )
    logger = AuditLogger(settings)
    events = []
    for i in range(3):
        request = EvaluateRequest(prompt=f"hello {i}", tenant_id="default")
        response = EvaluateResponse(
            request_id=f"r{i}",
            tenant_id="default",
            verdict=Verdict.ALLOW,
            findings=[],
            confidence=0.0,
            total_latency_ms=1.0,
            scan_layers_used=[],
        )
        arbitration = LayerArbitrationResult(
            verdict=Verdict.ALLOW,
            findings=[],
            scanner_results=[],
            total_latency_ms=1.0,
            arbitration_reason="test",
        )
        events.append(logger.log_evaluation(request, response, arbitration))
    if tamper and events:
        events[1].verdict = Verdict.BLOCK  # flip a recorded verdict post-hoc
    out = tmp_path / "copy"
    out.mkdir()
    with (out / "audit-2026-09-05.jsonl").open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(e.model_dump_json() + "\n")
    return out


class TestVerifyWithPubkey:
    def test_signed_chain_valid_with_pubkey(self, tmp_path: Any, keypair: tuple[str, str]) -> None:
        _seed, pubkey = keypair
        audit_dir = _build_audit_dir(tmp_path, keypair, sign=True)
        report = verify_audit_files(audit_dir, pubkey_hex=pubkey)
        assert report.all_valid

    def test_unsigned_chain_broken_with_pubkey(
        self, tmp_path: Any, keypair: tuple[str, str]
    ) -> None:
        _, pubkey = keypair
        audit_dir = _build_audit_dir(tmp_path, keypair, sign=False)
        report = verify_audit_files(audit_dir, pubkey_hex=pubkey)
        assert not report.all_valid  # hash-consistent but unsigned = BROKEN
        # Same chain still verifies WITHOUT the pubkey (hash-only mode).
        assert verify_audit_files(audit_dir).all_valid

    def test_forged_chain_detected(self, tmp_path: Any, keypair: tuple[str, str]) -> None:
        """THE threat signing exists for: internally-consistent forged file."""
        _seed, pubkey = keypair
        audit_dir = _build_audit_dir(tmp_path, keypair, sign=True, tamper=True)
        report = verify_audit_files(audit_dir, pubkey_hex=pubkey)
        assert not report.all_valid

    def test_foreign_key_broken(self, tmp_path: Any, keypair: tuple[str, str]) -> None:
        _, other_pub = generate_signing_keypair()
        audit_dir = _build_audit_dir(tmp_path, keypair, sign=True)
        report = verify_audit_files(audit_dir, pubkey_hex=other_pub)
        assert not report.all_valid


class TestCli:
    def test_keygen_outputs_usable_material(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "neuralguard.cli", "audit-keygen"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        seed_line = next(
            line for line in result.stdout.splitlines() if line.startswith("signing_key")
        )
        pubkey_line = next(
            line for line in result.stdout.splitlines() if line.startswith("public_key")
        )
        seed = seed_line.split(": ")[1].strip()
        pubkey = pubkey_line.split(": ")[1].strip()
        # The printed pair is functional.
        sig = sign_event_hash("ab" * 32, seed)
        assert verify_event_signature("ab" * 32, sig, pubkey)

    def test_audit_verify_pubkey_flag_end_to_end(
        self, tmp_path: Any, keypair: tuple[str, str]
    ) -> None:
        _seed, pubkey = keypair
        audit_dir = _build_audit_dir(tmp_path, keypair, sign=True)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "neuralguard.cli",
                "audit-verify",
                str(audit_dir),
                "--pubkey",
                pubkey,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ALL CHAINS VALID" in result.stdout
        # And a bogus pubkey fails with exit 1.
        bad = subprocess.run(
            [
                sys.executable,
                "-m",
                "neuralguard.cli",
                "audit-verify",
                str(audit_dir),
                "--pubkey",
                "ee" * 32,
            ],
            capture_output=True,
            text=True,
        )
        assert bad.returncode == 1
