"""F14: the operator audit-verify tool (per-worker chain verification).

A naive single-chain verify over an interleaved multi-worker JSONL file
fails BY DESIGN (chains are per-worker). The tool groups by worker_id and
verifies each chain: valid files -> all VALID + exit 0; a tampered line ->
BROKEN + exit 1; a corrupt line -> parse error + exit 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuralguard.config.settings import AuditSettings
from neuralguard.logging.audit import AuditLogger
from neuralguard.logging.verify import verify_audit_files
from neuralguard.models.schemas import EvaluateRequest, EvaluateResponse, Verdict


def _logger(tmp_path: Path, worker_id: str) -> AuditLogger:
    settings = AuditSettings(backend="jsonl", jsonl_path=str(tmp_path), retention_days=30)
    logger_ = AuditLogger(settings)
    logger_._worker_id = worker_id  # deterministic chain per worker
    return logger_


def _log_one(logger_: AuditLogger, prompt: str) -> None:
    req = EvaluateRequest(prompt=prompt)
    resp = EvaluateResponse(
        tenant_id="default",
        verdict=Verdict.ALLOW,
        findings=[],
        confidence=0.0,
        scan_layers_used=[],
        total_latency_ms=1.0,
    )
    from neuralguard.models.schemas import LayerArbitrationResult

    logger_.log_evaluation(
        req,
        resp,
        LayerArbitrationResult(
            verdict=Verdict.ALLOW,
            findings=[],
            scanner_results=[],
            total_latency_ms=1.0,
            arbitration_reason="test",
        ),
    )


class TestAuditVerify:
    def test_multi_worker_interleaved_file_all_valid(self, tmp_path: Path) -> None:
        """Two workers interleaving into ONE file: the naive single-chain
        verify fails by design; the tool scopes per worker -> all VALID."""
        w1 = _logger(tmp_path, "worker-one")
        w2 = _logger(tmp_path, "worker-two")
        # Interleave writes from two workers into the same directory.
        for i in range(4):
            _log_one(w1, f"worker one turn {i}")
            _log_one(w2, f"worker two turn {i}")

        report = verify_audit_files(tmp_path)
        assert report.all_valid
        assert report.parse_errors == 0
        assert (
            {c.worker_id for c in report.chains} == {"worker-one-id", "worker-two-id"}
            or {c.worker_id for c in report.chains} == {"worker-one", "worker-two"}
            or len(report.chains) == 2
        )
        for chain in report.chains:
            assert chain.valid
            assert chain.event_count == 4

    def test_naive_single_chain_verify_fails_by_design(self, tmp_path: Path) -> None:
        """Documenting the trap the tool exists for: one chain over an
        interleaved two-worker file is BROKEN (worker A's events are not the
        parents of worker B's). The per-worker grouping fixes it."""
        from neuralguard.logging.chain import verify_chain
        from neuralguard.models.schemas import AuditEvent

        w1 = _logger(tmp_path, "aaa")
        w2 = _logger(tmp_path, "bbb")
        events: list[AuditEvent] = []
        for i in range(3):
            _log_one(w1, f"a{i}")
            _log_one(w2, f"b{i}")
        for line in (tmp_path / next(iter(tmp_path.iterdir())).name).read_text().splitlines():
            if line.strip():
                events.append(AuditEvent.model_validate_json(line))
        # Sanity: the interleaved file parsed.
        assert len(events) == 6
        assert not verify_chain(events), "naive single-chain verify must fail on interleaved files"

    def test_tampered_event_detected(self, tmp_path: Path) -> None:
        w1 = _logger(tmp_path, "victim")
        _log_one(w1, "benign prompt one")
        _log_one(w1, "benign prompt two")

        audit_file = next(tmp_path.rglob("*.jsonl"))
        lines = audit_file.read_text().splitlines()
        event = json.loads(lines[0])
        event["verdict"] = "block"  # tamper: verdict rewritten after the fact
        lines[0] = json.dumps(event)
        audit_file.write_text("\n".join(lines) + "\n")

        report = verify_audit_files(tmp_path)
        assert not report.all_valid
        assert report.chains[0].valid is False

    def test_parse_error_counts_and_fails(self, tmp_path: Path) -> None:
        w1 = _logger(tmp_path, "w")
        _log_one(w1, "prompt")
        audit_file = next(tmp_path.rglob("*.jsonl"))
        audit_file.write_text(audit_file.read_text() + "{corrupt json\n")

        report = verify_audit_files(tmp_path)
        assert report.parse_errors == 1
        assert not report.all_valid

    def test_exit_codes(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """CLI contract: 0 all-valid, 1 broken, 2 unreadable."""
        import subprocess
        import sys

        w1 = _logger(tmp_path, "ok-worker")
        _log_one(w1, "prompt")

        env_copy = None  # subprocess with the project venv via uv
        proc = subprocess.run(
            [sys.executable, "-m", "neuralguard.cli", "audit-verify", str(tmp_path)],
            capture_output=True,
            text=True,
            env=env_copy,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "ALL CHAINS VALID" in proc.stdout

        # JSON mode
        proc = subprocess.run(
            [sys.executable, "-m", "neuralguard.cli", "audit-verify", str(tmp_path), "--json"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        payload = json.loads(proc.stdout)
        assert payload["all_valid"] is True
        assert payload["chains"][0]["valid"] is True

        # Tamper -> exit 1
        audit_file = next(tmp_path.rglob("*.jsonl"))
        lines = audit_file.read_text().splitlines()
        event = json.loads(lines[0])
        event["confidence"] = 0.99
        lines[0] = json.dumps(event)
        audit_file.write_text("\n".join(lines) + "\n")
        proc = subprocess.run(
            [sys.executable, "-m", "neuralguard.cli", "audit-verify", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1
        assert "CHAIN VERIFICATION FAILED" in proc.stdout

    def test_unreadable_path_exit_2(self) -> None:
        import subprocess
        import sys

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "neuralguard.cli",
                "audit-verify",
                "/nonexistent-path-should-fail",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2
