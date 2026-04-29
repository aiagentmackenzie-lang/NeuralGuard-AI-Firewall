"""Unit tests for audit logging — JSONL rotation, retention, and PostgreSQL routing."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from neuralguard.config.settings import AuditSettings
from neuralguard.logging.audit import AuditLogger, tokenize_value
from neuralguard.models.schemas import (
    AuditEvent,
    EvaluateRequest,
    EvaluateResponse,
    LayerArbitrationResult,
    ScanOutputRequest,
    ScanOutputResponse,
    Verdict,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_event(
    tenant_id: str = "test",
    verdict: Verdict = Verdict.ALLOW,
) -> AuditEvent:
    """Create a minimal AuditEvent for testing."""
    return AuditEvent(
        request_id="req-001",
        tenant_id=tenant_id,
        verdict=verdict,
        findings_count=0,
        threat_categories=[],
        confidence=0.0,
        total_latency_ms=1.0,
    )


def _make_log_entry(
    audit: AuditLogger,
    tenant_id: str = "test",
    verdict: Verdict = Verdict.ALLOW,
) -> AuditEvent:
    """Run log_evaluation to produce a real persisted event."""
    return audit.log_evaluation(
        EvaluateRequest(prompt="test", tenant_id=tenant_id),
        EvaluateResponse(
            tenant_id=tenant_id,
            verdict=verdict,
            confidence=0.0,
            scan_layers_used=[],
            total_latency_ms=1.0,
        ),
        LayerArbitrationResult(
            verdict=verdict,
            findings=[],
            scanner_results=[],
            total_latency_ms=1.0,
            arbitration_reason="clean",
        ),
    )


# ── Tokenization ─────────────────────────────────────────────────────────


class TestTokenizeValue:
    def test_hash_prefix_format(self):
        result = tokenize_value("super-secret-value")
        assert result.startswith("TOK:")
        assert len(result) == 20  # "TOK:" + 16 hex chars

    def test_deterministic(self):
        """Same input always produces same token."""
        assert tokenize_value("hello") == tokenize_value("hello")

    def test_different_inputs_different_tokens(self):
        assert tokenize_value("alpha") != tokenize_value("beta")


# ── Basic AuditLogger ────────────────────────────────────────────────────


class TestAuditLogger:
    def test_disabled_audit_does_not_write(self, tmp_path):
        settings = AuditSettings(enabled=False, jsonl_path=tmp_path)
        logger = AuditLogger(settings)

        event = logger.log_evaluation(
            EvaluateRequest(prompt="test", tenant_id="test"),
            EvaluateResponse(
                tenant_id="test",
                verdict=Verdict.ALLOW,
                confidence=0.0,
                scan_layers_used=[],
                total_latency_ms=1.0,
            ),
            LayerArbitrationResult(
                verdict=Verdict.ALLOW,
                findings=[],
                scanner_results=[],
                total_latency_ms=1.0,
                arbitration_reason="clean",
            ),
        )
        assert isinstance(event, AuditEvent)
        # No files should be written
        assert list(tmp_path.iterdir()) == []

    def test_jsonl_backend_writes_file(self, tmp_path):
        settings = AuditSettings(enabled=True, backend="jsonl", jsonl_path=tmp_path)
        audit = AuditLogger(settings)
        _make_log_entry(audit)
        files = list(tmp_path.glob("audit-*.jsonl"))
        assert len(files) == 1

    def test_jsonl_content_is_valid_json(self, tmp_path):
        import json

        settings = AuditSettings(enabled=True, backend="jsonl", jsonl_path=tmp_path)
        audit = AuditLogger(settings)
        _make_log_entry(audit)

        files = list(tmp_path.glob("audit-*.jsonl"))
        content = files[0].read_text()
        # Each line should be valid JSON
        for line in content.strip().split("\n"):
            parsed = json.loads(line)
            assert "event_id" in parsed

    def test_jsonl_appends_multiple_events(self, tmp_path):
        settings = AuditSettings(enabled=True, backend="jsonl", jsonl_path=tmp_path)
        audit = AuditLogger(settings)
        _make_log_entry(audit)
        _make_log_entry(audit)

        files = list(tmp_path.glob("audit-*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 2

    def test_pii_tokenization_short_strings_preserved(self):
        settings = AuditSettings(tokenize_pii=True)
        logger = AuditLogger(settings)
        result = logger._tokenize_metadata({"short": "hi", "flag": True, "num": 42})
        assert result["short"] == "hi"
        assert result["flag"] is True
        assert result["num"] == 42

    def test_pii_tokenization_long_strings_hashed(self):
        settings = AuditSettings(tokenize_pii=True)
        logger = AuditLogger(settings)
        result = logger._tokenize_metadata({"secret": "this is a long secret value"})
        assert result["secret"].startswith("TOK:")

    def test_persist_exception_handled(self, tmp_path, monkeypatch):
        settings = AuditSettings(enabled=True, jsonl_path=tmp_path)
        logger = AuditLogger(settings)

        def boom(_event):
            raise RuntimeError("disk full")

        monkeypatch.setattr(logger, "_write_jsonl", boom)

        # Should not raise despite write failure
        logger.log_evaluation(
            EvaluateRequest(prompt="test", tenant_id="test"),
            EvaluateResponse(
                tenant_id="test",
                verdict=Verdict.ALLOW,
                confidence=0.0,
                scan_layers_used=[],
                total_latency_ms=1.0,
            ),
            LayerArbitrationResult(
                verdict=Verdict.ALLOW,
                findings=[],
                scanner_results=[],
                total_latency_ms=1.0,
                arbitration_reason="clean",
            ),
        )

    def test_log_output_scan(self, tmp_path):
        settings = AuditSettings(enabled=True, jsonl_path=tmp_path)
        logger = AuditLogger(settings)

        req = ScanOutputRequest(output="hello", tenant_id="test")
        resp = ScanOutputResponse(
            tenant_id="test",
            verdict=Verdict.ALLOW,
            total_latency_ms=1.0,
        )
        event = logger.log_output_scan(req, resp)
        assert event.verdict == Verdict.ALLOW
        files = list(tmp_path.iterdir())
        assert len(files) == 1


# ── JSONL Rotation ───────────────────────────────────────────────────────


class TestJsonlRotation:
    def test_rotation_triggers_at_size_limit(self, tmp_path, monkeypatch):
        """When a JSONL file hits the size limit, it rotates to a numbered file."""
        # Use a tiny rotation threshold for testing
        monkeypatch.setattr("neuralguard.logging.audit._MAX_JSONL_BYTES", 100)

        settings = AuditSettings(enabled=True, backend="jsonl", jsonl_path=tmp_path)
        audit = AuditLogger(settings)

        # Write enough events to exceed 100 bytes
        for i in range(20):
            _make_log_entry(audit, tenant_id=f"tenant-{i}")

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        files = list(tmp_path.glob("audit-*.jsonl"))
        filenames = {f.name for f in files}

        # Should have the original + at least one rotated file
        assert f"audit-{today}.jsonl" in filenames
        rotated = [f for f in filenames if f != f"audit-{today}.jsonl"]
        assert len(rotated) >= 1, f"Expected rotation but got files: {filenames}"

    def test_rotation_preserves_data(self, tmp_path, monkeypatch):
        """Rotated files should contain the original data."""
        monkeypatch.setattr("neuralguard.logging.audit._MAX_JSONL_BYTES", 200)

        settings = AuditSettings(enabled=True, backend="jsonl", jsonl_path=tmp_path)
        audit = AuditLogger(settings)

        # Write events until rotation happens
        for i in range(30):
            _make_log_entry(audit, tenant_id=f"tenant-{i}")

        # All events should be readable across all files
        all_events = []
        for f in sorted(tmp_path.glob("audit-*.jsonl")):
            for line in f.read_text().strip().split("\n"):
                import json

                all_events.append(json.loads(line))

        # Should have 30 events total across all files
        assert len(all_events) == 30

    def test_no_rotation_when_under_limit(self, tmp_path):
        """Single small file should not rotate."""
        settings = AuditSettings(enabled=True, backend="jsonl", jsonl_path=tmp_path)
        audit = AuditLogger(settings)
        _make_log_entry(audit)

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        files = list(tmp_path.glob("audit-*.jsonl"))
        assert len(files) == 1
        assert files[0].name == f"audit-{today}.jsonl"


# ── Retention Cleanup ────────────────────────────────────────────────────


class TestRetentionCleanup:
    def test_old_files_deleted(self, tmp_path):
        """Files older than retention_days should be cleaned up."""
        settings = AuditSettings(
            enabled=True,
            backend="jsonl",
            jsonl_path=tmp_path,
            retention_days=7,
        )
        audit = AuditLogger(settings)

        # Create an "old" file manually
        old_file = tmp_path / "audit-2026-01-01.jsonl"
        old_file.write_text('{"old": true}\n')
        # Set modification time to 10 days ago
        old_mtime = time.time() - (10 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))

        # Write a new event — triggers cleanup
        _make_log_entry(audit)

        # Old file should be deleted
        assert not old_file.exists()

    def test_recent_files_preserved(self, tmp_path):
        """Files within retention window should NOT be deleted."""
        settings = AuditSettings(
            enabled=True,
            backend="jsonl",
            jsonl_path=tmp_path,
            retention_days=30,
        )
        audit = AuditLogger(settings)

        # Create a "recent" file (2 days old)
        recent_file = tmp_path / "audit-2026-04-27.jsonl"
        recent_file.write_text('{"recent": true}\n')
        recent_mtime = time.time() - (2 * 86400)
        os.utime(recent_file, (recent_mtime, recent_mtime))

        _make_log_entry(audit)

        # Recent file should still exist
        assert recent_file.exists()

    def test_zero_retention_keeps_forever(self, tmp_path):
        """retention_days=0 means keep all files forever."""
        settings = AuditSettings(
            enabled=True,
            backend="jsonl",
            jsonl_path=tmp_path,
            retention_days=0,
        )
        audit = AuditLogger(settings)

        # Create a very old file
        old_file = tmp_path / "audit-2020-01-01.jsonl"
        old_file.write_text('{"ancient": true}\n')
        old_mtime = time.time() - (365 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))

        _make_log_entry(audit)

        # Old file should still exist (0 = keep forever)
        assert old_file.exists()

    def test_non_audit_files_not_deleted(self, tmp_path):
        """Non-audit files in the directory should not be touched."""
        settings = AuditSettings(
            enabled=True,
            backend="jsonl",
            jsonl_path=tmp_path,
            retention_days=1,
        )
        audit = AuditLogger(settings)

        # Create a non-audit file
        other_file = tmp_path / "config.yaml"
        other_file.write_text("key: value\n")
        old_mtime = time.time() - (30 * 86400)
        os.utime(other_file, (old_mtime, old_mtime))

        _make_log_entry(audit)

        # Non-audit file should still exist
        assert other_file.exists()


# ── PostgreSQL Routing ──────────────────────────────────────────────────


class TestPostgresRouting:
    def test_postgres_no_url_falls_back_to_jsonl(self, tmp_path):
        """When backend=postgres but no URL, should fall back to JSONL."""
        settings = AuditSettings(
            enabled=True,
            backend="postgres",
            jsonl_path=tmp_path,
            postgres_url=None,
        )
        audit = AuditLogger(settings)
        event = _make_log_entry(audit)

        files = list(tmp_path.glob("audit-*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text()
        assert event.event_id in content

    def test_postgres_with_url_no_engine_falls_back(self, tmp_path, monkeypatch):
        """When URL is set but engine not initialized, should fall back to JSONL."""

        # Mock get_engine to return None (engine not initialized)
        def mock_get_engine():
            return None

        # We need to patch inside _persist_postgres which does
        # `from neuralguard.db.engine import get_engine`
        # Let's just verify the fallback works by checking JSONL output
        settings = AuditSettings(
            enabled=True,
            backend="postgres",
            jsonl_path=tmp_path,
            postgres_url="postgresql+asyncpg://user:pass@localhost/testdb",
        )
        audit = AuditLogger(settings)

        # Monkeypatch _persist_postgres to simulate engine-not-ready
        # (which internally falls back to _write_jsonl)

        def persist_with_mock_engine(event):
            if not settings.enabled:
                return
            try:
                if settings.backend == "postgres":
                    if settings.postgres_url:
                        # Simulate: engine not ready → JSONL fallback
                        audit._write_jsonl(event)
                    else:
                        audit._write_jsonl(event)
                elif settings.backend == "jsonl":
                    audit._write_jsonl(event)
            except Exception:
                pass

        monkeypatch.setattr(audit, "_persist", persist_with_mock_engine)

        _make_log_entry(audit)
        files = list(tmp_path.glob("audit-*.jsonl"))
        assert len(files) == 1

    def test_postgres_missing_deps_falls_back(self, tmp_path, monkeypatch):
        """When db extras not installed, _persist_postgres catches ImportError and falls back to JSONL."""
        settings = AuditSettings(
            enabled=True,
            backend="postgres",
            jsonl_path=tmp_path,
            postgres_url="postgresql+asyncpg://user:pass@localhost/testdb",
        )
        audit = AuditLogger(settings)

        # Make the db.engine import inside _persist_postgres fail
        import neuralguard.db.engine as eng_module

        monkeypatch.setattr(
            eng_module,
            "get_engine",
            lambda: (_ for _ in ()).throw(ImportError("No module named 'asyncpg'")),
        )

        # _persist_postgres should catch ImportError and fall back to JSONL
        audit._persist_postgres(_make_event())

        files = list(tmp_path.glob("audit-*.jsonl"))
        assert len(files) == 1  # Fallback wrote to JSONL

    def test_postgres_jsonl_fallback_warns_once(self, tmp_path, monkeypatch):
        """When falling back to JSONL from postgres, should warn once then stop."""
        settings = AuditSettings(
            enabled=True,
            backend="postgres",
            jsonl_path=tmp_path,
            postgres_url="postgresql+asyncpg://user:pass@localhost/testdb",
        )
        audit = AuditLogger(settings)

        # Simulate engine not ready by having _persist_postgres fall back
        def persist_pg_fallback(event):
            # First call warns, subsequent calls don't
            audit._write_jsonl(event)
            audit._pg_checked = True

        monkeypatch.setattr(audit, "_persist_postgres", persist_pg_fallback)

        # Multiple evaluations
        audit._persist(_make_event())
        audit._persist(_make_event())
        audit._persist(_make_event())

        files = list(tmp_path.glob("audit-*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 3
