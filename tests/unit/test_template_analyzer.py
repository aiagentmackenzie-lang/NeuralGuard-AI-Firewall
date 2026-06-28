"""Tests for the prompt-template analyzer (Sprint B, B2).

Covers the pure TemplateAnalyzer, the `neuralguard analyze-template` CLI,
and the POST /v1/analyze/template endpoint.
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from neuralguard.analysis import TemplateAnalyzer
from neuralguard.config.settings import NeuralGuardConfig
from neuralguard.main import create_app

# ── Analyzer unit tests ────────────────────────────────────────────────────


class TestTemplateAnalyzerSinks:
    def test_clean_template_no_sinks(self) -> None:
        t = "You are a helpful assistant. Answer concisely in {{language}}."
        result = TemplateAnalyzer().analyze(t)
        assert result.is_clean
        assert result.sinks == []

    def test_untrusted_variable_in_system_is_high_sink(self) -> None:
        t = "You are an assistant.\n{{user_input}}\n"
        result = TemplateAnalyzer().analyze(t)
        ids = [s.rule_id for s in result.sinks]
        assert "TPL-SINK-001" in ids
        sink = next(s for s in result.sinks if s.rule_id == "TPL-SINK-001")
        assert sink.severity == "high"
        assert sink.location == 2

    def test_untrusted_angle_bracket_marker_flagged(self) -> None:
        t = "System: follow the rules.\n<user_input>\n"
        result = TemplateAnalyzer().analyze(t)
        assert any(s.rule_id == "TPL-SINK-001" for s in result.sinks)

    def test_trusted_variable_not_flagged(self) -> None:
        t = "Answer in {{language}} on {{date}} as {{persona}}."
        result = TemplateAnalyzer().analyze(t)
        # Trusted names produce no untrusted sink; unknown-but-bounded check
        # only fires for non-trusted names, so a fully-trusted template is clean.
        assert result.is_clean

    def test_unbounded_unknown_variable_medium_sink(self) -> None:
        t = "Use the value {{foobar}} for the calculation."
        result = TemplateAnalyzer().analyze(t)
        assert any(s.rule_id == "TPL-SINK-002" for s in result.sinks)

    def test_action_adjacent_variable_high_sink(self) -> None:
        t = "Execute {{query}} with the shell tool."
        result = TemplateAnalyzer().analyze(t)
        ids = [s.rule_id for s in result.sinks]
        assert "TPL-SINK-005" in ids
        sink = next(s for s in result.sinks if s.rule_id == "TPL-SINK-005")
        assert sink.severity == "high"

    def test_structured_data_raw_variable_low_sink(self) -> None:
        t = "Inject the payload:\n{{json}}\n"
        result = TemplateAnalyzer().analyze(t)
        assert any(s.rule_id == "TPL-SINK-006" for s in result.sinks)

    def test_missing_fence_when_user_content_present(self) -> None:
        t = "You are an assistant.\n{{user_input}}\n"
        result = TemplateAnalyzer().analyze(t)
        # No delimiter fence + untrusted content -> TPL-SINK-003.
        assert any(s.rule_id == "TPL-SINK-003" for s in result.sinks)

    def test_fence_present_no_missing_fence_sink(self) -> None:
        t = (
            "You are an assistant.\n"
            "--- BEGIN SYSTEM ---\n"
            "Follow the rules.\n"
            "--- END SYSTEM ---\n"
            "{{user_input}}\n"
        )
        result = TemplateAnalyzer().analyze(t)
        assert not any(s.rule_id == "TPL-SINK-003" for s in result.sinks)

    def test_multiple_system_headers_ambiguous_precedence(self) -> None:
        t = "System: rule one.\nDo things.\nSystem: rule two.\n"
        result = TemplateAnalyzer().analyze(t)
        assert any(s.rule_id == "TPL-SINK-004" for s in result.sinks)

    def test_user_header_before_system_header_inverts_precedence(self) -> None:
        t = "User: here is my question.\nSystem: follow the rules.\n"
        result = TemplateAnalyzer().analyze(t)
        ids = [s.rule_id for s in result.sinks]
        assert "TPL-SINK-004" in ids

    def test_empty_template_is_clean(self) -> None:
        assert TemplateAnalyzer().analyze("").is_clean
        assert TemplateAnalyzer().analyze("   \n").is_clean

    def test_result_to_dict_shape(self) -> None:
        t = "{{user_input}}\n"
        d = TemplateAnalyzer().analyze(t).to_dict()
        assert d["is_clean"] is False
        assert d["sink_count"] >= 1
        assert isinstance(d["sinks"], list)
        assert {"rule_id", "severity", "description", "remediation", "evidence", "location"} <= set(
            d["sinks"][0]
        )

    def test_sinks_deduplicated(self) -> None:
        """The same sink (rule_id + evidence + location) is not reported twice."""
        t = "Execute {{query}} with the tool.\n"
        result = TemplateAnalyzer().analyze(t)
        sink005 = [s for s in result.sinks if s.rule_id == "TPL-SINK-005"]
        assert len(sink005) == 1


# ── CLI tests ──────────────────────────────────────────────────────────────


class TestAnalyzeTemplateCLI:
    def _run(self, template: str, *extra: str) -> tuple[int, str]:
        import argparse
        import io
        import sys

        from neuralguard.cli import _cmd_analyze_template

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        at = sub.add_parser("analyze-template")
        at.add_argument("template")
        at.add_argument("--json", action="store_true")
        at.add_argument("--fail-on-high", action="store_true")
        at.set_defaults(func=_cmd_analyze_template)

        argv = ["analyze-template", "-", *extra]
        args = parser.parse_args(argv)
        old_stdin, old_stdout = sys.stdin, sys.stdout
        sys.stdin = io.StringIO(template)
        sys.stdout = io.StringIO()
        try:
            code = _cmd_analyze_template(args)
        finally:
            sys.stdin, sys.stdout = old_stdin, old_stdout
        return code, ""  # stdout captured but we test exit codes + json here

    def test_clean_template_exits_zero(self) -> None:
        code, _ = self._run("You are a helpful assistant. Answer in {{language}}.")
        assert code == 0

    def test_high_sink_with_fail_on_high_exits_one(self) -> None:
        code, _ = self._run("{{user_input}}\n", "--fail-on-high")
        assert code == 1

    def test_sinks_without_fail_on_high_exits_zero(self) -> None:
        # Has sinks (incl high) but no --fail-on-high -> exit 0 (reported, not gating).
        code, _ = self._run("{{user_input}}\n")
        assert code == 0

    def test_json_output(self) -> None:
        import argparse
        import io
        import sys

        from neuralguard.cli import _cmd_analyze_template

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        at = sub.add_parser("analyze-template")
        at.add_argument("template")
        at.add_argument("--json", action="store_true")
        at.add_argument("--fail-on-high", action="store_true")
        at.set_defaults(func=_cmd_analyze_template)
        args = parser.parse_args(["analyze-template", "-", "--json"])
        old_stdin, old_stdout = sys.stdin, sys.stdout
        sys.stdin = io.StringIO("{{user_input}}\n")
        sys.stdout = io.StringIO()
        try:
            _cmd_analyze_template(args)
            out = sys.stdout.getvalue()
        finally:
            sys.stdin, sys.stdout = old_stdin, old_stdout
        data = json.loads(out)
        assert data["is_clean"] is False
        assert data["sink_count"] >= 1


# ── Endpoint tests ─────────────────────────────────────────────────────────


@pytest.fixture
def app():
    return create_app(NeuralGuardConfig(environment="development"))


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestAnalyzeTemplateEndpoint:
    @pytest.mark.asyncio
    async def test_clean_template(self, client) -> None:
        r = await client.post(
            "/v1/analyze/template",
            json={"template": "You are a helpful assistant. Answer in {{language}}."},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["is_clean"] is True
        assert data["sink_count"] == 0
        assert data["sinks"] == []
        assert "total_latency_ms" in data

    @pytest.mark.asyncio
    async def test_untrusted_sink_reported(self, client) -> None:
        r = await client.post(
            "/v1/analyze/template",
            json={"template": "You are an assistant.\n{{user_input}}\nExecute {{query}}."},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["is_clean"] is False
        ids = {s["rule_id"] for s in data["sinks"]}
        assert "TPL-SINK-001" in ids
        assert "TPL-SINK-005" in ids

    @pytest.mark.asyncio
    async def test_empty_template_rejected(self, client) -> None:
        r = await client.post("/v1/analyze/template", json={"template": "   "})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_template_rejected(self, client) -> None:
        r = await client.post("/v1/analyze/template", json={})
        assert r.status_code == 422
