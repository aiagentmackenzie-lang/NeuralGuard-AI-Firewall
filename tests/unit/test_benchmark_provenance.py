"""Contracts for benchmark-artifact provenance and ignore/tracking agreement.

Two residuals were recorded in ``PRODUCTION_HARDENING_PLAN.md`` on 2026-11-13 and
left unscheduled; both are evidence-hygiene defects, not capability gaps:

1. ``benchmarks/ng_vs_ns/results/a2_results.json`` recorded models, iterations and
   configs but **no timestamp and no corpus digest**, so the tracked artifact could
   not prove which corpus built it. A JSON file cannot be trusted to describe a run
   it cannot locate in time.
2. ``.gitignore`` ignored ``benchmarks/ng_vs_ns/results/`` as "machine-local" while
   four files in that directory were tracked — a rule that says "not versioned"
   about files that are the published provenance record.

The contracts below are written so that a *correct* artifact passes and a *wrong*
one fails: they re-derive the digest from the corpus bytes rather than reading the
digest the harness wrote, and they ask git which paths it would ignore rather than
parsing the ignore file by eye.

Deliberately NOT asserted: that a historical artifact's ``generated_at`` is
non-null. The 2026-09-05 run predates the stamp and its clock time is not
recoverable; the contract is that the absence is *named*, never silent.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is required to ask what is tracked"
)


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        pytest.fail(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _tracked(relpath: str) -> list[str]:
    out = _git("ls-files", "--", relpath)
    return [line for line in out.splitlines() if line.strip()]


def _tracked_results_files() -> list[Path]:
    tracked = _tracked("benchmarks/ng_vs_ns/results")
    if not tracked:
        pytest.fail(
            "no tracked files under benchmarks/ng_vs_ns/results — the provenance record is gone"
        )
    return [_REPO / p for p in tracked]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestHarnessStampsProvenance:
    """The harness must record what it loaded, not just what it measured."""

    def test_corpus_provenance_digests_are_derived_from_the_corpus_bytes(self) -> None:
        from benchmarks.ng_vs_ns.live_harness import corpus_provenance

        prov = corpus_provenance()

        assert set(prov) == {"attack", "benign"}, (
            f"corpus_provenance() must name both corpora, got {sorted(prov)}"
        )
        for kind, expected_file in (
            ("attack", _REPO / "benchmarks" / "ng_vs_ns" / "attack_corpus.jsonl"),
            ("benign", _REPO / "benchmarks" / "ng_vs_ns" / "benign_corpus.jsonl"),
        ):
            entry = prov[kind]
            assert entry["file"].endswith(expected_file.name), (
                f"{kind} corpus path in the stamp ({entry['file']!r}) is not the "
                f"corpus the harness loads ({expected_file.name})"
            )
            assert entry["sha256"] == _sha256(expected_file), (
                f"{kind} corpus digest does not describe the tracked corpus bytes — a "
                "digest that cannot be re-derived is decoration, not provenance"
            )
            non_blank = [
                line
                for line in expected_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert entry["records"] == len(non_blank), (
                f"{kind} corpus record count {entry['records']} != {len(non_blank)} "
                "non-blank lines on disk"
            )

    def test_payload_stamps_generated_at_as_utc(self) -> None:
        from benchmarks.ng_vs_ns.live_harness import _payload

        payload = _payload(
            attacker_model="m",
            judge_model="m",
            judge_resolves_escalate=False,
            jb_iterations=1,
            selected_configs=["pattern_only"],
            attacks=[],
            benign=[],
            results=[],
        )

        assert "generated_at" in payload, "the results JSON must carry a timestamp"
        stamp = payload["generated_at"]
        assert isinstance(stamp, str) and stamp, "generated_at must be an ISO-8601 string"
        parsed = datetime.fromisoformat(stamp)
        assert parsed.tzinfo is not None, "generated_at must be timezone-aware (UTC)"
        assert parsed.utcoffset() == UTC.utcoffset(None), (
            "generated_at must be UTC: a local-time stamp in a tracking artifact is "
            "ambiguous for every reader in another timezone"
        )
        from benchmarks.ng_vs_ns.live_harness import corpus_provenance

        assert payload["corpus"] == corpus_provenance(), (
            "the stamped corpus provenance must be the one derived from the corpus "
            "files this checkout holds"
        )


class TestTrackedArtifactsDeclareProvenance:
    """Every tracked results JSON must be able to say when and from what."""

    def test_each_tracked_result_json_declares_its_origin(self) -> None:
        json_files = [p for p in _tracked_results_files() if p.suffix == ".json"]
        if not json_files:
            pytest.fail("no tracked results JSON — the README cites one as the provenance record")

        for path in json_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rel = path.relative_to(_REPO)

            assert "generated_at" in payload, (
                f"{rel} has no `generated_at` key. A results artifact that cannot date "
                "itself cannot prove it corresponds to the corpus it sits beside."
            )
            assert "corpus" in payload, (
                f"{rel} has no `corpus` key — it cannot prove which corpus produced it."
            )

            if payload["generated_at"] is None:
                reason = payload.get("provenance_unrecorded")
                assert isinstance(reason, str) and reason.strip(), (
                    f"{rel} has generated_at = null, so it MUST carry a "
                    "`provenance_unrecorded` sentence explaining the gap. An unnamed "
                    "absence is indistinguishable from an oversight."
                )
            else:
                parsed = datetime.fromisoformat(str(payload["generated_at"]))
                assert parsed.tzinfo is not None, f"{rel}: generated_at must be timezone-aware"


class TestIgnoreAndTrackingAgree:
    """The ignore rule must not describe tracked evidence as machine-local."""

    def test_no_tracked_result_file_is_ignored(self) -> None:
        for path in _tracked_results_files():
            rel = path.relative_to(_REPO).as_posix()
            # --no-index asks what the rules *would* do, independently of the index,
            # so a force-added file cannot hide a rule that contradicts its tracking.
            proc = subprocess.run(
                ["git", "check-ignore", "--no-index", "--", rel],
                cwd=_REPO,
                capture_output=True,
                text=True,
                check=False,
            )
            assert proc.returncode == 1, (
                f"{rel} is tracked but the ignore rules still match it "
                f"({proc.stdout.strip() or 'unknown rule'}). Either the rule moves or the "
                "tracking does — a contradiction that only survives because git was told "
                "force-add once is a trap for the next author."
            )
