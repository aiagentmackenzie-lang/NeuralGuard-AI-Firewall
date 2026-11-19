"""Docs-vs-docs consistency guards for measurable claims.

The README is the product's public contract. When it quotes a number that a
sibling doc also quotes, the two must agree — that is the cheapest available
check on documentation drift, and it is the exact defect this file was added
to catch (2026-11-11: README headlined a 7,623-vector semantic corpus while
``docs/FPR_SLO.md`` measured on the reproducible 6,503-vector rebuild).

Only mechanically comparable claims belong here. Benchmark tables are dated
measurements and are deliberately excluded: they record what a run produced,
not what the current build holds.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_README = _REPO / "README.md"
_FPR_SLO = _REPO / "docs" / "FPR_SLO.md"


def _read(path: Path) -> str:
    if not path.exists():  # pragma: no cover - a missing doc is its own failure
        pytest.fail(f"expected doc at {path}")
    return path.read_text(encoding="utf-8")


def _collected_test_count() -> int:
    """How many tests the suite contains, counted by pytest in a subprocess.

    ``--collect-only`` imports the test modules but executes nothing, so the
    guard test cannot recurse into itself. The cache provider is disabled so
    this collection cannot contend with the session that spawned it.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    matches = re.findall(r"(\d+) tests? collected", proc.stdout + proc.stderr)
    if proc.returncode != 0 or not matches:  # pragma: no cover - guards the guard
        pytest.fail(
            "could not collect the suite to count it "
            f"(exit {proc.returncode}):\n{(proc.stdout + proc.stderr)[-2000:]}"
        )
    return int(matches[-1])


def _corpus_size(text: str, *, where: str) -> int:
    """First ``N-vector corpus`` / ``corpus: N vectors`` figure in the text."""
    match = re.search(r"([\d,]+)-vector corpus", text) or re.search(
        r"corpus:\s*([\d,]+)\s+vectors", text
    )
    if match is None:  # pragma: no cover - guards the guard
        pytest.fail(f"no corpus-size claim found in {where}")
    return int(match.group(1).replace(",", ""))


class TestSemanticCorpusClaim:
    def test_readme_and_fpr_slo_quote_the_same_corpus_size(self) -> None:
        readme_size = _corpus_size(_read(_README), where="README.md")
        slo_size = _corpus_size(_read(_FPR_SLO), where="docs/FPR_SLO.md")
        assert readme_size == slo_size, (
            "README and docs/FPR_SLO.md disagree on the semantic corpus size "
            f"({readme_size:,} vs {slo_size:,}). One of them is stale: the "
            "figure must describe what scripts/rebuild_corpus_vectors.py "
            "actually produces from tracked sources."
        )


class TestTestCountClaim:
    """README's headline test count must equal what pytest actually collects.

    The other two guards here compare the README against a sibling doc or a
    runtime count. This one exists because the README drifted in the same way
    twice (2026-11-13: headlined 1,288 while the suite held 1,471; 2026-11-18:
    still headlined 1,518 while it held 1,565) and nothing caught it — the
    corpus and rule guards did not cover the number, so the number was
    decoration. A count that no test compares is a claim, not a measurement.

    Counting is delegated to pytest itself, in a subprocess, because a test
    cannot see the runner's own item list and hand-maintained arithmetic here
    would be the same unverified claim one level down.
    """

    def test_readme_test_count_matches_the_suite(self) -> None:
        readme = _read(_README)
        match = re.search(r"\*\*Current state:\*\*\s*([\d,]+) tests collected", readme)
        assert match is not None, (
            "README no longer states a collected test count in the form "
            "'**Current state:** N tests collected'"
        )
        claimed = int(match.group(1).replace(",", ""))
        collected = _collected_test_count()
        assert claimed == collected, (
            f"README claims {claimed:,} collected tests; the suite holds "
            f"{collected:,}. Update the README (or add/remove tests), never "
            "the test."
        )


class TestPatternRuleClaim:
    """README's rule count must equal the rules the scanner actually loads."""

    def test_readme_rule_count_matches_loaded_rules(self) -> None:
        from neuralguard.scanners.pattern import ALL_PATTERN_SETS
        from neuralguard.scanners.pattern_i18n import I18N_FLAT

        loaded = sum(len(pats) for _cat, pats in ALL_PATTERN_SETS) + len(I18N_FLAT)
        readme = _read(_README)
        match = re.search(r"(\d+) regex/heuristic rules", readme)
        assert match is not None, "README no longer states a rule count"
        assert int(match.group(1)) == loaded, (
            f"README claims {match.group(1)} regex/heuristic rules; the scanner "
            f"loads {loaded}. Update the README (or the rule packs), never the test."
        )
