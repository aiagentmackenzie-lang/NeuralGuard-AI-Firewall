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
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_README = _REPO / "README.md"
_FPR_SLO = _REPO / "docs" / "FPR_SLO.md"


def _read(path: Path) -> str:
    if not path.exists():  # pragma: no cover - a missing doc is its own failure
        pytest.fail(f"expected doc at {path}")
    return path.read_text(encoding="utf-8")


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
