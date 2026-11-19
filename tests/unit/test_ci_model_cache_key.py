"""The model/vector cache key must name the inputs that change a VECTOR.

History, because this key has already been wrong twice in opposite directions:

* Over-broad (v0.3.2 era): the key hashed all of ``src/neuralguard/semantic/**``,
  so an edit to ``hybrid.py`` — which cannot touch a vector — invalidated 1.1 GB
  and cost three consecutive 40-46 minute coverage runs where 10 would have done
  (LRN-20261119-004).
* Still over-broad (v0.3.3): the key hashed ``uv.lock`` wholesale and
  ``settings.py`` wholesale. Bumping the project's own version string rewrites
  ``uv.lock`` by one line, and the two coverage runs of v0.3.3 therefore paid the
  full cold path — measured 21 m 19 s and 30 m 14 s of corpus rebuild against a
  warm job of 7-10 min.

The correct input is neither a directory nor a lockfile: it is **the set of
values the embedding and rebuild paths actually read**, plus the resolved
versions of the packages that tokenize or do the arithmetic. Those are the only
things that can move a vector.

This test derives that set from the source rather than restating it, so adding
a setting to the embedding path fails here until the workflow covers it — the
failure mode LRN-20261118-001 describes (a guard that covers two of three
numbers reads as covering all three).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CI_YML = _REPO / ".github" / "workflows" / "ci.yml"

# The two modules that decide what a vector IS: one embeds, one rebuilds.
_VECTOR_INPUTS = (
    _REPO / "src" / "neuralguard" / "semantic" / "embedding.py",
    _REPO / "scripts" / "rebuild_corpus_vectors.py",
)


def _ci() -> str:
    return _CI_YML.read_text(encoding="utf-8")


def _settings_reads() -> set[str]:
    """Every ``settings.<name>`` the embedding/rebuild path reads, from source.

    Derived, not hard-coded: a new setting that changes an embedding must land
    in the fingerprint, and the only honest way to know the set is to read the
    code that consumes it.
    """
    names: set[str] = set()
    for path in _VECTOR_INPUTS:
        names.update(re.findall(r"settings\.([a-z_]+)", path.read_text(encoding="utf-8")))
    assert names, "the derived set is empty; the regex or the sources moved"
    return names


def _fingerprint_step() -> str:
    """The body of the step that computes the cache fingerprint."""
    ci = _ci()
    start = ci.index("Compute the semantic-stack fingerprint")
    end = ci.index("- name:", start)
    return ci[start:end]


class TestCacheKeyNamesTheRightInputs:
    def test_fingerprint_covers_every_setting_the_vector_path_reads(self) -> None:
        step = _fingerprint_step()
        missing = sorted(n for n in _settings_reads() if f"s.{n}" not in step)
        assert not missing, (
            "these settings change an embedding but are not in the cache "
            f"fingerprint, so a cache hit could serve vectors built under a "
            f"different value: {missing}"
        )

    def test_fingerprint_covers_the_numeric_stack_versions(self) -> None:
        step = _fingerprint_step()
        missing = [
            pkg
            for pkg in ("onnxruntime", "tokenizers", "numpy", "transformers", "optimum")
            if f'"{pkg}"' not in step
        ]
        assert not missing, (
            f"the fingerprint does not pin the numeric stack: {missing}. A "
            "tokenizer or onnxruntime bump changes embeddings with no other "
            "file changing."
        )

    def test_the_key_consumes_the_fingerprint(self) -> None:
        assert "steps.semfp.outputs.fingerprint" in _ci(), (
            "the fingerprint step exists but the cache key does not use it"
        )

    @pytest.mark.parametrize("bad_input", ["uv.lock", "config/settings.py"])
    def test_the_key_does_not_hash_whole_files_that_change_for_unrelated_reasons(
        self, bad_input: str
    ) -> None:
        """``uv.lock`` and ``settings.py`` are not inputs; their *relevant
        values* are. Hashing the file means a ruff bump or an auth setting
        invalidates 1.1 GB and buys a 20-30 minute cold rebuild."""
        ci = _ci()
        key_line = next(
            line for line in ci.splitlines() if line.strip().startswith("key: ng-model-")
        )
        assert bad_input not in key_line, (
            f"the cache key hashes {bad_input!r} wholesale. That is how v0.3.3 "
            "paid two cold rebuilds (21 m 19 s + 30 m 14 s) for a version bump: "
            "a near-miss is not the point — the file is simply not the input."
        )

    def test_restore_keys_stay_absent(self) -> None:
        """A partial match restores a stale artifact AND skips saving the new
        one, so every run pays the cold path (LRN-20261119-005). The comment in
        the workflow says why; this makes it a contract."""
        step = _ci().index("- name: Restore model + vector cache")
        block = _ci()[step : _ci().index("- name:", step + 10)]
        code = "\n".join(line for line in block.splitlines() if not line.strip().startswith("#"))
        assert "restore-keys" not in code, (
            "restore-keys is back in the model cache: a partial match will "
            "restore a stale artifact and skip the save"
        )
