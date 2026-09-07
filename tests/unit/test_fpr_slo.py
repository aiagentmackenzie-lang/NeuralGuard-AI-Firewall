"""Tests for the NG-6 guarded-FPR SLO machinery.

Unit tests use a mocked engine + synthetic corpus (no model artifacts). The
final test is the PUBLISHED INVARIANT: on the real rebuilt corpus + real
embedding model, the guarded FPR must be 0.00% — the same claim
docs/FPR_SLO.md makes. CI rebuilds the corpus before the test suite, so this
test enforces the published number on every PR.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from neuralguard.config.settings import ScannerSettings
from neuralguard.semantic.similarity import SimilarityScanner

_REAL_CORPUS = Path("models/attack_vectors.npy")
_REAL_MODEL = Path("models/embedding-onnx")

requires_corpus = pytest.mark.skipif(
    not (_REAL_CORPUS.exists() and _REAL_MODEL.exists()),
    reason="rebuilt corpus + ONNX model not present (CI order: rebuild first)",
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _write_guard(path: Path, ids: list[str]) -> None:
    with open(path, "w") as f:
        for i in ids:
            f.write(
                json.dumps({"id": i, "prompt": f"benign guard probe {i}", "category": "test"})
                + "\n"
            )


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


class _ScriptedEngine:
    """Returns a FIXED vector per probe text (deterministic, no model)."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vectors[t] for t in texts])


def _make_scanner(
    settings: ScannerSettings,
    corpus_vecs: np.ndarray,
    probe_vecs: dict[str, np.ndarray],
) -> SimilarityScanner:
    scanner = SimilarityScanner(settings)
    scanner._initialized = True  # skip real initialize() — no model I/O
    engine = MagicMock()
    engine.embed_batch.side_effect = _ScriptedEngine(probe_vecs).embed_batch
    scanner._engine = engine
    corpus = MagicMock()
    corpus.vectors = corpus_vecs
    corpus.corpus_size = corpus_vecs.shape[0]
    scanner._corpus = corpus
    return scanner


@pytest.fixture
def slo_env(tmp_path: Path) -> tuple[Path, Path]:
    benign = tmp_path / "benign.jsonl"
    hardneg = tmp_path / "hardneg.jsonl"
    _write_guard(benign, ["BEN-001", "BEN-002", "BEN-003", "BEN-004"])
    _write_guard(hardneg, ["HGN-001", "HGN-002", "HGN-003", "HGN-004"])
    return benign, hardneg


# ── Unit tests (no model required) ─────────────────────────────────────────


class TestMeasuredFprReport:
    def test_clean_corpus_reports_zero_fpr(self, tmp_path: Path, slo_env) -> None:
        benign, hardneg = slo_env
        settings = ScannerSettings(
            semantic_benign_guard_path=str(benign),
            semantic_hard_negatives_path=str(hardneg),
        )
        # Orthogonal probe embeddings; the corpus holds ONE vector that is
        # the (dim-1)-th basis vector — orthogonal to every probe basis
        # vector, so no probe can match above 0.
        probe_names = [f"benign guard probe BEN-{i:03d}" for i in range(1, 5)] + [
            f"benign guard probe HGN-{i:03d}" for i in range(1, 5)
        ]
        dim = len(probe_names) + 1
        probe_vecs = {
            name: _unit([1.0 if j == i else 0.0 for j in range(dim)])
            for i, name in enumerate(probe_names)
        }
        corpus_vecs = np.stack([_unit([0.0 if j < dim - 1 else 1.0 for j in range(dim)])])
        scanner = _make_scanner(settings, corpus_vecs, probe_vecs)

        report = scanner.measured_fpr_report()
        assert report is not None
        assert report["guarded_fpr_percent"] == 0.0
        assert report["slo_met"] is True
        assert report["benign"]["block_count"] == 0
        assert report["hard_negatives"]["escalate_count"] == 0
        assert report["benign"]["probes"] == 4
        assert report["hard_negatives"]["probes"] == 4

    def test_blocking_probe_is_counted_and_listed(self, tmp_path: Path, slo_env) -> None:
        benign, hardneg = slo_env
        settings = ScannerSettings(
            semantic_benign_guard_path=str(benign),
            semantic_hard_negatives_path=str(hardneg),
            semantic_fpr_slo=0.0,
        )
        probe_names = [f"benign guard probe BEN-{i:03d}" for i in range(1, 5)] + [
            f"benign guard probe HGN-{i:03d}" for i in range(1, 5)
        ]
        dim = len(probe_names) + 1
        probe_vecs = {
            name: _unit([1.0 if j == i else 0.0 for j in range(dim)])
            for i, name in enumerate(probe_names)
        }
        # Corpus contains a vector IDENTICAL to BEN-002's probe → sim 1.0 → BLOCK.
        corpus_vecs = np.stack([probe_vecs["benign guard probe BEN-002"]])
        scanner = _make_scanner(settings, corpus_vecs, probe_vecs)

        report = scanner.measured_fpr_report()
        assert report is not None
        assert report["guarded_fpr_percent"] > 0.0
        assert report["slo_met"] is False
        assert report["benign"]["block_count"] == 1
        assert report["benign"]["blockers"][0]["prompt"] == "benign guard probe BEN-002"

    def test_missing_guard_file_returns_none(self, tmp_path: Path, slo_env) -> None:
        benign, hardneg = slo_env
        hardneg.unlink()
        settings = ScannerSettings(
            semantic_benign_guard_path=str(benign),
            semantic_hard_negatives_path=str(hardneg),
        )
        scanner = _make_scanner(settings, np.zeros((1, 8), np.float32), {})
        assert scanner.measured_fpr_report() is None

    def test_empty_corpus_returns_none(self, tmp_path: Path, slo_env) -> None:
        benign, hardneg = slo_env
        settings = ScannerSettings(
            semantic_benign_guard_path=str(benign),
            semantic_hard_negatives_path=str(hardneg),
        )
        scanner = _make_scanner(
            settings, np.zeros((0, 8), np.float32), {"any": np.ones(8, np.float32)}
        )
        assert scanner.measured_fpr_report() is None


# ── The published invariant (CI-enforced on every PR) ──────────────────────


class TestPublishedInvariant:
    @requires_corpus
    def test_guarded_fpr_is_zero_on_published_corpus(self) -> None:
        """docs/FPR_SLO.md's headline number, enforced per-PR.

        The rebuild's hard-negative guard drops any corpus vector that would
        BLOCK a guard probe, so the measured guarded FPR on the rebuilt
        corpus MUST be 0.00%. A failure here means the shipped artifact and
        the guard sets have drifted apart — rebuild the corpus.
        """
        scanner = SimilarityScanner(ScannerSettings())
        report = scanner.measured_fpr_report()
        assert report is not None, "guard files missing on a corpus-bearing checkout"
        assert report["guarded_fpr_percent"] == 0.0, (
            f"guarded FPR regressed to {report['guarded_fpr_percent']}% — "
            f"blockers: {report['benign']['blockers'] + report['hard_negatives']['blockers']}"
        )
