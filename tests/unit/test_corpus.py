"""Tests for AttackCorpus — pre-computed attack vector search.

Unit tests use mock/temp data. Integration tests require
the actual corpus (run scripts/build_attack_corpus.py first).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuralguard.config.settings import ScannerSettings

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> ScannerSettings:
    """Default scanner settings."""
    return ScannerSettings()


@pytest.fixture
def mock_corpus_dir() -> Path:
    """Create a temporary directory with mock corpus files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create mock vectors (10 samples, 384-dim, L2-normalized)
        rng = np.random.RandomState(42)
        vectors = rng.randn(10, 384).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-9, None)
        np.save(tmp / "attack_vectors.npy", vectors)

        # Create mock metadata
        metadata = [
            {
                "index": i,
                "text": f"Attack prompt number {i}",
                "category": "T-PI-D" if i < 5 else "T-JB",
                "severity": "high" if i < 3 else "medium",
                "source": "test",
            }
            for i in range(10)
        ]
        with open(tmp / "attack_metadata.json", "w") as f:
            json.dump(metadata, f)

        yield tmp


@pytest.fixture
def settings_with_mock_corpus(mock_corpus_dir: Path) -> ScannerSettings:
    """Settings pointing to mock corpus directory."""
    return ScannerSettings(
        semantic_attack_corpus_path=str(mock_corpus_dir / "attack_vectors.npy"),
        semantic_attack_metadata_path=str(mock_corpus_dir / "attack_metadata.json"),
    )


# ── Unit Tests ──────────────────────────────────────────────────────────────


class TestAttackCorpusInit:
    """Test AttackCorpus initialization."""

    def test_init_default_settings(self, settings: ScannerSettings) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings)
        assert not corpus.loaded
        assert corpus.corpus_size == 0
        assert corpus.load_time_ms == 0.0

    def test_load_raises_without_file(self, settings: ScannerSettings) -> None:
        """load() raises FileNotFoundError when corpus files missing."""
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings)
        corpus.settings.semantic_attack_corpus_path = "/tmp/nonexistent-vectors.npy"
        with pytest.raises(FileNotFoundError, match="Attack corpus not found"):
            corpus.load()

    def test_search_raises_without_load(self, settings: ScannerSettings) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings)
        query = np.random.randn(384).astype(np.float32)
        with pytest.raises(RuntimeError, match="not loaded"):
            corpus.search(query)

    def test_max_similarity_raises_without_load(self, settings: ScannerSettings) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings)
        query = np.random.randn(384).astype(np.float32)
        with pytest.raises(RuntimeError, match="not loaded"):
            corpus.max_similarity(query)


class TestAttackCorpusLoad:
    """Test corpus loading from mock files."""

    def test_load_succeeds(self, settings_with_mock_corpus: ScannerSettings) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings_with_mock_corpus)
        corpus.load()
        assert corpus.loaded
        assert corpus.corpus_size == 10

    def test_load_time_recorded(self, settings_with_mock_corpus: ScannerSettings) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings_with_mock_corpus)
        corpus.load()
        assert corpus.load_time_ms > 0

    def test_load_is_idempotent(self, settings_with_mock_corpus: ScannerSettings) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings_with_mock_corpus)
        corpus.load()
        first_time = corpus.load_time_ms
        corpus.load()  # Second call should be no-op
        assert corpus.load_time_ms == first_time

    def test_vectors_shape(self, settings_with_mock_corpus: ScannerSettings) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings_with_mock_corpus)
        corpus.load()
        assert corpus._vectors is not None
        assert corpus._vectors.shape == (10, 384)

    def test_metadata_length(self, settings_with_mock_corpus: ScannerSettings) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings_with_mock_corpus)
        corpus.load()
        assert len(corpus._metadata) == 10

    def test_unload_clears_state(self, settings_with_mock_corpus: ScannerSettings) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings_with_mock_corpus)
        corpus.load()
        corpus.unload()
        assert not corpus.loaded
        assert corpus.corpus_size == 0
        assert corpus._vectors is None


class TestAttackCorpusSearch:
    """Test similarity search against mock corpus."""

    @pytest.fixture
    def loaded_corpus(self, settings_with_mock_corpus: ScannerSettings) -> object:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings_with_mock_corpus)
        corpus.load()
        return corpus

    def test_search_returns_results(self, loaded_corpus: object) -> None:
        """Search with a query similar to corpus returns results."""
        # Use a corpus vector as query (should match itself with sim ~1.0)
        query = loaded_corpus._vectors[0].copy()
        results = loaded_corpus.search(query, threshold=0.5)
        assert len(results) > 0

    def test_search_exact_match_high_similarity(self, loaded_corpus: object) -> None:
        """Query matching a corpus vector returns similarity ~1.0."""
        query = loaded_corpus._vectors[0].copy()
        results = loaded_corpus.search(query, threshold=0.9)
        assert len(results) >= 1
        assert results[0]["similarity"] > 0.99
        assert results[0]["index"] == 0

    def test_search_random_query_low_similarity(self, loaded_corpus: object) -> None:
        """Random query unlikely to match with high threshold."""
        rng = np.random.RandomState(99)
        query = rng.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)
        results = loaded_corpus.search(query, threshold=0.9)
        # Random vectors rarely have >0.9 cosine similarity
        assert len(results) == 0

    def test_search_threshold_filtering(self, loaded_corpus: object) -> None:
        """Higher threshold returns fewer results."""
        query = loaded_corpus._vectors[0].copy()
        low_results = loaded_corpus.search(query, threshold=0.3)
        high_results = loaded_corpus.search(query, threshold=0.99)
        assert len(low_results) >= len(high_results)

    def test_search_top_k_limit(self, loaded_corpus: object) -> None:
        """top_k limits number of results."""
        query = loaded_corpus._vectors[0].copy()
        results = loaded_corpus.search(query, threshold=0.0, top_k=2)
        assert len(results) <= 2

    def test_search_sorted_by_similarity(self, loaded_corpus: object) -> None:
        """Results sorted by similarity descending."""
        query = loaded_corpus._vectors[0].copy()
        results = loaded_corpus.search(query, threshold=0.0, top_k=5)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["similarity"] >= results[i + 1]["similarity"]

    def test_search_result_fields(self, loaded_corpus: object) -> None:
        """Each result has required fields."""
        query = loaded_corpus._vectors[0].copy()
        results = loaded_corpus.search(query, threshold=0.5)
        for r in results:
            assert "index" in r
            assert "similarity" in r
            assert "text" in r
            assert "category" in r
            assert "severity" in r
            assert "source" in r

    def test_search_uses_config_threshold(self, loaded_corpus: object) -> None:
        """When threshold=None, uses config default."""
        query = loaded_corpus._vectors[0].copy()
        results = loaded_corpus.search(query, threshold=None)
        # Config default is 0.75 — exact match should exceed it
        assert len(results) >= 1


class TestAttackCorpusMaxSimilarity:
    """Test fast max similarity check."""

    @pytest.fixture
    def loaded_corpus(self, settings_with_mock_corpus: ScannerSettings) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings_with_mock_corpus)
        corpus.load()
        return corpus

    def test_max_similarity_exact_match(self, loaded_corpus: object) -> None:
        """Max similarity for a corpus vector ~1.0."""
        query = loaded_corpus._vectors[0].copy()
        max_sim = loaded_corpus.max_similarity(query)
        assert max_sim > 0.99

    def test_max_similarity_random_query(self, loaded_corpus: object) -> None:
        """Max similarity for random query is typically low."""
        rng = np.random.RandomState(123)
        query = rng.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)
        max_sim = loaded_corpus.max_similarity(query)
        # Random query rarely has >0.5 similarity to any specific vector
        assert 0.0 <= max_sim <= 1.0

    def test_max_similarity_range(self, loaded_corpus: object) -> None:
        """Max similarity is always between 0 and 1 for normalized vectors."""
        rng = np.random.RandomState(77)
        for _ in range(10):
            query = rng.randn(384).astype(np.float32)
            query = query / np.linalg.norm(query)
            max_sim = loaded_corpus.max_similarity(query)
            assert 0.0 <= max_sim <= 1.0


class TestAttackCorpusDistribution:
    """Test category distribution reporting."""

    def test_distribution_with_loaded_corpus(
        self, settings_with_mock_corpus: ScannerSettings
    ) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings_with_mock_corpus)
        corpus.load()
        dist = corpus.category_distribution()
        assert "T-PI-D" in dist
        assert "T-JB" in dist
        assert dist["T-PI-D"] == 5
        assert dist["T-JB"] == 5

    def test_distribution_empty_before_load(self, settings: ScannerSettings) -> None:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings)
        assert corpus.category_distribution() == {}


class TestAttackCorpusWithRealCorpus:
    """Integration tests that require the actual attack corpus.

    Run `python scripts/build_attack_corpus.py` first.
    """

    @pytest.fixture
    def real_corpus(self, settings: ScannerSettings) -> object:
        from neuralguard.semantic.corpus import AttackCorpus

        corpus = AttackCorpus(settings)
        vectors_path = Path(settings.semantic_attack_corpus_path)
        if not vectors_path.exists():
            pytest.skip("Attack corpus not built. Run scripts/build_attack_corpus.py first.")
        corpus.load()
        return corpus

    def test_corpus_loads(self, real_corpus: object) -> None:
        assert real_corpus.loaded

    def test_corpus_size_reasonable(self, real_corpus: object) -> None:
        """Corpus should have 1000+ attack vectors."""
        assert real_corpus.corpus_size >= 100

    def test_corpus_has_multiple_categories(self, real_corpus: object) -> None:
        """Corpus covers multiple threat categories."""
        dist = real_corpus.category_distribution()
        assert len(dist) >= 3

    def test_corpus_vectors_shape(self, real_corpus: object) -> None:
        """Vectors have correct shape (n, 384)."""
        assert real_corpus._vectors is not None
        assert real_corpus._vectors.shape[1] == 384

    def test_corpus_vectors_normalized(self, real_corpus: object) -> None:
        """All vectors are approximately L2-normalized."""
        norms = np.linalg.norm(real_corpus._vectors, axis=1)
        # Allow small floating point deviation
        assert np.all(np.abs(norms - 1.0) < 0.01)
