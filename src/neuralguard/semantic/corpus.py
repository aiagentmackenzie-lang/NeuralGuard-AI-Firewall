"""Attack corpus loader — loads pre-computed attack vectors for similarity search.

Loads attack_vectors.npy and attack_metadata.json produced by
scripts/build_attack_corpus.py. Provides efficient cosine similarity
search against the corpus.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from neuralguard.config.settings import ScannerSettings

logger = structlog.get_logger(__name__)


class AttackCorpus:
    """Pre-computed attack vector corpus for similarity search.

    Loads .npy embedding matrix and .json metadata once.
    Search via cosine similarity (dot product on L2-normalized vectors).
    """

    def __init__(self, settings: ScannerSettings) -> None:
        self.settings = settings
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] = []
        self._loaded = False
        self._load_time_ms: float = 0.0
        self._corpus_size: int = 0

    @property
    def loaded(self) -> bool:
        """Whether the corpus has been loaded."""
        return self._loaded

    @property
    def corpus_size(self) -> int:
        """Number of attack vectors in the corpus."""
        return self._corpus_size

    @property
    def load_time_ms(self) -> float:
        """Time taken to load the corpus in milliseconds."""
        return self._load_time_ms

    def load(self) -> None:
        """Load the attack corpus from disk.

        Raises:
            FileNotFoundError: If corpus files not found at configured paths.
        """
        if self._loaded:
            return

        start = time.perf_counter()

        vectors_path = Path(self.settings.semantic_attack_corpus_path)
        metadata_path = Path(self.settings.semantic_attack_metadata_path)

        if not vectors_path.exists():
            raise FileNotFoundError(
                f"Attack corpus not found at {vectors_path}. "
                f"Run: python scripts/build_attack_corpus.py"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Attack metadata not found at {metadata_path}. "
                f"Run: python scripts/build_attack_corpus.py"
            )

        # Load vectors — memory-mapped for efficiency with large corpora
        self._vectors = np.load(str(vectors_path), mmap_mode="r")
        self._corpus_size = self._vectors.shape[0]

        # Load metadata
        with open(metadata_path) as f:
            self._metadata = json.load(f)

        # Validate consistency
        if len(self._metadata) != self._corpus_size:
            logger.warning(
                "corpus_metadata_mismatch",
                vectors=self._corpus_size,
                metadata=len(self._metadata),
            )

        self._loaded = True
        self._load_time_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "attack_corpus_loaded",
            corpus_size=self._corpus_size,
            embedding_dim=self._vectors.shape[1] if self._vectors is not None else 0,
            load_time_ms=f"{self._load_time_ms:.1f}",
        )

    def search(
        self,
        query_embedding: np.ndarray,
        threshold: float | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search corpus for vectors similar to query_embedding.

        Args:
            query_embedding: L2-normalized 384-dim query vector.
            threshold: Minimum cosine similarity. Uses config default if None.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with: index, similarity, text, category, severity, source.
            Sorted by similarity descending. Only includes results above threshold.
        """
        if not self._loaded:
            raise RuntimeError("AttackCorpus not loaded. Call load() first.")

        if threshold is None:
            threshold = self.settings.semantic_similarity_threshold

        # Cosine similarity = dot product for L2-normalized vectors
        # query: (384,), vectors: (n, 384) → similarities: (n,)
        similarities = np.dot(self._vectors, query_embedding)  # type: ignore[union-attr]

        # Filter by threshold
        above_mask = similarities >= threshold
        if not np.any(above_mask):
            return []

        # Get top-k above threshold
        above_indices = np.where(above_mask)[0]
        above_sims = similarities[above_indices]

        # Sort descending
        sorted_order = np.argsort(above_sims)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for idx in sorted_order:
            orig_idx = int(above_indices[idx])
            sim = float(above_sims[idx])
            meta = self._metadata[orig_idx] if orig_idx < len(self._metadata) else {}

            results.append(
                {
                    "index": orig_idx,
                    "similarity": sim,
                    "text": meta.get("text", ""),
                    "category": meta.get("category", "unknown"),
                    "severity": meta.get("severity", "unknown"),
                    "source": meta.get("source", "unknown"),
                }
            )

        return results

    def max_similarity(self, query_embedding: np.ndarray) -> float:
        """Get the maximum cosine similarity to any vector in the corpus.

        Fast O(n) operation — useful for quick threshold checks.

        Args:
            query_embedding: L2-normalized 384-dim query vector.

        Returns:
            Maximum similarity score (0.0 to 1.0).
        """
        if not self._loaded:
            raise RuntimeError("AttackCorpus not loaded. Call load() first.")

        similarities = np.dot(self._vectors, query_embedding)  # type: ignore[union-attr]
        return float(np.max(similarities))

    def category_distribution(self) -> dict[str, int]:
        """Get the distribution of attack categories in the corpus."""
        if not self._metadata:
            return {}
        from collections import Counter

        return dict(Counter(m.get("category", "unknown") for m in self._metadata))

    def unload(self) -> None:
        """Release corpus from memory."""
        self._vectors = None
        self._metadata = []
        self._loaded = False
        self._corpus_size = 0
        logger.info("attack_corpus_unloaded")
