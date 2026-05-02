"""NeuralGuard semantic detection package (Phase 2).

Lazy imports — safe when [semantic] extra is not installed.
"""

from __future__ import annotations


def __getattr__(name: str) -> object:
    if name == "AttackCorpus":
        from neuralguard.semantic.corpus import AttackCorpus

        return AttackCorpus
    if name == "EmbeddingEngine":
        from neuralguard.semantic.embedding import EmbeddingEngine

        return EmbeddingEngine
    if name == "HybridScoringEngine":
        from neuralguard.semantic.hybrid import HybridScoringEngine

        return HybridScoringEngine
    if name == "JudgeScanner":
        from neuralguard.semantic.judge import JudgeScanner

        return JudgeScanner
    if name == "SimilarityScanner":
        from neuralguard.semantic.similarity import SimilarityScanner

        return SimilarityScanner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AttackCorpus",
    "EmbeddingEngine",
    "HybridScoringEngine",
    "JudgeScanner",
    "SimilarityScanner",
]
