"""Scanners package."""

from neuralguard.scanners.base import BaseScanner
from neuralguard.scanners.pattern import PatternScanner
from neuralguard.scanners.pipeline import ScannerPipeline
from neuralguard.scanners.structural import StructuralScanner

__all__ = ["BaseScanner", "PatternScanner", "ScannerPipeline", "StructuralScanner"]

# Lazy import for SimilarityScanner — requires [semantic] extra
try:
    from neuralguard.semantic.similarity import SimilarityScanner  # noqa: F401

    __all__.append("SimilarityScanner")
except ImportError:
    pass
