"""Semantic similarity scanner — Layer 3.

Compares incoming prompts against a pre-computed attack corpus using
cosine similarity on L2-normalized embeddings. Catches novel attacks
that bypass regex patterns but are semantically close to known attacks.

Design:
  - Inherits BaseScanner contract (fail-closed, latency tracking)
  - Uses EmbeddingEngine for ONNX inference
  - Uses AttackCorpus for similarity search
  - Only fires when semantic_enabled=True in config
  - Skips when previous layer already BLOCKed (early exit)
  - Maps similarity scores to verdicts via configurable thresholds
  - Maps corpus categories to NeuralGuard ThreatCategory enum

Target: <50ms P95 on CPU (embedding ~10ms + search ~1ms).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from neuralguard.models.schemas import (
    EvaluateRequest,
    Finding,
    ScanLayer,
    ScannerResult,
    Severity,
    ThreatCategory,
    Verdict,
)
from neuralguard.scanners.base import BaseScanner
from neuralguard.semantic.corpus import AttackCorpus
from neuralguard.semantic.embedding import EmbeddingEngine

if TYPE_CHECKING:
    from neuralguard.config.settings import ScannerSettings

logger = structlog.get_logger(__name__)

# ── Category mapping: corpus string → ThreatCategory enum ────────────────

_CORPUS_CATEGORY_MAP: dict[str, ThreatCategory] = {
    "T-PI-D": ThreatCategory.PROMPT_INJECTION_DIRECT,
    "T-PI-I": ThreatCategory.PROMPT_INJECTION_INDIRECT,
    "T-JB": ThreatCategory.JAILBREAK,
    "T-EXT": ThreatCategory.SYSTEM_PROMPT_EXTRACTION,
    "T-EXF": ThreatCategory.DATA_EXFILTRATION,
    "T-TOOL": ThreatCategory.TOOL_MISUSE,
    "T-AGT": ThreatCategory.AGENT_GOAL_HIJACK,
    "T-ENC": ThreatCategory.ENCODING_EVASION,
    "T-DOS": ThreatCategory.DOS_ABUSE,
    "T-OUT": ThreatCategory.IMPROPER_OUTPUT,
    "T-MEM": ThreatCategory.MEMORY_POISONING,
    "T-CASC": ThreatCategory.CASCADING_FAILURE,
    "T-NG": ThreatCategory.SELF_ATTACK,
}

# ── Severity mapping: corpus string → Severity enum ─────────────────────

_CORPUS_SEVERITY_MAP: dict[str, Severity] = {
    "critical": Severity.CRITICAL,
    "high": Severity.HIGH,
    "medium": Severity.MEDIUM,
    "low": Severity.LOW,
    "info": Severity.INFO,
}

# ── Rule ID prefix for semantic findings ─────────────────────────────────

_SEMANTIC_RULE_PREFIX = "SEM"

# Minimum similarity that produces an ESCALATE finding.
# Matches between this and semantic_similarity_threshold are ambiguous
# and need hybrid scoring + judge to resolve.
ESCALATE_FLOOR = 0.60


class SimilarityScanner(BaseScanner):
    """Layer 3: Semantic similarity scanner.

    Compares prompt embeddings against a corpus of known attack vectors.
    Returns findings with cosine similarity scores and matched attack metadata.
    """

    layer = ScanLayer.SEMANTIC

    def __init__(self, settings: ScannerSettings) -> None:
        super().__init__(settings)
        self._engine = EmbeddingEngine(settings)
        self._corpus = AttackCorpus(settings)
        self._initialized = False

    def initialize(self) -> None:
        """Load the embedding model and attack corpus.

        Called once before first scan. Safe to call multiple times (idempotent).
        Raises if model or corpus files are missing.
        """
        if self._initialized:
            return

        self._engine.load()
        self._corpus.load()
        self._initialized = True

        logger.info(
            "similarity_scanner_initialized",
            corpus_size=self._corpus.corpus_size,
            engine_load_ms=f"{self._engine.load_time_ms:.1f}",
            corpus_load_ms=f"{self._corpus.load_time_ms:.1f}",
        )

    @property
    def initialized(self) -> bool:
        """Whether the scanner has been initialized."""
        return self._initialized

    @property
    def engine(self) -> EmbeddingEngine:
        """Access the embedding engine (for testing/debugging)."""
        return self._engine

    @property
    def corpus(self) -> AttackCorpus:
        """Access the attack corpus (for testing/debugging)."""
        return self._corpus

    def scan(
        self, request: EvaluateRequest, context: dict[str, Any] | None = None
    ) -> ScannerResult:
        """Execute semantic similarity scan.

        Steps:
          1. Get input text from request
          2. Compute embedding via ONNX
          3. Search attack corpus for similar vectors
          4. Map top matches to findings with verdicts

        Args:
            request: The evaluation request.
            context: Pipeline context from previous layers.

        Returns:
            ScannerResult with semantic findings.
        """
        start = time.perf_counter()

        # Ensure initialized
        if not self._initialized:
            try:
                self.initialize()
            except Exception as exc:
                logger.error("similarity_scanner_init_failed", error=str(exc))
                return self._result(
                    Verdict.BLOCK,
                    [self._init_error_finding(str(exc))],
                    start,
                    error=f"Scanner init failed: {exc!r}",
                )

        # Get input text
        text = self._extract_text(request)
        if not text:
            return self._result(Verdict.ALLOW, [], start)

        # Skip if previous layer already BLOCKed (early exit)
        if context and context.get("pattern_verdict") == Verdict.BLOCK:
            logger.debug("similarity_scanner_skip_pattern_blocked")
            return self._result(Verdict.ALLOW, [], start)

        # Compute embedding
        try:
            embedding = self._engine.embed(text)
        except Exception as exc:
            logger.error("embedding_failed", error=str(exc))
            # Fail-closed: embedding error → BLOCK
            return self._result(
                Verdict.BLOCK,
                [self._embedding_error_finding(str(exc))],
                start,
                error=f"Embedding failed: {exc!r}",
            )

        # Search corpus
        # Compute the search threshold: use the lower of the configured
        # similarity threshold and ESCALATE_FLOOR, so we catch ambiguous
        # matches (0.60-0.74) that hybrid scoring + judge need to evaluate.
        threshold = self.settings.semantic_similarity_threshold
        search_threshold = min(threshold, ESCALATE_FLOOR)
        try:
            matches = self._corpus.search(embedding, threshold=search_threshold, top_k=3)
        except Exception as exc:
            logger.error("corpus_search_failed", error=str(exc))
            return self._result(
                Verdict.BLOCK,
                [self._corpus_error_finding(str(exc))],
                start,
                error=f"Corpus search failed: {exc!r}",
            )

        if not matches:
            logger.debug("similarity_scanner_no_matches", threshold=threshold)
            return self._result(Verdict.ALLOW, [], start)

        # Convert matches to findings
        findings: list[Finding] = []
        max_similarity = 0.0

        for i, match in enumerate(matches):
            sim = match["similarity"]
            if sim > max_similarity:
                max_similarity = sim

            category = self._map_category(match.get("category", "T-PI-D"))
            severity = self._map_severity(match.get("severity", "medium"))
            verdict = self._similarity_to_verdict(sim)
            rule_id = f"{_SEMANTIC_RULE_PREFIX}-{(i + 1):03d}"

            findings.append(
                Finding(
                    category=category,
                    severity=severity,
                    verdict=verdict,
                    confidence=sim,
                    layer=self.layer,
                    rule_id=rule_id,
                    description=(
                        f"Semantic similarity to known attack ({sim:.2f}): "
                        f"{match.get('text', '')[:80]}"
                    ),
                    evidence=f"category={match.get('category', '?')} source={match.get('source', '?')}",
                    mitigation=f"Review prompt for {category.value} intent",
                    metadata={
                        "similarity": sim,
                        "matched_index": match.get("index"),
                        "matched_category": match.get("category"),
                        "matched_source": match.get("source"),
                    },
                )
            )

        # Overall verdict: strictest from all matches
        overall_verdict = self._findings_to_verdict(findings)

        logger.info(
            "similarity_scan_complete",
            verdict=overall_verdict.value,
            max_similarity=f"{max_similarity:.3f}",
            matches=len(matches),
            latency_ms=f"{(time.perf_counter() - start) * 1000:.2f}",
        )

        return self._result(overall_verdict, findings, start)

    def _extract_text(self, request: EvaluateRequest) -> str:
        """Extract text content from the request."""
        if request.messages:
            return " ".join(m.content for m in request.messages)
        if request.prompt:
            return request.prompt
        return ""

    def _similarity_to_verdict(self, similarity: float) -> Verdict:
        """Map similarity score to verdict using config thresholds.

        Thresholds:
          >= block_threshold (0.75) → BLOCK (high confidence attack match)
          >= 0.60 → ESCALATE (ambiguous, needs hybrid + judge)
          < 0.60 → ALLOW (likely benign)
        """
        block_threshold = self.settings.semantic_similarity_threshold
        escalate_floor = ESCALATE_FLOOR

        if similarity >= block_threshold:
            return Verdict.BLOCK
        if similarity >= escalate_floor:
            return Verdict.ESCALATE
        return Verdict.ALLOW

    def _findings_to_verdict(self, findings: list[Finding]) -> Verdict:
        """Strictest verdict from findings."""
        if not findings:
            return Verdict.ALLOW

        priority = {
            Verdict.BLOCK: 6,
            Verdict.SANITIZE: 5,
            Verdict.ESCALATE: 4,
            Verdict.QUARANTINE: 3,
            Verdict.RATE_LIMIT: 2,
            Verdict.ALLOW: 0,
        }

        highest = Verdict.ALLOW
        highest_p = 0

        for f in findings:
            p = priority.get(f.verdict, 0)
            if p > highest_p:
                highest_p = p
                highest = f.verdict

        return highest

    @staticmethod
    def _map_category(corpus_category: str) -> ThreatCategory:
        """Map corpus category string to ThreatCategory enum."""
        return _CORPUS_CATEGORY_MAP.get(corpus_category, ThreatCategory.PROMPT_INJECTION_DIRECT)

    @staticmethod
    def _map_severity(corpus_severity: str) -> Severity:
        """Map corpus severity string to Severity enum."""
        return _CORPUS_SEVERITY_MAP.get(corpus_severity.lower(), Severity.MEDIUM)

    @staticmethod
    def _init_error_finding(error: str) -> Finding:
        """Finding for initialization failure."""
        return Finding(
            category=ThreatCategory.SELF_ATTACK,
            severity=Severity.HIGH,
            verdict=Verdict.BLOCK,
            confidence=1.0,
            layer=ScanLayer.SEMANTIC,
            rule_id="SEM-INIT-001",
            description=f"Semantic scanner initialization failed: {error}",
            mitigation="Ensure ONNX model and attack corpus are available",
        )

    @staticmethod
    def _embedding_error_finding(error: str) -> Finding:
        """Finding for embedding computation failure."""
        return Finding(
            category=ThreatCategory.SELF_ATTACK,
            severity=Severity.HIGH,
            verdict=Verdict.BLOCK,
            confidence=1.0,
            layer=ScanLayer.SEMANTIC,
            rule_id="SEM-EMB-001",
            description=f"Embedding computation failed: {error}",
            mitigation="Check ONNX Runtime installation and model integrity",
        )

    @staticmethod
    def _corpus_error_finding(error: str) -> Finding:
        """Finding for corpus search failure."""
        return Finding(
            category=ThreatCategory.SELF_ATTACK,
            severity=Severity.HIGH,
            verdict=Verdict.BLOCK,
            confidence=1.0,
            layer=ScanLayer.SEMANTIC,
            rule_id="SEM-CORP-001",
            description=f"Attack corpus search failed: {error}",
            mitigation="Verify attack_vectors.npy and attack_metadata.json",
        )
