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
from pathlib import Path
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
from neuralguard.semantic.hygiene import blocking_report, load_benign_guard_probes
from neuralguard.semantic.overflow import chunk_windows, contiguity_gate

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

# NG-6: ceiling for a per-tenant BLOCK threshold override. Above this a
# threshold stops meaning anything (a match that rare never protects anyone).
_TENANT_BLOCK_CEILING = 0.95

# Pipeline context key carrying the tenant's semantic BLOCK threshold.
_TENANT_BLOCK_THRESHOLD_KEY = "semantic_block_threshold"


class SimilarityScanner(BaseScanner["ScannerSettings"]):
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

    def measured_fpr_report(self) -> dict[str, Any] | None:
        """NG-6: guarded-FPR measurement (boot-time self-check, reusable).

        Embeds both guard probe sets — the F12 benign corpus and the NG-6
        NotInject-style hard negatives — and measures how the LOADED corpus
        treats each probe: how many probes' worst corpus match reaches the
        BLOCK threshold (an outright false positive) and how many land in
        the ambiguous zone (escalate; judge-resolvable, reported as
        pre-judge FPR).

        Returns None when the corpus is empty or a guard file is missing —
        an UNAVAILABLE metric is never silently reported as zero. Raises on
        engine failure (caller decides fail-closed vs degrade). The returned
        dict is JSON-safe for /v1/info.
        """
        self.initialize()
        corpus_vecs = self._corpus.vectors
        if corpus_vecs is None or self._corpus.corpus_size == 0:
            logger.warning("fpr_check_corpus_empty")
            return None

        block_threshold = self.settings.semantic_similarity_threshold
        guard_sets: dict[str, list[str]] = {}
        for name, path_str in (
            ("benign", self.settings.semantic_benign_guard_path),
            ("hard_negatives", self.settings.semantic_hard_negatives_path),
        ):
            path = Path(path_str)
            if not path.exists():
                logger.warning("fpr_guard_file_missing", guard=name, path=str(path))
                return None
            probes = load_benign_guard_probes(path)
            if not probes:
                logger.warning("fpr_guard_file_empty", guard=name, path=str(path))
                return None
            guard_sets[name] = probes

        combined_probes = 0
        combined_blocks = 0
        per_set: dict[str, Any] = {}
        for name, probes in guard_sets.items():
            probe_embs = self._engine.embed_batch(probes)
            report = blocking_report(
                probes, probe_embs, corpus_vecs, block_threshold, ESCALATE_FLOOR
            )
            report["fpr_percent"] = round(100.0 * report["block_count"] / report["probes"], 2)
            report["escalate_percent"] = round(
                100.0 * report["escalate_count"] / report["probes"], 2
            )
            per_set[name] = report
            combined_probes += report["probes"]
            combined_blocks += report["block_count"]

        guarded = round(100.0 * combined_blocks / combined_probes, 2) if combined_probes else 0.0
        return {
            "slo_percent": self.settings.semantic_fpr_slo,
            "enforced": self.settings.semantic_fpr_slo_enforce,
            "block_threshold": block_threshold,
            "guarded_fpr_percent": guarded,
            "slo_met": guarded <= self.settings.semantic_fpr_slo,
            "benign": per_set["benign"],
            "hard_negatives": per_set["hard_negatives"],
        }

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

        # NG-6: effective BLOCK threshold for this scan (per-tenant override
        # when the pipeline injected one, global setting otherwise).
        block_threshold = self._resolve_block_threshold(context)

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
        # Compute the search threshold: use the lower of the effective
        # similarity threshold and ESCALATE_FLOOR, so we catch ambiguous
        # matches (0.60-0.74) that hybrid scoring + judge need to evaluate.
        threshold = block_threshold
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

        # Convert matches to findings
        findings: list[Finding] = []
        max_similarity = 0.0

        for i, match in enumerate(matches):
            sim = match["similarity"]
            if sim > max_similarity:
                max_similarity = sim

            category = self._map_category(match.get("category", "T-PI-D"))
            severity = self._map_severity(match.get("severity", "medium"))
            verdict = self._similarity_to_verdict(sim, block_threshold)
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

        # Overall verdict from the full-text pass (pre-windowing).
        overall_verdict = self._findings_to_verdict(findings)

        # NG-4: overflow-resistant windowed pass for long inputs. Prompt
        # Overflow fragments a malicious instruction across an overlong
        # prompt; the full-text embedding gets diluted below threshold while
        # the LLM reads the fragments together. The windowed pass restores
        # the per-window evidence and aggregates it with the contiguity
        # gate. Skipped when the full-text pass already BLOCKed (nothing new
        # to learn at extra latency) and for short inputs (single window).
        if (
            self.settings.semantic_overflow_detection
            and overall_verdict is not Verdict.BLOCK
            and len(text) > self.settings.semantic_overflow_window_chars
        ):
            try:
                overflow_findings = self._overflow_scan(text, block_threshold)
            except Exception as exc:
                logger.error("semantic_overflow_scan_failed", error=str(exc))
                return self._result(
                    Verdict.BLOCK,
                    [*findings, self._overflow_error_finding(str(exc))],
                    start,
                    error=f"Overflow scan failed: {exc!r}",
                )
            findings.extend(overflow_findings)

        if not findings:
            logger.debug("similarity_scanner_no_matches", threshold=threshold)
            return self._result(Verdict.ALLOW, [], start)

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
        """Extract text content from the request (F6: user-role turns only by
        default — the judge must not verdict the defender's own system prompt
        against itself; scan_all_roles opts into the full conversation)."""
        texts = request.input_texts()
        return " ".join(texts)

    def _resolve_block_threshold(self, context: dict[str, Any] | None) -> float:
        """NG-6: resolve the effective semantic BLOCK threshold for this scan.

        The pipeline injects ``semantic_block_threshold`` into the scan
        context when the request's tenant config carries an override (the
        tenant's FPR/sensitivity dial). Defense in depth: the value is
        re-validated HERE — anything malformed or out of the
        [ESCALATE_FLOOR, _TENANT_BLOCK_CEILING] bounds falls back to the
        global threshold with a loud log. Never silent, never below the
        ambiguous zone (a threshold under 0.60 would map matches the
        corpus search does not even surface).
        """
        override = context.get(_TENANT_BLOCK_THRESHOLD_KEY) if context else None
        if override is None:
            return self.settings.semantic_similarity_threshold
        try:
            value = float(override)
        except (TypeError, ValueError):
            logger.warning("tenant_block_threshold_invalid", value=repr(override))
            return self.settings.semantic_similarity_threshold
        if not (ESCALATE_FLOOR <= value <= _TENANT_BLOCK_CEILING):
            logger.warning(
                "tenant_block_threshold_out_of_bounds",
                value=value,
                floor=ESCALATE_FLOOR,
                ceiling=_TENANT_BLOCK_CEILING,
            )
            return self.settings.semantic_similarity_threshold
        return value

    def _similarity_to_verdict(
        self, similarity: float, block_threshold: float | None = None
    ) -> Verdict:
        """Map similarity score to verdict using config thresholds.

        Thresholds:
          >= block_threshold (default 0.75) → BLOCK (high confidence attack match)
          >= 0.60 → ESCALATE (ambiguous, needs hybrid + judge)
          < 0.60 → ALLOW (likely benign)

        ``block_threshold`` carries the per-tenant override resolved in
        scan() (NG-6); it defaults to the global setting.
        """
        if block_threshold is None:
            block_threshold = self.settings.semantic_similarity_threshold
        if similarity >= block_threshold:
            return Verdict.BLOCK
        if similarity >= ESCALATE_FLOOR:
            return Verdict.ESCALATE
        return Verdict.ALLOW

    def _overflow_scan(self, text: str, block_threshold: float) -> list[Finding]:
        """NG-4: windowed overflow-resistant pass over a long input.

        Embeds overlapping windows in a single batch call, searches the
        corpus per window, and aggregates with the contiguity gate:
        - a single window matching at/above the BLOCK floor emits the same
          finding class the main path would (dilution must not save an
          attack that would block if concentrated);
        - accumulated sub-floor risk across a contiguous run of windows
          (summed excess above the benign background) emits one ESCALATE
          finding — ambiguous-zone evidence for the judge, never a silent
          pass.

        Callers wrap this in try/except and fail closed (BLOCK) on any
        engine/corpus error, same contract as the main path.
        """
        window_chars = self.settings.semantic_overflow_window_chars
        windows = chunk_windows(text, window_chars, self.settings.semantic_overflow_max_windows)
        if len(windows) < 2:
            return []

        embeddings = self._engine.embed_batch(windows)
        threshold = block_threshold
        search_threshold = min(threshold, ESCALATE_FLOOR)

        per_window_max: list[float] = []
        findings: list[Finding] = []
        best_category: ThreatCategory | None = None
        best_match_text = ""
        best_match_sim = 0.0

        for w_idx, emb in enumerate(embeddings):
            window_matches = self._corpus.search(emb, threshold=search_threshold, top_k=3)
            w_max = max((m["similarity"] for m in window_matches), default=0.0)
            per_window_max.append(w_max)

            for m in window_matches:
                # Concentrated match: a single window at/above the BLOCK
                # floor gets the same verdict treatment as the main path —
                # fragment dilution across the full text must not save it.
                if m["similarity"] >= threshold and m["similarity"] == w_max:
                    category = self._map_category(m.get("category", "T-PI-D"))
                    findings.append(
                        Finding(
                            category=category,
                            severity=self._map_severity(m.get("severity", "medium")),
                            verdict=Verdict.BLOCK,
                            confidence=m["similarity"],
                            layer=self.layer,
                            rule_id=f"{_SEMANTIC_RULE_PREFIX}-W-{w_idx:03d}",
                            description=(
                                "Semantic match in analysis window "
                                f"{w_idx + 1}/{len(windows)} "
                                f"({m['similarity']:.2f}): {m.get('text', '')[:80]}"
                            ),
                            evidence=(
                                f"category={m.get('category', '?')} "
                                f"source={m.get('source', '?')} window={w_idx + 1}"
                            ),
                            mitigation=(
                                "Review prompt for concentrated attack content in a long input"
                            ),
                            metadata={
                                "similarity": m["similarity"],
                                "window_index": w_idx,
                                "matched_category": m.get("category"),
                                "matched_source": m.get("source"),
                                "overflow": True,
                            },
                        )
                    )

            # Representative category/text for the gate finding: the strongest
            # window match seen so far (deterministic tie-break: first seen).
            if window_matches and window_matches[0]["similarity"] > best_match_sim:
                best_match_sim = window_matches[0]["similarity"]
                best_category = self._map_category(window_matches[0].get("category", "T-PI-D"))
                best_match_text = str(window_matches[0].get("text", ""))[:80]

        gate = contiguity_gate(
            per_window_max,
            self.settings.semantic_overflow_benign_threshold,
            self.settings.semantic_overflow_decision_threshold,
            self.settings.semantic_overflow_min_run,
        )
        if gate.flagged:
            findings.append(
                Finding(
                    category=best_category or ThreatCategory.PROMPT_INJECTION_DIRECT,
                    severity=Severity.HIGH,
                    verdict=Verdict.ESCALATE,
                    confidence=min(0.95, max(gate.max_run_sum, 0.30)),
                    layer=self.layer,
                    rule_id="SEM-OVERFLOW-001",
                    description=(
                        "Overflow-resistant aggregation flagged a contiguous run "
                        f"of {gate.max_run_len} windows with summed excess risk "
                        f"{gate.max_run_sum:.2f} above the benign background "
                        "(Prompt Overflow shape: sub-threshold fragments that "
                        "assemble downstream)."
                    ),
                    evidence=(
                        f"windows={len(per_window_max)} run_len={gate.max_run_len} "
                        f"run_sum={gate.max_run_sum:.3f}"
                        + (f" sample={best_match_text}" if best_match_text else "")
                    ),
                    mitigation=(
                        "Escalate for review/judge; do not reconstruct intent from "
                        "fragments in a single window alone"
                    ),
                    metadata={
                        "overflow": True,
                        "run_len": gate.max_run_len,
                        "run_sum": gate.max_run_sum,
                        "windows": len(per_window_max),
                    },
                )
            )
            logger.info(
                "semantic_overflow_flagged",
                windows=len(per_window_max),
                run_len=gate.max_run_len,
                run_sum=f"{gate.max_run_sum:.3f}",
            )

        return findings

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
    def _overflow_error_finding(error: str) -> Finding:
        """Finding for overflow-pass failure (fail-closed, NG-4)."""
        return Finding(
            category=ThreatCategory.SELF_ATTACK,
            severity=Severity.HIGH,
            verdict=Verdict.BLOCK,
            confidence=1.0,
            layer=ScanLayer.SEMANTIC,
            rule_id="SEM-OVERFLOW-ERR",
            description=f"Overflow windowed scan failed: {error}",
            mitigation="Verify embedding engine and attack corpus health",
        )

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
