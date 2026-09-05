#!/usr/bin/env python3
"""Rebuild the semantic corpus artifacts deterministically from TRACKED sources.

The gitignored runtime artifacts (models/attack_vectors.npy +
models/attack_metadata.json) are produced by a two-stage pipeline:

  1. ``scripts/build_attack_corpus.py`` — dataset originals + embeddings.
  2. ``scripts/augment_attack_corpus.py`` (F12) — LLM paraphrases, checkpointed
     during generation, then finalized with hygiene + a benign guard.

Both stages consumed untracked inputs (HF dataset drift, local Ollama), so CI
cannot re-run them as-is. This script reproduces the FINAL corpus from the two
git-tracked sources that capture its end state:

  - ``corpus/attack_corpus_full.jsonl``   — the original rows (stage 1 output).
  - ``corpus/augment_checkpoint.jsonl``   — full paraphrase texts grouped by
    parent index (the F12 checkpoint the finalize step consumed).

Replaying dedup → conversational hygiene → embedding → benign guard over those
sources is deterministic given the embedding model. HONEST NOTE (measured
2026-09-05): the rebuild yields 6,503 vectors vs the shipped 7,623 — the
augment script's checkpoint OVERWRITES the row per parent index on every
pass, so paraphrases from earlier passes for re-attempted parents (~1,120)
are in the shipped artifact but not reconstructible from the final checkpoint.
The rebuild is therefore a faithful SUBSET, not an identical copy. The arbiter
of its fitness is the A1 regression gate: run with the rebuilt corpus it
PASSES (ASR 0.00% / FPR 0.00%, exit 0), so CI tests a corpus that is
detection-equivalent on the gated set. The shipped appliance artifact (7,623)
is built on the operator machine and stays gitignored.

Usage (CI order):

    uv run python scripts/export_onnx.py
    uv run python scripts/rebuild_corpus_vectors.py
    uv run pytest tests/ --cov=neuralguard ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

from neuralguard.config.settings import ScannerSettings
from neuralguard.semantic.hygiene import (
    drop_benign_blocking_vectors,
    drop_conversational_vectors,
    load_benign_guard_probes,
)

CORPUS_DIR = Path("corpus")
MODEL_DIR = Path("models")
BENIGN_GUARD_PATH = Path("benchmarks/ng_vs_ns/benign_corpus.jsonl")
METADATA_TEXT_CAP = 200  # same truncation build_attack_corpus.py applies


class EmbeddingEngineLike(Protocol):
    """Structural subset of neuralguard.semantic.embedding.EmbeddingEngine."""

    def load(self) -> None: ...

    def embed_batch(self, texts: list[str]) -> Any: ...  # (n, dim) float array


def load_originals(corpus_path: Path) -> list[dict[str, Any]]:
    """Load the tracked original rows (stage-1 output) in file order."""
    rows: list[dict[str, Any]] = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    "text": row["text"],
                    "category": row.get("category", "unknown"),
                    "severity": row.get("severity", "unknown"),
                    "source": row.get("source", "unknown"),
                }
            )
    return rows


def load_checkpoint(checkpoint_path: Path) -> dict[int, dict[str, Any]]:
    """Load the F12 checkpoint rows keyed by parent index (order preserved)."""
    rows: dict[int, dict[str, Any]] = {}
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["index"])] = row
    return rows


def build_paraphrase_candidates(
    originals: list[dict[str, Any]],
    checkpoint: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Replay finalize()'s dedup + ordering over checkpointed paraphrases.

    Candidates are ordered by parent index then paraphrase order — the exact
    order finalize() used. Returns ``(candidates, dropped_duplicate_count)``.
    """
    existing_texts = {o["text"].strip().lower() for o in originals}
    candidates: list[dict[str, Any]] = []
    dropped_dupes = 0
    for parent_idx in sorted(checkpoint):
        row = checkpoint[parent_idx]
        parent = originals[parent_idx] if parent_idx < len(originals) else {}
        for paraphrase in row.get("paraphrases", []):
            key = paraphrase.strip().lower()
            if key in existing_texts:
                dropped_dupes += 1
                continue
            existing_texts.add(key)
            candidates.append(
                {
                    "text": paraphrase,
                    "category": parent.get("category", "unknown"),
                    "severity": parent.get("severity", "unknown"),
                    "source": f"paraphrase-{row.get('model', 'unknown')}",
                }
            )
    return candidates, dropped_dupes


def rebuild(
    originals: list[dict[str, Any]],
    checkpoint: dict[int, dict[str, Any]],
    engine: EmbeddingEngineLike,
    benign_probe_texts: list[str],
    block_threshold: float,
) -> tuple[Any, list[dict[str, Any]], dict[str, int]]:
    """Produce (vectors, metadata, stats) from tracked sources.

    Mirrors ``build_attack_corpus.py`` (originals) + ``augment_attack_corpus.py
    finalize()`` (dedup → conversational drop → embed → benign guard → append).
    """
    import numpy as np

    stats: dict[str, int] = {}

    # 1. Originals: embed all, no re-hygiene (the tracked rows ARE the
    #    post-hygiene stage-1 output).
    orig_embs = engine.embed_batch([o["text"] for o in originals]).astype(np.float32)

    # 2. Paraphrase candidates: dedup against originals + each other.
    candidates, dropped_dupes = build_paraphrase_candidates(originals, checkpoint)
    stats["paraphrase_duplicates_dropped"] = dropped_dupes

    # 3. Conversational hygiene (same tight filter finalize() applied).
    candidates, dropped_conv = drop_conversational_vectors(candidates)
    stats["paraphrases_conversational_dropped"] = len(dropped_conv)

    # 4. Embed the surviving candidates (full texts).
    texts = [c["text"] for c in candidates]
    cand_embs = (
        engine.embed_batch(texts).astype(np.float32)
        if texts
        else np.zeros((0, orig_embs.shape[1]), np.float32)
    )

    # 5. Benign guard over the paraphrase candidates (parents were guarded at
    #    build time; finalize() guarded new rows only).
    probe_embs = engine.embed_batch(benign_probe_texts).astype(np.float32)
    kept, dropped_guard = drop_benign_blocking_vectors(
        candidates, cand_embs, probe_embs, block_threshold
    )
    stats["paraphrases_benign_blocking_dropped"] = len(dropped_guard)

    kept_texts = [c["text"] for c in kept]
    kept_embs = (
        engine.embed_batch(kept_texts).astype(np.float32)
        if kept_texts
        else np.zeros((0, orig_embs.shape[1]), np.float32)
    )

    vectors = np.vstack([orig_embs, kept_embs]).astype(np.float32)

    metadata = [
        {
            "index": i,
            "text": o["text"][:METADATA_TEXT_CAP],
            "category": o["category"],
            "severity": o["severity"],
            "source": o["source"],
        }
        for i, o in enumerate(originals)
    ] + [
        {
            "index": len(originals) + i,
            "text": c["text"][:METADATA_TEXT_CAP],
            "category": c["category"],
            "severity": c["severity"],
            "source": c["source"],
        }
        for i, c in enumerate(kept)
    ]

    stats["originals"] = len(originals)
    stats["paraphrases_kept"] = len(kept)
    stats["total"] = vectors.shape[0]
    return vectors, metadata, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild models/ corpus artifacts from tracked sources."
    )
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR), help="Tracked corpus sources dir.")
    parser.add_argument(
        "--output-dir", default=str(MODEL_DIR), help="Where to write .npy + metadata."
    )
    parser.add_argument(
        "--benign-guard",
        default=str(BENIGN_GUARD_PATH),
        help="Benign probe corpus for the benign guard.",
    )
    args = parser.parse_args()

    from neuralguard.semantic.embedding import EmbeddingEngine

    corpus_dir, out_dir = Path(args.corpus_dir), Path(args.output_dir)
    originals = load_originals(corpus_dir / "attack_corpus_full.jsonl")
    checkpoint = load_checkpoint(corpus_dir / "augment_checkpoint.jsonl")

    settings = ScannerSettings()
    engine: EmbeddingEngineLike = EmbeddingEngine(settings)
    engine.load()

    probes = load_benign_guard_probes(Path(args.benign_guard))
    vectors, metadata, stats = rebuild(
        originals, checkpoint, engine, probes, settings.semantic_similarity_threshold
    )

    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "attack_vectors.npy", vectors)
    with open(out_dir / "attack_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(
        f"rebuild: {stats['originals']} originals + {stats['paraphrases_kept']} "
        f"paraphrases = {stats['total']} vectors"
    )
    print(
        f"  hygiene: {stats['paraphrase_duplicates_dropped']} duplicates, "
        f"{stats['paraphrases_conversational_dropped']} conversational, "
        f"{stats['paraphrases_benign_blocking_dropped']} benign-blocking dropped"
    )
    print(
        f"  wrote {out_dir / 'attack_vectors.npy'} ({vectors.shape}) + "
        f"{out_dir / 'attack_metadata.json'}"
    )


if __name__ == "__main__":
    main()
