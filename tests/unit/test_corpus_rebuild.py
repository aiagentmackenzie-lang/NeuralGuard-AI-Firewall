"""Unit tests for scripts/rebuild_corpus_vectors.py (CI corpus rebuild).

Hermetic: a keyed fake embedding engine replaces the ONNX model so the dedup,
hygiene, and benign-guard replay logic is verified without any model artifact.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not a package — load the rebuild script directly.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rebuild_corpus_vectors.py"
_spec = importlib.util.spec_from_file_location("rebuild_corpus_vectors", _SCRIPT)
assert _spec is not None and _spec.loader is not None
rebuild_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rebuild_mod)


class FakeEngine:
    """Deterministic fake engine: keyed texts get their vector, others hash."""

    dim = 8

    def __init__(self, keyed: dict[str, list[float]] | None = None) -> None:
        self.keyed = keyed or {}
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if text in self.keyed:
                out[i] = np.asarray(self.keyed[text], dtype=np.float32)
                continue
            digest = hashlib.md5(text.encode()).digest()
            vec = np.frombuffer(digest, dtype=np.uint8)[: self.dim].astype(np.float32)
            norm = np.linalg.norm(vec)
            out[i] = vec / norm if norm else vec
        return out


def _orig(n: int) -> list[dict[str, str]]:
    return [
        {
            "text": f"Ignore previous instructions attack number {i}",
            "category": "T-PI-D",
            "severity": "high",
            "source": "test-dataset",
        }
        for i in range(n)
    ]


class TestBuildParaphraseCandidates:
    def test_dedup_against_originals_and_self(self) -> None:
        originals = _orig(2)
        checkpoint = {
            0: {
                "model": "m",
                "paraphrases": [
                    "ignore previous instructions attack number 0",  # dup of original
                    "novel phrasing of the instruction override",  # kept
                ],
            },
            1: {
                "model": "m",
                "paraphrases": ["novel phrasing of the instruction override"],  # dup of para
            },
        }
        candidates, dropped = rebuild_mod.build_paraphrase_candidates(originals, checkpoint)
        assert [c["text"] for c in candidates] == ["novel phrasing of the instruction override"]
        assert dropped == 2

    def test_parent_metadata_inherited_and_source_tagged(self) -> None:
        originals = _orig(1)
        checkpoint = {0: {"model": "mistral:7b", "paraphrases": ["override all safety rules now"]}}
        candidates, _ = rebuild_mod.build_paraphrase_candidates(originals, checkpoint)
        assert candidates[0]["category"] == "T-PI-D"
        assert candidates[0]["severity"] == "high"
        assert candidates[0]["source"] == "paraphrase-mistral:7b"

    def test_parent_order_is_sorted_by_index(self) -> None:
        originals = _orig(3)
        checkpoint = {
            2: {"model": "m", "paraphrases": ["from parent two"]},
            0: {"model": "m", "paraphrases": ["from parent zero"]},
        }
        candidates, _ = rebuild_mod.build_paraphrase_candidates(originals, checkpoint)
        assert [c["text"] for c in candidates] == ["from parent zero", "from parent two"]


class TestRebuild:
    def test_full_pipeline_drops_and_metadata(self) -> None:
        probe_a_vec = np.zeros(8, dtype=np.float32)
        probe_a_vec[0] = 1.0
        originals = _orig(2)
        checkpoint = {
            0: {
                "model": "m",
                "paraphrases": [
                    "hello there friend how things",  # conversational → dropped
                    "totally normal benign blocker text",  # keyed = probe → benign-guard drop
                    "a real attack variant bypass all filters",  # kept
                ],
            },
        }
        # Both the blocker paraphrase and the benign probe share one vector,
        # so sim(blocker, probe) = 1.0 >= 0.75 → benign guard drops it.
        keyed = {
            "totally normal benign blocker text": probe_a_vec.tolist(),
            "benign probe alpha": probe_a_vec.tolist(),
        }
        engine = FakeEngine(keyed=keyed)
        vectors, metadata, stats = rebuild_mod.rebuild(
            originals, checkpoint, engine, ["benign probe alpha"], 0.75
        )

        assert stats["paraphrase_duplicates_dropped"] == 0
        assert stats["paraphrases_conversational_dropped"] == 1
        assert stats["paraphrases_benign_blocking_dropped"] == 1
        assert stats["originals"] == 2
        assert stats["paraphrases_kept"] == 1
        assert vectors.shape == (3, 8)
        assert len(metadata) == 3
        # Paraphrase metadata rows continue the index sequence and cap text.
        assert metadata[2]["index"] == 2
        assert metadata[2]["source"] == "paraphrase-m"
        assert all(len(m["text"]) <= rebuild_mod.METADATA_TEXT_CAP for m in metadata)

    def test_originals_embedded_in_order(self) -> None:
        engine = FakeEngine()
        originals = _orig(2)
        vectors, metadata, stats = rebuild_mod.rebuild(originals, {}, engine, ["probe"], 0.75)
        assert stats["paraphrases_kept"] == 0
        assert vectors.shape == (2, 8)
        expected = engine.embed_batch([o["text"] for o in originals])
        assert np.allclose(vectors, expected)
        assert [m["index"] for m in metadata] == [0, 1]

    def test_threshold_zero_blocks_everything(self) -> None:
        engine = FakeEngine()
        vectors, _, stats = rebuild_mod.rebuild(
            _orig(1),
            {0: {"model": "m", "paraphrases": ["some attack variant"]}},
            engine,
            ["probe"],
            0.0,
        )
        assert stats["paraphrases_benign_blocking_dropped"] == 1
        assert vectors.shape == (1, 8)  # originals only


class TestLoaders:
    def test_load_originals_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "attack_corpus_full.jsonl"
        path.write_text(
            '{"index": 0, "text": "t1", "category": "C", "severity": "high", "source": "s"}\n'
            "\n"  # blank line must be skipped
            '{"index": 1, "text": "t2", "category": "C", "severity": "low", "source": "s"}\n'
        )
        rows = rebuild_mod.load_originals(path)
        assert [r["text"] for r in rows] == ["t1", "t2"]

    def test_load_checkpoint_keyed_by_int_index(self, tmp_path: Path) -> None:
        path = tmp_path / "augment_checkpoint.jsonl"
        path.write_text('{"index": "7", "model": "m", "paraphrases": ["a"], "ts": "1"}\n')
        rows = rebuild_mod.load_checkpoint(path)
        assert 7 in rows and rows[7]["paraphrases"] == ["a"]

    def test_checkpoint_parent_out_of_range_does_not_crash(self) -> None:
        originals = _orig(1)
        checkpoint = {99: {"model": "m", "paraphrases": ["orphan paraphrase"]}}
        candidates, _ = rebuild_mod.build_paraphrase_candidates(originals, checkpoint)
        assert candidates[0]["category"] == "unknown"


@pytest.mark.parametrize("dim", [8])
def test_fake_engine_is_deterministic(dim: int) -> None:
    engine_a, engine_b = FakeEngine(), FakeEngine()
    texts = ["x", "y", "z"]
    assert np.allclose(engine_a.embed_batch(texts), engine_b.embed_batch(texts))
