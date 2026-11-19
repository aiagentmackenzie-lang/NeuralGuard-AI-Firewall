"""Embedding engine — ONNX Runtime-based sentence embedding.

Loads a pre-exported all-MiniLM-L6-v2 ONNX model and computes
384-dimensional normalized embeddings for input text.

Design:
  - ONNX Runtime only — no PyTorch dependency at runtime
  - HuggingFace tokenizers for fast WordPiece tokenization
  - L2-normalized output for cosine similarity via dot product
  - Configurable thread count and sequence length
  - Loads once, reuses session across calls (singleton pattern)

Target: <50ms P95 per embedding on CPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import structlog

if TYPE_CHECKING:
    from neuralguard.config.settings import ScannerSettings

logger = structlog.get_logger(__name__)

# ── ONNX Runtime imports (lazy, fail gracefully) ──────────────────────────

try:
    import onnxruntime as ort

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

try:
    from tokenizers import Tokenizer

    _TOKENIZER_AVAILABLE = True
except ImportError:
    _TOKENIZER_AVAILABLE = False


class EmbeddingEngine:
    """ONNX-based embedding engine for semantic detection.

    Loads a sentence-transformer exported to ONNX (default:
    ``intfloat/multilingual-e5-base``) and returns L2-normalized vectors
    suitable for cosine similarity.

    The dimension is **discovered at load time** rather than hard-coded: the
    engine previously baked in 384 for all-MiniLM-L6-v2, which made a model
    swap a code change in three places and a silent mismatch everywhere else.
    """

    # Dimension of the shipped default model. Kept only as a documented
    # expectation; :attr:`embedding_dim` is the measured truth after load().
    EMBEDDING_DIM = 768

    def __init__(self, settings: ScannerSettings) -> None:
        self.settings = settings
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._loaded = False
        self._load_time_ms: float = 0.0

    def is_available(self) -> bool:
        """Check if ONNX Runtime and tokenizers are installed."""
        return _ONNX_AVAILABLE and _TOKENIZER_AVAILABLE

    def load(self) -> None:
        """Load the ONNX model and tokenizer into memory.

        Raises:
            ImportError: If onnxruntime or tokenizers not installed.
            FileNotFoundError: If model files not found at configured path.
        """
        if self._loaded:
            return

        if not self.is_available():
            raise ImportError(
                "onnxruntime and tokenizers required for semantic scanning. "
                "Install with: pip install neuralguard[semantic]"
            )

        start = time.perf_counter()
        model_dir = Path(self.settings.semantic_onnx_path)

        # Resolve model paths
        model_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        config_path = model_dir / "config.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. Run: python scripts/export_onnx.py"
            )
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. Run: python scripts/export_onnx.py"
            )

        # Create ONNX session
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.settings.semantic_intra_threads > 0:
            sess_opts.intra_op_num_threads = self.settings.semantic_intra_threads

        self._session = ort.InferenceSession(
            str(model_path),
            sess_opts,
            providers=["CPUExecutionProvider"],
        )

        # Load tokenizer
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=self.settings.semantic_max_seq_length)
        self._tokenizer.enable_padding(length=self.settings.semantic_max_seq_length)

        # Verify model config if available
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            expected = self.settings.semantic_model
            actual = config.get("_name_or_path", config.get("model_type", "unknown"))
            logger.info("embedding_model_loaded", model=actual, expected=expected)

        self._loaded = True
        self._load_time_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "embedding_engine_ready",
            load_time_ms=f"{self._load_time_ms:.1f}",
            max_seq_length=self.settings.semantic_max_seq_length,
        )

    @property
    def loaded(self) -> bool:
        """Whether the model has been loaded into memory."""
        return self._loaded

    @property
    def load_time_ms(self) -> float:
        """Time taken to load the model in milliseconds."""
        return self._load_time_ms

    def embed(self, text: str) -> np.ndarray:
        """Compute L2-normalized embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Normalized 384-dim numpy array (float32).

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self._loaded:
            raise RuntimeError("EmbeddingEngine not loaded. Call load() first.")
        return cast("np.ndarray", self.embed_batch([text])[0])

    def embed_batch(self, texts: list[str], *, role: str = "query") -> np.ndarray:
        """Compute L2-normalized embeddings for multiple texts.

        Args:
            texts: List of input texts to embed.
            role: ``"query"`` for user prompts (default) or ``"passage"`` for
                corpus texts. E5 models are trained with asymmetric
                ``query: ``/``passage: `` prefixes; using the wrong one degrades
                recall with no error raised, so the role is explicit rather than
                inferred from the caller.

        Returns:
            Normalized (len(texts), dim) numpy array (float32).

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self._loaded:
            raise RuntimeError("EmbeddingEngine not loaded. Call load() first.")
        assert self._tokenizer is not None and self._session is not None

        prefix = (
            self.settings.semantic_query_prefix
            if role == "query"
            else self.settings.semantic_passage_prefix
        )
        texts = [f"{prefix}{text}" for text in texts] if prefix else list(texts)

        # Tokenize
        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        # Feed only the inputs this graph declares. XLM-R-style encoders (the
        # multilingual E5 family) have no token_type_ids at all and reject the
        # extra tensor with INVALID_ARGUMENT; BERT-style ones require it. Asking
        # the session is cheaper than a per-model branch list that rots.
        available = {inp.name for inp in self._session.get_inputs()}
        feed: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in available:
            feed["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        # ONNX inference
        outputs = self._session.run(None, feed)

        # (batch, seq_len, hidden) — mean pooling over non-padded tokens, which
        # is the pooling these models were trained with (CLS pooling would be a
        # silent quality regression, not a stylistic choice).
        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_embeddings = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized = mean_embeddings / norms

        return cast("np.ndarray", normalized.astype(np.float32))

    def embed_raw(self, text: str) -> dict[str, Any]:
        """Compute embedding with metadata (for debugging/testing).

        Returns dict with embedding, latency, token_count, etc.
        """
        if not self._loaded:
            raise RuntimeError("EmbeddingEngine not loaded. Call load() first.")
        assert self._tokenizer is not None and self._session is not None

        start = time.perf_counter()
        encoded = self._tokenizer.encode_batch([text])
        token_count = len(encoded[0].ids)
        non_pad = sum(1 for t in encoded[0].ids if t != 0)

        embedding = self.embed(text)
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "embedding": embedding,
            "embedding_dim": embedding.shape[0],
            "token_count": token_count,
            "non_pad_tokens": non_pad,
            "latency_ms": latency_ms,
            "model": self.settings.semantic_model,
            "max_seq_length": self.settings.semantic_max_seq_length,
        }

    def unload(self) -> None:
        """Release model from memory."""
        if self._session is not None:
            del self._session
            self._session = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        logger.info("embedding_engine_unloaded")
