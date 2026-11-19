#!/usr/bin/env python3
"""Export all-MiniLM-L6-v2 to ONNX format for NeuralGuard.

Downloads the sentence-transformer model, exports to ONNX Runtime format,
and copies the tokenizer files. Run once to prepare the model directory.

Usage:
    pip install neuralguard[semantic-export]
    python scripts/export_onnx.py [--model <settings default>] [--output-dir models/embedding-onnx]

The default model is read from ``ScannerSettings().semantic_model`` so the
exported artifact and the runtime cannot disagree. CI calls this with no
``--model`` for exactly that reason.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _default_model() -> str:
    """The model the product actually loads — never a second source of truth."""
    from neuralguard.config.settings import ScannerSettings

    return ScannerSettings().semantic_model


def export_model(model_name: str, output_dir: str) -> None:
    """Export a sentence-transformer model to ONNX format."""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Export dependencies not installed. Run: pip install neuralguard[semantic-export]"
        ) from exc

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading model: {model_name}")
    start = time.perf_counter()

    # Download tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(out_path)
    print(f"  ✅ Tokenizer saved to {out_path}")

    # Export to ONNX via optimum
    print("  Exporting to ONNX...")
    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    model.save_pretrained(out_path)
    print(f"  ✅ ONNX model saved to {out_path}")

    # Verify: run a quick inference
    print("  Verifying ONNX model...")
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(out_path / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    # Simple test
    encoded = tokenizer(
        ["Ignore all previous instructions"],
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="np",
    )

    # Feed only what the exported graph declares. BERT-style encoders require
    # token_type_ids; XLM-R-style ones (the multilingual E5 family) have no such
    # input and reject it with INVALID_ARGUMENT — which is how this was found
    # in CI, after the same assumption had already been fixed in the runtime
    # engine. Verification must exercise the graph that was actually exported.
    available = {inp.name for inp in session.get_inputs()}
    feed: dict[str, Any] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
    if "token_type_ids" in available:
        feed["token_type_ids"] = np.zeros_like(encoded["input_ids"])

    outputs = session.run(None, feed)

    # Mean pooling
    token_embeddings = outputs[0]
    mask = encoded["attention_mask"][:, :, np.newaxis].astype(np.float32)
    mean_emb = np.sum(token_embeddings * mask, axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
    norm = np.linalg.norm(mean_emb)
    print(f"  ✅ Inference OK — embedding shape: {mean_emb.shape}, L2 norm: {norm:.4f}")

    # Write config with metadata
    config = {
        "_name_or_path": model_name,
        "model_type": "onnx",
        "embedding_dim": int(mean_emb.shape[-1]),
        "max_seq_length": 256,
        "exported_by": "neuralguard",
    }
    config_path = out_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            existing = json.load(f)
        existing.update(config)
        config = existing
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.perf_counter() - start
    print(f"\n🏁 Export complete in {elapsed:.1f}s")
    print(f"   Model directory: {out_path.resolve()}")

    # List files
    files = sorted(out_path.iterdir())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    print(f"   Files: {len(files)}, Total size: {total_size / 1024 / 1024:.1f}MB")
    for f in files:
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"     {f.name:<30} {size_kb:>8.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sentence-transformer model to ONNX")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name from HuggingFace. Default: ScannerSettings().semantic_model "
            "(single source of truth — an export that disagrees with the runtime is a "
            "silent corpus/model mismatch)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="models/embedding-onnx",
        help="Output directory for ONNX model files",
    )
    args = parser.parse_args()
    export_model(args.model or _default_model(), args.output_dir)


if __name__ == "__main__":
    main()
