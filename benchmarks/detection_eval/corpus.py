"""Corpus loading for the detection-capability evaluation (P0, 2026-11-18).

One place that knows where every labelled item lives, so the evaluation harness
and the i18n sweep cannot drift apart. The corpora are **deliberately
heterogeneous in provenance** and each item carries it:

* ``corpus_multilingual_attacks.jsonl`` — 87 machine-authored attacks across
  10 languages across 4 families plus 7 language-neutral agentic/obfuscation items.
  Not human-reviewed. Directional evidence, not authority.
* ``benchmarks/ng_vs_ns/attack_corpus.jsonl`` — the existing 27-item English
  regression corpus that the A1 gate scores (kept, not duplicated).
* ``corpus_multilingual_benign.jsonl`` — 76 everyday benign sentences across
  the 10 languages (the i18n sweep's corpus, imported so there is one copy).
* ``benchmarks/ng_vs_ns/benign_corpus.jsonl`` + ``corpus/benign_hard_negatives.jsonl``
  — the 95 English benign probes behind the published FPR SLO, labelled
  ``hard_negative`` so the report can separate "everyday" from "adversarially
  benign" instead of blending them into one flattering number.

Language is a first-class field on every item: a detector that is 98 % accurate
in English and blind in Vietnamese must not be able to report "98 %".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parent

ATTACK_FILES: tuple[Path, ...] = (
    _EVAL_DIR / "corpus_multilingual_attacks.jsonl",
    _REPO / "benchmarks" / "ng_vs_ns" / "attack_corpus.jsonl",
)

BENIGN_FILES: tuple[Path, ...] = (
    _EVAL_DIR / "corpus_multilingual_benign.jsonl",
    _REPO / "benchmarks" / "ng_vs_ns" / "benign_corpus.jsonl",
    _REPO / "corpus" / "benign_hard_negatives.jsonl",
)

CONTROL_FILE = _EVAL_DIR / "corpus_controls.jsonl"

# The 10 UI languages NeuralGuard claims to detect in. A report that omits one
# is a report about a smaller product.
LANGUAGES: tuple[str, ...] = ("PT", "ES", "FR", "DE", "ZH", "JA", "KO", "RU", "AR", "VI", "EN")

# Regression-corpus family labels are prefixed (T-JB); strip it so a family
# means the same thing whichever file the item came from.
_FAMILY_ALIASES = {
    "T-PI-D": "PI-D",
    "T-PI-I": "PI-I",
    "T-JB": "JB",
    "T-EXT": "EXT",
    "T-EXF": "EXF",
    "T-TOOL": "TOOL",
    "T-DOS": "DOS",
}


@dataclass(frozen=True, slots=True)
class Attack:
    id: str
    language: str
    family: str
    prompt: str
    provenance: str


@dataclass(frozen=True, slots=True)
class Benign:
    id: str
    language: str
    kind: str  # "everyday" | "hard_negative"
    prompt: str
    provenance: str


@dataclass(frozen=True, slots=True)
class Control:
    id: str
    language: str
    prompt: str
    provenance: str


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():  # pragma: no cover - a missing corpus is its own failure
        raise FileNotFoundError(f"expected corpus at {path}")
    items: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def _dedupe(items: list[Attack] | list[Benign]) -> list[Attack] | list[Benign]:
    """Drop exact duplicate prompts, keeping the first (stable, ordered)."""
    seen: set[str] = set()
    out: list[Attack | Benign] = []
    for item in items:
        key = item.prompt.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def load_attacks() -> list[Attack]:
    attacks: list[Attack] = []
    for path in ATTACK_FILES:
        for row in _read_jsonl(path):
            raw_family = str(row.get("family") or row.get("block_family") or "UNKNOWN")
            attacks.append(
                Attack(
                    id=str(row["id"]),
                    language=str(row.get("language", "EN")),
                    family=_FAMILY_ALIASES.get(raw_family, raw_family),
                    prompt=str(row["prompt"]),
                    provenance=str(row.get("provenance", f"{path.name} (existing corpus)")),
                )
            )
    return _dedupe(attacks)  # type: ignore[return-value]


def load_benign() -> list[Benign]:
    benign: list[Benign] = []
    for path in BENIGN_FILES:
        for row in _read_jsonl(path):
            kind = str(row.get("kind") or "everyday")
            if path.name == "benign_hard_negatives.jsonl":
                kind = "hard_negative"
            benign.append(
                Benign(
                    id=str(row["id"]),
                    language=str(row.get("language", "EN")),
                    kind=kind,
                    prompt=str(row["prompt"]),
                    provenance=str(row.get("provenance", f"{path.name} (existing corpus)")),
                )
            )
    return _dedupe(benign)  # type: ignore[return-value]


def load_controls() -> list[Control]:
    return [
        Control(
            id=str(row["id"]),
            language=str(row["language"]),
            prompt=str(row["prompt"]),
            provenance=str(row.get("provenance", CONTROL_FILE.name)),
        )
        for row in _read_jsonl(CONTROL_FILE)
    ]
