"""NG-5 mutation operators — the 16-operator character-level red-team suite.

Companion to the A1 corpus harness (Sprint A). The HF robustness study
(2026-05, kalyan-ks) showed guard models lose up to ~45% Unsafe-ASR under
character-level mutation AND over-block benign text by up to ~25%: guards
rely on surface patterns, and perturbation breaks tokenization into
subwords the guard never learned. NG-5 makes NeuralGuard's coverage of
that surface *measurable and regression-gated* — both directions
(Unsafe-ASR: mutated attacks allowed through; Safe-ASR: mutated benign
wrongly caught).

Properties of every operator:
- deterministic (no randomness — a CI run is reproducible byte-for-byte),
- pure text-in/text-out, bounded output growth (no amplification bombs),
- applied INDEPENDENTLY to the original sample (never stacked).

What the pipeline already handles is part of the measurement, not an
assumption: the structural layer NFKD-normalizes (folds fullwidth, strips
combining marks' compatibility forms), strips zero-width characters, and
detects base64/hex/ROT13; the pattern layer has ENC-001 (homoglyph +
instruction context) and ENC-003 (leetspeak). The per-operator report in
the mutation harness shows exactly which operators the deterministic
layers already absorb and which are gaps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["MUTATION_OPERATORS", "MutationOperator", "apply_mutation"]


class MutationOperator:
    """One named, deterministic text mutation (str -> str)."""

    __slots__ = ("description", "fn", "name")

    def __init__(self, name: str, description: str, fn: Callable[[str], str]) -> None:
        self.name = name
        self.description = description
        self.fn = fn

    def __call__(self, text: str) -> str:
        return self.fn(text)


# ── Translation tables ───────────────────────────────────────────────────
#
# These are written as PAIR-EXPLICIT dicts (not the two-string maketrans
# form): the two-string form silently accepts a misaligned pair list, and it
# shipped one — the Cyrillic uppercase rows were rotated by one (P→U+041E,
# O→U+0425, X→U+0420, C→U+0423, Y→U+0421), plus two "Greek" entries that
# were actually Cyrillic and one that is not a confusable at all (n→U+03BD,
# which is a lookalike of v, not n).
# A substitution that is not a real lookalike is unrecoverable information
# loss — it measures noise, not a homoglyph attack. The invariant is pinned
# by `test_operator_substitutions_round_trip_through_the_fold`.

# Latin → Cyrillic lookalikes (Unicode confusables.txt, the classic set).
_CYRILLIC_PAIRS = {
    "a": "\u0430",
    "e": "\u0435",
    "o": "\u043e",
    "x": "\u0445",
    "p": "\u0440",
    "y": "\u0443",
    "c": "\u0441",
    "i": "\u0456",
    "j": "\u0458",
    "s": "\u0455",
    "A": "\u0410",
    "E": "\u0415",
    "P": "\u0420",
    "O": "\u041e",
    "X": "\u0425",
    "C": "\u0421",
    "Y": "\u0423",
}
_CYRILLIC_MAP = str.maketrans(_CYRILLIC_PAIRS)

# Latin → Greek lookalikes (Unicode confusables.txt). Latin v→U+03BD (nu),
# and Latin c→U+03F2 (lunate sigma) — the previous table used Cyrillic
# U+0441 for c.
_GREEK_PAIRS = {
    "a": "\u03b1",
    "e": "\u03b5",
    "i": "\u03b9",
    "o": "\u03bf",
    "p": "\u03c1",
    "c": "\u03f2",  # U+03F2 lunate sigma
    "x": "\u03c7",
    "u": "\u03c5",
    "v": "\u03bd",  # U+03BD nu (looks like v)
    "A": "\u0391",
    "B": "\u0392",
    "E": "\u0395",
    "Z": "\u0396",
    "H": "\u0397",
    "I": "\u0399",
    "K": "\u039a",
    "M": "\u039c",
    "N": "\u039d",
    "O": "\u039f",
    "P": "\u03a1",
    "T": "\u03a4",
    "Y": "\u03a5",
    "X": "\u03a7",
}
_GREEK_MAP = str.maketrans(_GREEK_PAIRS)

# Leetspeak: a→4 e→3 i→1 o→0 s→5 t→7 l→1 g→9.
_LEET_MAP = str.maketrans("aeiostlgAEIOSTLG", "4310571943105719")

_FULLWIDTH_OFFSET = 0xFEE0  # ASCII 0x21-0x7E → fullwidth forms (NFKD-foldable)

# Common combining acute on vowels (NFKD decomposes these — structural absorbs).
_COMBINING_ACUTE = "\u0301"
_VOWELS = "aeiouAEIOU"

_EMOJI = "🙂"
_SOFT_HYPHEN = "\u00ad"
_NBSP = "\u00a0"


def _homoglyph_cyrillic(text: str) -> str:
    return text.translate(_CYRILLIC_MAP)


def _homoglyph_greek(text: str) -> str:
    return text.translate(_GREEK_MAP)


def _fullwidth(text: str) -> str:
    return "".join(
        chr(ord(ch) + _FULLWIDTH_OFFSET) if 0x21 <= ord(ch) <= 0x7E else ch for ch in text
    )


def _leetspeak(text: str) -> str:
    return text.translate(_LEET_MAP)


def _diacritics(text: str) -> str:
    return "".join(ch + _COMBINING_ACUTE if ch in _VOWELS else ch for ch in text)


def _zero_width(text: str) -> str:
    return "\u200b".join(text)


def _token_splitting(text: str) -> str:
    """Split every word >3 letters into 3-char chunks ("ign ore")."""
    out: list[str] = []
    for word in text.split(" "):
        if len(word) > 3 and word.isalpha():
            chunks = [word[i : i + 3] for i in range(0, len(word), 3)]
            out.append(" ".join(chunks))
        else:
            out.append(word)
    return " ".join(out)


def _alternating_case(text: str) -> str:
    flip = True
    out: list[str] = []
    for ch in text:
        if ch.isalpha():
            out.append(ch.upper() if flip else ch.lower())
            flip = not flip
        else:
            out.append(ch)
    return "".join(out)


def _dot_interleave(text: str) -> str:
    return ".".join(text)


def _word_reversal(text: str) -> str:
    return " ".join(word[::-1] for word in text.split(" "))


def _char_duplication(text: str) -> str:
    return "".join(ch + ch if ch.isalpha() else ch for ch in text)


def _whitespace_swap(text: str) -> str:
    """Spaces → non-breaking space (whitespace-normalizing layers must fold)."""
    return text.replace(" ", _NBSP)


def _emoji_interleave(text: str) -> str:
    return _EMOJI.join(text.split(" "))


def _soft_hyphen(text: str) -> str:
    return _SOFT_HYPHEN.join(text)


def _vowel_stretch(text: str) -> str:
    return "".join(ch * 2 if ch in _VOWELS else ch for ch in text)


def _accent_homoglyph_mix(text: str) -> str:
    """Cyrillic homoglyphs + NFKD-strippable accents on vowels — the combo
    shape (detection must survive one normalizer)."""
    return _homoglyph_cyrillic(_diacritics(text))


MUTATION_OPERATORS: list[MutationOperator] = [
    MutationOperator("homoglyph_cyrillic", "Latin → Cyrillic lookalikes", _homoglyph_cyrillic),
    MutationOperator("homoglyph_greek", "Latin → Greek lookalikes", _homoglyph_greek),
    MutationOperator("fullwidth", "ASCII → fullwidth forms (NFKD-foldable)", _fullwidth),
    MutationOperator("leetspeak", "a→4 e→3 i→1 o→0 s→5 t→7 l→1 g→9", _leetspeak),
    MutationOperator("diacritics", "combining acute on vowels (NFKD-strippable)", _diacritics),
    MutationOperator("zero_width", "U+200B between every char", _zero_width),
    MutationOperator("token_splitting", "3-char chunks inside words", _token_splitting),
    MutationOperator("alternating_case", "AlTeRnAtInG capitalization", _alternating_case),
    MutationOperator("dot_interleave", "dots between every character", _dot_interleave),
    MutationOperator("word_reversal", "each word's characters reversed", _word_reversal),
    MutationOperator("char_duplication", "every letter doubled", _char_duplication),
    MutationOperator("whitespace_swap", "spaces → NBSP", _whitespace_swap),
    MutationOperator("emoji_interleave", "emoji between words", _emoji_interleave),
    MutationOperator("soft_hyphen", "U+00AD between every char", _soft_hyphen),
    MutationOperator("vowel_stretch", "vowels doubled", _vowel_stretch),
    # The combo operator closes the 16-slot set — the HF study's point is
    # that PERTURBATION COMBINATIONS, not single tricks, are what breaks
    # guards. It stays in the same table so the per-operator report is one
    # flat 16-row view.
    MutationOperator(
        "accent_homoglyph_mix", "Cyrillic homoglyph + combining accents", _accent_homoglyph_mix
    ),
]


def apply_mutation(text: str, operator: MutationOperator) -> str:
    """Apply one mutation with sanity bounds: output must be non-empty and
    never more than 3x the input length (amplification guard)."""
    mutated = operator.fn(text)
    if not mutated or len(mutated) > 3 * max(1, len(text)):
        return text
    return mutated
