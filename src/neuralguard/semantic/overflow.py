"""Overflow-resistant window aggregation (NG-4).

Prompt Overflow (arXiv:2605.23196, 2026) fragments a malicious instruction
into low-density pieces interleaved with benign filler across an overlong
prompt: no single inspection window holds enough correlated evidence to
cross the detector threshold, while the downstream LLM reads the full
context and reconstructs the intent. The paper reports ~100% bypass of
Llama Prompt Guard 2 (86M & 22M) and IBM Granite Guardian 125M with an
interleaved layout — and shows the two standard mitigations (sliding-window
inspection, max-pooling aggregation) both fail.

The paper's validated defense, implemented here: **stateful,
contiguity-gated aggregation**. Instead of taking the single worst window
score (max-pooling), calibrate a benign background threshold ``θ_b``,
compute each window's excess risk ``e_i = max(0, s_i - θ_b)``, and flag
when a maximal contiguous run of ≥ ``min_run`` windows accumulates summed
excess risk ≥ the decision threshold. In the paper's validation this
recovered a bypass case that max-pooling scored 0.32 → aggregate 0.63
(threshold 0.5).

This module is PURE (no I/O, no model calls) so the gate itself is
unit-testable in isolation; the SimilarityScanner wires it to real
embeddings (see SimilarityScanner._overflow_scan).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ContiguityResult", "chunk_windows", "contiguity_gate"]


@dataclass(frozen=True)
class ContiguityResult:
    """Outcome of the contiguity-gated excess-risk aggregation.

    Attributes:
        flagged: True when a contiguous run of ≥ min_run windows with
            nonzero excess risk accumulates ≥ decision_threshold total.
        max_run_sum: The largest summed excess risk found over any maximal
            contiguous run (0.0 when no window exceeds the background).
        max_run_len: Length of that best run (windows with excess > 0).
    """

    flagged: bool
    max_run_sum: float
    max_run_len: int


def chunk_windows(text: str, window_chars: int, max_windows: int) -> list[str]:
    """Split text into overlapping character windows, bounded by max_windows.

    Uses a 50% overlap stride by default so fragment boundaries can't hide
    on a cut (a fragment split across a non-overlapping boundary would be
    diluted in both windows). For very long inputs the stride widens so the
    whole text stays covered with at most ``max_windows`` windows — latency
    is bounded, coverage is not traded away.

    Args:
        text: The input text to window.
        window_chars: Window size in characters (typical semantic window).
        max_windows: Hard cap on the number of windows.

    Returns:
        List of window strings. Empty text → empty list. Text shorter than
        one window → the text itself as a single window.
    """
    if not text:
        return []
    if len(text) <= window_chars:
        return [text]

    stride = max(1, window_chars // 2)
    # Widen the stride when the default would exceed the window cap, so the
    # entire text remains covered end-to-end with ≤ max_windows windows.
    needed_stride = math.ceil((len(text) - window_chars) / max(1, max_windows - 1))
    stride = max(stride, needed_stride)

    windows: list[str] = []
    pos = 0
    while pos < len(text) and len(windows) < max_windows:
        windows.append(text[pos : pos + window_chars])
        if pos + window_chars >= len(text):
            break
        pos += stride
    return windows


def contiguity_gate(
    similarities: list[float],
    benign_threshold: float,
    decision_threshold: float,
    min_run: int = 2,
) -> ContiguityResult:
    """Flag accumulated sub-threshold risk across contiguous windows.

    Per window: excess risk ``e_i = max(0, s_i - benign_threshold)``.
    A *maximal contiguous run* is a stretch of consecutive windows each
    with excess > 0 (bounded by any window at/below background). The gate
    flags when such a run of length ≥ ``min_run`` accumulates summed
    excess ≥ ``decision_threshold``.

    Rationale (NG-4): max-pooling discards correlated sub-threshold
    evidence — exactly the Prompt Overflow shape. Summing excess risk over
    contiguous windows restores the evidence the fragments were designed
    to hide, while isolated single-window noise (one window slightly above
    background, neighbors clean) still does not flag: FPR stays governed
    by the run requirement + the sum threshold. The threshold comparison
    carries a 1e-9 epsilon so float noise (0.70-0.60 = 0.0999...) cannot
    silently flip a boundary case to fail-open.

    Args:
        similarities: Per-window corpus-similarity scores (0.0-1.0), in
            window order.
        benign_threshold: Benign background level ``θ_b``; excess is
            measured above this.
        decision_threshold: Minimum summed excess risk over a run to flag.
        min_run: Minimum number of consecutive above-background windows.

    Returns:
        ContiguityResult with the flag decision and the evidence numbers.
    """
    best_sum = 0.0
    best_len = 0
    run_sum = 0.0
    run_len = 0

    for sim in similarities:
        excess = max(0.0, sim - benign_threshold)
        if excess > 0.0:
            run_sum += excess
            run_len += 1
            if run_sum > best_sum:
                best_sum = run_sum
                best_len = run_len
        else:
            run_sum = 0.0
            run_len = 0

    flagged = best_len >= min_run and best_sum >= (decision_threshold - 1e-9)
    return ContiguityResult(flagged=flagged, max_run_sum=best_sum, max_run_len=best_len)
