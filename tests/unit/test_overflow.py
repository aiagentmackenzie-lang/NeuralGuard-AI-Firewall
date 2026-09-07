"""Tests for the NG-4 overflow module — pure windowing + contiguity gate.

Prompt Overflow (arXiv:2605.23196) fragments a malicious instruction into
low-density pieces interleaved with benign filler; max-pooling aggregation
discards the correlated sub-threshold evidence. The contiguity-gated
excess-risk aggregation implemented here is the paper's validated defense.
"""

from __future__ import annotations

import pytest

from neuralguard.semantic.overflow import ContiguityResult, chunk_windows, contiguity_gate


class TestChunkWindows:
    def test_short_text_single_window(self) -> None:
        assert chunk_windows("hello world", 400, 12) == ["hello world"]

    def test_empty_text_no_windows(self) -> None:
        assert chunk_windows("", 400, 12) == []

    def test_long_text_multi_window_with_overlap(self) -> None:
        text = "a" * 1000
        windows = chunk_windows(text, 400, 12)
        assert len(windows) > 1
        # Coverage proof: the union of windows spans the whole text —
        # the first window starts at 0 and the last ends at len(text).
        assert windows[0].startswith(text[0])
        assert text.endswith(windows[-1])

    def test_max_windows_respected_by_widening_stride(self) -> None:
        text = "x" * 10_000
        windows = chunk_windows(text, 400, 8)
        assert len(windows) <= 8
        # The last window must reach the end of the text (full coverage).
        assert text.endswith(windows[-1])

    def test_exact_window_size_single_window(self) -> None:
        text = "y" * 400
        assert chunk_windows(text, 400, 12) == [text]

    def test_one_char_over_boundary(self) -> None:
        text = "z" * 401
        windows = chunk_windows(text, 400, 12)
        assert len(windows) >= 2
        assert windows[-1] == "z" * 201  # tail content retained


class TestContiguityGate:
    """The paper's example shape: max-pooling sees 0.32, the aggregate sees
    enough summed excess to flag."""

    def test_accumulated_fragments_flag(self) -> None:
        # Three contiguous windows each carrying sub-BLOCK-floor signal.
        result = contiguity_gate(
            [0.0, 0.72, 0.73, 0.72, 0.0], benign_threshold=0.60, decision_threshold=0.30
        )
        assert result.flagged
        assert result.max_run_len == 3
        assert result.max_run_sum == pytest.approx(0.37, abs=1e-9)

    def test_paper_example_recovers_maxpool_miss(self) -> None:
        # Windows at 0.45/0.48/0.47 above a 0.35 benign background: each
        # individually below a 0.5-style decision point, the summed excess
        # (0.10 + 0.13 + 0.12 = 0.35) flags.
        result = contiguity_gate([0.45, 0.48, 0.47], benign_threshold=0.35, decision_threshold=0.30)
        assert result.flagged
        assert result.max_run_sum == pytest.approx(0.35, abs=1e-9)

    def test_isolated_single_window_noise_does_not_flag(self) -> None:
        # One window above background, neighbors clean — FPR control.
        result = contiguity_gate([0.0, 0.70, 0.0], benign_threshold=0.60, decision_threshold=0.30)
        assert not result.flagged
        assert result.max_run_len == 1

    def test_two_window_run_below_sum_does_not_flag(self) -> None:
        result = contiguity_gate([0.65, 0.65, 0.0], benign_threshold=0.60, decision_threshold=0.30)
        assert not result.flagged  # sum = 0.10 < 0.30

    def test_clean_windows_never_flag(self) -> None:
        result = contiguity_gate([0.10, 0.20, 0.30], benign_threshold=0.60, decision_threshold=0.30)
        assert result == ContiguityResult(flagged=False, max_run_sum=0.0, max_run_len=0)

    def test_two_separate_small_runs_do_not_flag(self) -> None:
        # Runs are maximal and independent: 0.10 + 0.10 must not merge
        # across the clean window between them.
        result = contiguity_gate([0.70, 0.0, 0.70], benign_threshold=0.60, decision_threshold=0.30)
        assert not result.flagged

    def test_gap_bounded_by_clean_window(self) -> None:
        # A run ends when a window drops to/below background.
        result = contiguity_gate(
            [0.80, 0.60, 0.80, 0.80], benign_threshold=0.60, decision_threshold=0.30
        )
        # First run: one window (0.20 excess). Second run: two windows (0.40).
        assert result.flagged
        assert result.max_run_len == 2
        assert result.max_run_sum == pytest.approx(0.40, abs=1e-9)

    def test_min_run_one_allows_single_window_flag(self) -> None:
        result = contiguity_gate(
            [0.0, 0.95, 0.0], benign_threshold=0.60, decision_threshold=0.30, min_run=1
        )
        assert result.flagged

    def test_all_above_background_long_run(self) -> None:
        result = contiguity_gate(
            [0.70, 0.71, 0.72, 0.73], benign_threshold=0.60, decision_threshold=0.30
        )
        assert result.flagged
        assert result.max_run_len == 4
        assert result.max_run_sum == pytest.approx(0.46, abs=1e-9)

    def test_empty_similarities(self) -> None:
        result = contiguity_gate([], benign_threshold=0.60, decision_threshold=0.30)
        assert not result.flagged
        assert result.max_run_len == 0
