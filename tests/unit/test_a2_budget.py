"""Contracts for the A2 attacker-generation budget.

The 2026-11-17 scheduled Bench gate went red for the first time in six runs:
step "Run A2 live benchmark (pattern_only)" ran **39 m 16 s** and was killed by
`timeout-minutes: 40`. The step's cost was `5 goals x (jb_iterations - 1)` =
10 unbounded local 7B generations on a 2-core runner, with **no per-call
timeout and no overall budget** — so wall-clock was a function of the runner's
CPU. A job timeout also *cancels the run* rather than failing the job, which is
why the job's `continue-on-error: true` never got a chance to keep the run green.

The fix is not a bigger timeout — that is how the 39 minutes got legitimised.
It is two bounds plus an honest label:

1. a per-call timeout, so one slow generation cannot own the step;
2. an overall generation budget, so the set is truncated deterministically;
3. `generation_complete: false` when it is truncated, so a partial attack set is
   never quoted as the README baseline.

These contracts are pure — no NeuralStrike, no Ollama, no network — so they run
in the ordinary unit suite, which the failing job could not.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_BENCH_YML = _REPO / ".github" / "workflows" / "bench.yml"


class _Clock:
    """Monotonic clock the tests advance by hand (no wall-clock waiting)."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestPerCallTimeout:
    async def test_a_hung_generation_is_abandoned_not_awaited(self) -> None:
        from benchmarks.ng_vs_ns.budget import run_with_budget

        clock = _Clock()

        async def never_returns(item: int) -> str:
            await asyncio.sleep(3600)
            return ""  # pragma: no cover - unreachable

        outcome = await run_with_budget(
            [1, 2],
            never_returns,
            per_call_timeout=0.01,
            clock=clock,
        )

        assert outcome.results == [], "a hung call must not contribute a payload"
        assert outcome.report.timed_out_calls == 2, (
            "each abandoned call must be counted; a silent skip is indistinguishable "
            "from 'the model produced nothing'"
        )
        assert outcome.report.complete is False

    async def test_one_slow_call_does_not_cancel_the_rest(self) -> None:
        from benchmarks.ng_vs_ns.budget import run_with_budget

        async def slow_then_fast(item: int) -> str:
            if item == 1:
                await asyncio.sleep(3600)
            return f"payload-{item}"

        outcome = await run_with_budget([1, 2], slow_then_fast, per_call_timeout=0.02)

        assert outcome.results == ["payload-2"], (
            "the budget must bound the step, not abort the whole set: one bad call "
            "should cost one attack, not the benchmark"
        )
        assert outcome.report.timed_out_calls == 1


class TestResultsStayAlignedWithTheirItems:
    """A dropped call must not shift every later payload onto the wrong goal."""

    async def test_produced_items_identify_which_item_produced_which_result(self) -> None:
        from benchmarks.ng_vs_ns.budget import run_with_budget

        async def first_hangs_then_ok(item: int) -> str:
            if item == "goal-a":
                await asyncio.sleep(3600)
            return f"payload-{item}"

        outcome = await run_with_budget(
            ["goal-a", "goal-b", "goal-c"], first_hangs_then_ok, per_call_timeout=0.02
        )

        assert outcome.results == ["payload-goal-b", "payload-goal-c"]
        assert outcome.produced_items == ["goal-b", "goal-c"], (
            "results must carry their originating item: zipping the input list against a "
            "list that lost an entry attributes goal-a's slot to goal-b's payload"
        )


class TestOverallBudget:
    async def test_budget_stops_the_set_and_says_so(self) -> None:
        from benchmarks.ng_vs_ns.budget import run_with_budget

        clock = _Clock()

        async def advances_clock(item: int) -> str:
            clock.now += 100.0  # every call costs 100 s of the budget
            return f"payload-{item}"

        outcome = await run_with_budget(
            list(range(10)),
            advances_clock,
            budget_seconds=250.0,
            clock=clock,
        )

        assert outcome.report.complete is False, (
            "a truncated attack set must be labelled incomplete — the ASR of a partial "
            "corpus is not the number the README quotes"
        )
        assert outcome.report.planned == 10
        assert outcome.report.produced == len(outcome.results) < 10
        assert outcome.report.budget_seconds == 250.0
        assert clock.now <= 250.0 + 100.0, (
            "the budget must be enforced: it may not start a call it cannot finish "
            "without exceeding the budget by more than one call"
        )

    async def test_no_budget_means_unbounded_for_local_runs(self) -> None:
        from benchmarks.ng_vs_ns.budget import run_with_budget

        async def produces(item: int) -> str:
            return f"payload-{item}"

        outcome = await run_with_budget(list(range(5)), produces, budget_seconds=None)

        assert outcome.results == [f"payload-{i}" for i in range(5)]
        assert outcome.report.complete is True, (
            "a local run with no deadline must stay complete; the budget exists for CI, "
            "not to truncate the benchmark by default"
        )

    async def test_empty_worker_output_is_not_counted_as_complete_work(self) -> None:
        from benchmarks.ng_vs_ns.budget import run_with_budget

        async def produces_nothing(item: int) -> str:
            return ""

        outcome = await run_with_budget([1, 2, 3], produces_nothing)

        assert outcome.results == []
        assert outcome.report.produced == 0
        assert outcome.report.complete is False, (
            "producing nothing for every item is not a complete run; treating it as one "
            "would report an attack set that does not exist"
        )


class TestTheWorkflowCannotRecreateTheTimeout:
    """A job timeout that cancels the run is not a safety net — the budget is."""

    def _a2_job(self) -> tuple[int, str]:
        text = _BENCH_YML.read_text(encoding="utf-8")
        start = text.index("bench-a2-live:")
        end = text.index("bench-b4-deterministic:")
        block = text[start:end]
        timeout = re.search(r"timeout-minutes:\s*(\d+)", block)
        assert timeout is not None, "the A2 job must declare a timeout-minutes"
        return int(timeout.group(1)), block

    def test_the_step_passes_a_generation_budget(self) -> None:
        _, block = self._a2_job()
        assert "--generation-budget-seconds" in block, (
            "the A2 step must bound its own work; without a budget its runtime is a "
            "function of the runner's CPU, which is exactly the 39-minute failure"
        )

    def test_the_budget_plus_slack_fits_inside_the_job_timeout(self) -> None:
        job_timeout_minutes, block = self._a2_job()
        budget = re.search(r"--generation-budget-seconds\s+(\d+)", block)
        assert budget is not None, "expected a literal budget on the A2 command line"
        budget_seconds = int(budget.group(1))
        # Slack for install + model pull (measured: ~55 s) + scoring + upload.
        slack_seconds = 300
        assert budget_seconds + slack_seconds <= job_timeout_minutes * 60, (
            f"generation budget {budget_seconds}s + {slack_seconds}s slack exceeds the "
            f"job's {job_timeout_minutes}-minute timeout: the job would cancel the run "
            "again instead of failing a job that continue-on-error can absorb"
        )


class TestPartialResultsCannotBeQuotedAsTheBaseline:
    def test_payload_labels_incomplete_generation(self) -> None:
        from benchmarks.ng_vs_ns.budget import BudgetReport
        from benchmarks.ng_vs_ns.live_harness import _payload

        report = BudgetReport(
            planned=10,
            produced=4,
            elapsed_seconds=120.0,
            budget_seconds=100.0,
            timed_out_calls=3,
            complete=False,
        )
        payload = _payload(
            attacker_model="m",
            judge_model="m",
            judge_resolves_escalate=False,
            jb_iterations=3,
            selected_configs=["pattern_only"],
            attacks=[],
            benign=[],
            results=[],
            generation=report,
        )

        assert payload["generation"]["complete"] is False
        assert payload["generation"]["produced"] == 4
        assert payload["generation"]["planned"] == 10

    def test_formatting_warns_when_the_set_is_truncated(self) -> None:
        from benchmarks.ng_vs_ns.budget import BudgetReport
        from benchmarks.ng_vs_ns.live_harness import _format_results

        payload = {
            "generated_at": "2026-11-17T00:00:00+00:00",
            "source_commit": None,
            "corpus": {},
            "attacker_model": "m",
            "judge_model": "m",
            "jb_iterations": 3,
            "n_attacks": 4,
            "n_benign": 45,
            "generation": BudgetReport(
                planned=10,
                produced=4,
                elapsed_seconds=120.0,
                budget_seconds=100.0,
                timed_out_calls=3,
                complete=False,
            ).__dict__,
            "configs": [
                {
                    "config": "pattern_only",
                    "n_attacks": 4,
                    "n_benign": 45,
                    "asr": 0.5,
                    "fpr": 0.0,
                    "p95_latency_ms": 2.0,
                    "attack_verdicts": [],
                    "benign_verdicts": [],
                }
            ],
        }

        out = _format_results(payload)

        assert "INCOMPLETE" in out, (
            "the console output is what a human reads in the job log; a truncated set "
            "must say so there, not only in the JSON"
        )
        assert "4 of 10" in out


@pytest.mark.parametrize("field", ["planned", "produced", "elapsed_seconds", "complete"])
def test_report_carries_the_fields_the_payload_publishes(field: str) -> None:
    from benchmarks.ng_vs_ns.budget import BudgetReport

    report = BudgetReport(
        planned=1,
        produced=1,
        elapsed_seconds=0.0,
        budget_seconds=None,
        timed_out_calls=0,
        complete=True,
    )
    assert hasattr(report, field)


class TestTheIncompleteRunFlagsItself:
    """A truncated set must not report success to the orchestrator."""

    async def test_cli_exits_non_zero_when_generation_is_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchmarks.ng_vs_ns import live_harness

        payload = {
            "generated_at": "2026-11-17T00:00:00+00:00",
            "source_commit": None,
            "corpus": {},
            "attacker_model": "m",
            "judge_model": "m",
            "jb_iterations": 3,
            "n_attacks": 2,
            "n_benign": 1,
            "generation": {
                "planned": 10,
                "produced": 2,
                "elapsed_seconds": 600.0,
                "budget_seconds": 540.0,
                "timed_out_calls": 8,
                "complete": False,
            },
            "configs": [
                {
                    "config": "pattern_only",
                    "n_attacks": 2,
                    "n_benign": 1,
                    "asr": 0.5,
                    "fpr": 0.0,
                    "p95_latency_ms": 2.0,
                    "attack_verdicts": [],
                    "benign_verdicts": [],
                }
            ],
        }

        async def fake_run(**_kwargs: object) -> dict[str, object]:
            return payload

        monkeypatch.setattr(live_harness, "run", fake_run)
        code = await live_harness._cli_main(["--save", str(tmp_path / "out.json")])

        assert code == 2, (
            "an incomplete set is not a passing benchmark; exit 2 flags the job "
            "(continue-on-error keeps the scheduled run green) instead of reporting "
            "success for numbers that describe a partial corpus"
        )
