"""Bounded execution for the live attacker's model calls.

The A2 benchmark's cost is set by the *model*, not by the code: `5 goals x
(jb_iterations - 1)` local generations, each unbounded, on whatever CPU the
runner happens to offer. On 2026-11-17 that produced a 39 m 16 s step killed by
the job's 40-minute timeout — and a job timeout cancels the whole run, so the
job's `continue-on-error: true` never applied.

Two bounds fix that shape without pretending the slowdown did not happen:

- **per call** — a hung generation is abandoned and counted, so one slow call
  costs one attack instead of the benchmark;
- **overall** — once the budget is spent the set is truncated, deterministically.

Neither bound is allowed to lie about the result: truncation is reported
(`complete=False`, planned vs produced, timeouts counted) so a partial attack
set can never be quoted as the baseline in the README.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence


@dataclass
class BudgetReport:
    """What actually happened, so the artifact can say whether it is whole."""

    planned: int
    produced: int
    elapsed_seconds: float
    budget_seconds: float | None
    timed_out_calls: int
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetOutcome:
    """Results that were produced, plus the report that bounds them."""

    results: list[Any]
    report: BudgetReport
    produced_items: list[Any] = field(default_factory=list)


async def run_with_budget(
    items: Sequence[Any],
    worker: Callable[[Any], Awaitable[Any]],
    *,
    budget_seconds: float | None = None,
    per_call_timeout: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> BudgetOutcome:
    """Run ``worker`` over ``items`` under a wall-clock budget and per-call timeout.

    Args:
        items: Work units (e.g. one (goal, iteration) pair per mutation).
        worker: Awaited per item; a falsy return means "no output produced" and is
            not counted (an empty generation is not a generated attack).
        budget_seconds: Stop starting new items once this much time has elapsed.
            ``None`` runs the whole set (the local default — the budget exists to
            bound CI, not to truncate the benchmark by design).
        per_call_timeout: Abandon a single call after this long; a timeout is
            counted and the remaining items still run.
        clock: Injectable monotonic clock (tests advance it without waiting).

    Returns:
        :class:`BudgetOutcome` — produced results plus a report that says whether
        the set is complete.
    """
    started = clock()
    results: list[Any] = []
    produced_items: list[Any] = []
    timed_out = 0
    truncated = False

    for item in items:
        if budget_seconds is not None and (clock() - started) >= budget_seconds:
            truncated = True
            break
        try:
            if per_call_timeout is not None:
                value = await asyncio.wait_for(worker(item), timeout=per_call_timeout)
            else:
                value = await worker(item)
        except TimeoutError:
            timed_out += 1
            continue
        if value:
            results.append(value)
            produced_items.append(item)

    elapsed = clock() - started
    produced = len(results)
    complete = not truncated and timed_out == 0 and produced == len(items)

    return BudgetOutcome(
        results=results,
        report=BudgetReport(
            planned=len(items),
            produced=produced,
            elapsed_seconds=round(elapsed, 3),
            budget_seconds=budget_seconds,
            timed_out_calls=timed_out,
            complete=complete,
        ),
        produced_items=produced_items,
    )
