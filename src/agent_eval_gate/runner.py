"""The runner: k runs per scenario, every layer scored, traces saved.

This is the orchestration layer and nothing else. The loop lives in the
adapter, the verdicts live in the scorers, the numbers live in stats. The
runner's jobs are to fan the work out with bounded concurrency, to never let
one failure take the suite down, and to leave every trace on disk so a human
can label it later.

Concurrency is bounded by a semaphore. Each run launches a fresh world
subprocess and makes several API calls; a few in flight at once is a large
speedup, dozens is a rate-limit error.
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from anthropic import AsyncAnthropic

from agent_eval_gate.adapters.claude import run_scenario
from agent_eval_gate.results import RunOutcome, SuiteResult
from agent_eval_gate.scenarios.schema import Scenario
from agent_eval_gate.scorers.end_state import score_end_state
from agent_eval_gate.scorers.faithfulness import score_faithfulness
from agent_eval_gate.scorers.lexical import score_lexical
from agent_eval_gate.scorers.trajectory import score_trajectory
from agent_eval_gate.trace import Trace

DEFAULT_K = 5
DEFAULT_CONCURRENCY = 4


def trace_path(out_dir: Path, scenario_id: str, run_index: int) -> Path:
    return out_dir / "traces" / f"{scenario_id}-{run_index}.json"


def save_trace(out_dir: Path, trace: Trace) -> None:
    path = trace_path(out_dir, trace.scenario_id, trace.run_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(trace.model_dump_json(indent=2))


async def score_run(
    scenario: Scenario,
    trace: Trace,
    *,
    judge_model: str | None,
    client: AsyncAnthropic | None = None,
) -> RunOutcome:
    """Apply every layer to one trace and fold the verdicts into an outcome."""
    end_state = score_end_state(scenario, trace)
    trajectory = score_trajectory(scenario, trace)
    lexical = score_lexical(trace) if not trace.error else None

    faithfulness = None
    if judge_model is not None:
        faithfulness = await score_faithfulness(
            scenario.task, trace, client=client, judge_model=judge_model
        )

    details = []
    if not end_state.passed:
        details.append(f"end_state: {end_state.summary}")
    if not trajectory.passed:
        details.append(f"trajectory: {trajectory.summary}")
    if faithfulness is not None and not faithfulness.passed:
        details.append(f"faithfulness: {faithfulness.summary}")

    layers = [end_state.passed, trajectory.passed]
    if faithfulness is not None:
        layers.append(faithfulness.passed)

    return RunOutcome(
        scenario_id=scenario.id,
        category=scenario.category,
        run_index=trace.run_index,
        end_state=end_state.passed,
        trajectory=trajectory.passed,
        faithfulness=faithfulness.passed if faithfulness is not None else None,
        lexical=lexical.faithful if lexical is not None else None,
        passed=all(layers),
        error=trace.error,
        detail=" | ".join(details),
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        duration_ms=trace.duration_ms,
    )


async def run_suite(
    scenarios: list[Scenario],
    *,
    model: str,
    judge_model: str | None,
    k: int = DEFAULT_K,
    out_dir: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
    client: AsyncAnthropic | None = None,
    on_outcome=None,
) -> SuiteResult:
    """Run every scenario k times, score each run, save traces, return results."""
    client = client or AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    suite = SuiteResult(
        model=model,
        judge_model=judge_model,
        k=k,
        started_at=datetime.now(UTC),
        scenario_count=len(scenarios),
    )

    async def one(scenario: Scenario, run_index: int) -> RunOutcome:
        async with semaphore:
            trace = await run_scenario(scenario, client=client, model=model, run_index=run_index)
            save_trace(out_dir, trace)
            outcome = await score_run(scenario, trace, judge_model=judge_model, client=client)
            if on_outcome:
                on_outcome(outcome)
            return outcome

    jobs = [one(scenario, i) for scenario in scenarios for i in range(k)]
    suite.outcomes = sorted(await asyncio.gather(*jobs), key=lambda o: (o.scenario_id, o.run_index))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(suite.model_dump_json(indent=2))
    return suite


def load_suite(path: Path) -> SuiteResult:
    return SuiteResult.model_validate(json.loads(path.read_text()))
