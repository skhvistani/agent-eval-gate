"""The gate. The property that matters: "cannot tell" does not pass."""

from datetime import UTC, datetime

from agent_eval_gate.gate import evaluate
from agent_eval_gate.results import RunOutcome, SuiteResult
from agent_eval_gate.scenarios.schema import Category


def suite(pattern: dict[str, list[bool]], model="m") -> SuiteResult:
    outcomes = [
        RunOutcome(
            scenario_id=sid,
            category=Category.SINGLE_ACTION,
            run_index=i,
            end_state=p,
            trajectory=p,
            passed=p,
        )
        for sid, runs in pattern.items()
        for i, p in enumerate(runs)
    ]
    return SuiteResult(
        model=model,
        judge_model=None,
        k=max((len(r) for r in pattern.values()), default=0),
        started_at=datetime.now(UTC),
        scenario_count=len(pattern),
        outcomes=outcomes,
    )


def all_pass(n=20, k=5):
    return suite({f"s{i}": [True] * k for i in range(n)})


# --- the core property --------------------------------------------------


def test_identical_runs_pass_even_with_few_scenarios():
    """Finding #2. The absolute rule failed a run against itself at n=5;
    the paired rule reports a difference of exactly zero and passes."""
    five = suite({f"s{i}": [i != 0] * 5 for i in range(5)})  # 80%, wide interval
    verdict = evaluate(five, five)
    assert verdict.passed
    assert verdict.paired is not None
    assert verdict.paired.point == 0.0 and verdict.paired.low == 0.0


def test_clear_regression_fails():
    baseline = all_pass()
    candidate = suite({f"s{i}": [i % 2 == 0] * 5 for i in range(20)})  # 50%
    verdict = evaluate(candidate, baseline)
    assert not verdict.passed
    assert "below" in verdict.reason
    assert verdict.paired.point < -0.4


def test_improvement_on_the_same_scenarios_passes():
    """The gate is about regression. Getting better is not a failure."""
    baseline = suite({f"s{i}": [i >= 2] * 5 for i in range(10)})  # 80%
    candidate = suite({f"s{i}": [i >= 1] * 5 for i in range(10)})  # 90%
    verdict = evaluate(candidate, baseline)
    assert verdict.passed
    assert verdict.paired.point > 0


def test_cannot_tell_does_not_pass():
    """Same pass rate, but a different scenario failed: the point difference
    is zero and the interval reaches well below −tolerance. That is not
    "no regression", it is "cannot tell", and the gate says no."""
    baseline = suite({f"s{i}": [i != 0] * 5 for i in range(10)})  # s0 fails
    candidate = suite({f"s{i}": [i != 1] * 5 for i in range(10)})  # s1 fails
    verdict = evaluate(candidate, baseline, tolerance=0.05)
    assert verdict.paired.point == 0.0
    assert not verdict.passed
    assert verdict.paired.low < -0.05


def test_same_swap_with_more_scenarios_clears():
    """The same one-scenario swap across 100 scenarios: the interval tightens
    to within tolerance. A narrower gate is earned with scenarios."""
    baseline = suite({f"s{i}": [i != 0] * 5 for i in range(100)})
    candidate = suite({f"s{i}": [i != 1] * 5 for i in range(100)})
    verdict = evaluate(candidate, baseline, tolerance=0.05)
    assert verdict.paired.point == 0.0
    assert verdict.passed


# --- tolerance ----------------------------------------------------------


def test_tolerance_widens_what_passes():
    baseline = all_pass(40)
    candidate = suite({f"s{i}": [i % 10 != 0] * 5 for i in range(40)})  # 90%: −10%
    assert not evaluate(candidate, baseline, tolerance=0.05).passed
    assert evaluate(candidate, baseline, tolerance=0.25).passed


# --- comparability ------------------------------------------------------


def test_different_scenario_set_is_not_comparable():
    baseline = all_pass(20)
    candidate = suite({f"other{i}": [True] * 5 for i in range(20)})
    verdict = evaluate(candidate, baseline)
    assert not verdict.passed
    assert "not comparable" in verdict.reason
    assert verdict.paired is None


def test_partial_overlap_below_threshold_is_not_comparable():
    baseline = all_pass(20)
    pattern = {f"s{i}": [True] * 5 for i in range(10)} | {f"x{i}": [True] * 5 for i in range(10)}
    verdict = evaluate(suite(pattern), baseline)
    assert not verdict.passed
    assert verdict.scenario_overlap == 0.5


def test_empty_candidate_fails():
    verdict = evaluate(suite({}), all_pass())
    assert not verdict.passed
    assert "no runs" in verdict.reason


# --- reporting ----------------------------------------------------------


def test_harness_errors_are_named_in_the_verdict():
    baseline = all_pass(10)
    candidate = all_pass(10)
    for o in candidate.outcomes[:20]:
        o.passed = False
        o.error = "AuthenticationError: 401"
    verdict = evaluate(candidate, baseline)
    assert not verdict.passed
    assert verdict.candidate_errors == 20
    assert "harness errors" in verdict.reason


def test_summary_is_readable():
    verdict = evaluate(all_pass(), all_pass())
    text = verdict.summary
    assert text.startswith("gate: PASS")
    assert "paired difference" in text and "baseline" in text
