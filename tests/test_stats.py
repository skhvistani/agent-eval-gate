"""The statistics. Hand-checkable cases, plus the clustering property that
justifies resampling scenarios rather than runs."""

import pytest

from agent_eval_gate.results import RunOutcome
from agent_eval_gate.scenarios.schema import Category
from agent_eval_gate.stats import (
    bootstrap,
    paired_difference,
    pass_power_k,
    pass_rate,
    per_category,
    per_layer,
)


def outcome(sid, run, passed, category=Category.SINGLE_ACTION, **layers):
    return RunOutcome(
        scenario_id=sid,
        category=category,
        run_index=run,
        end_state=layers.get("end_state", passed),
        trajectory=layers.get("trajectory", passed),
        faithfulness=layers.get("faithfulness", passed),
        passed=passed,
    )


def suite(pattern: dict[str, list[bool]], category=Category.SINGLE_ACTION):
    """{"s1": [True, False, ...], ...} -> outcomes."""
    return [
        outcome(sid, i, p, category) for sid, runs in pattern.items() for i, p in enumerate(runs)
    ]


# --- point estimates ----------------------------------------------------


def test_pass_rate_is_over_runs():
    assert pass_rate(suite({"a": [True, True], "b": [False, False]})) == 0.5


def test_pass_power_k_is_over_scenarios_and_requires_all_runs():
    runs = suite({"a": [True, True, True], "b": [True, True, False], "c": [True, True, True]})
    assert pass_rate(runs) == pytest.approx(8 / 9)
    assert pass_power_k(runs) == pytest.approx(2 / 3)


def test_empty_is_safe():
    assert pass_rate([]) == 0.0
    assert pass_power_k([]) == 0.0
    assert bootstrap([]).n_runs == 0


# --- bootstrap ----------------------------------------------------------


def test_all_pass_gives_a_degenerate_interval_at_one():
    estimate = bootstrap(suite({f"s{i}": [True] * 5 for i in range(10)}))
    assert estimate.point == 1.0
    assert estimate.low == 1.0 and estimate.high == 1.0


def test_interval_contains_the_point():
    runs = suite({f"s{i}": [i % 3 != 0] * 5 for i in range(12)})
    estimate = bootstrap(runs)
    assert estimate.low <= estimate.point <= estimate.high
    assert estimate.n_scenarios == 12 and estimate.n_runs == 60


def test_bootstrap_is_reproducible_under_a_seed():
    """Same seed, same interval: a published number must be re-derivable."""
    runs = suite({f"s{i}": [i % 2 == 0] * 3 for i in range(8)})
    assert bootstrap(runs, seed=1) == bootstrap(runs, seed=1)


def test_resampling_scenarios_gives_wider_intervals_than_resampling_runs():
    """The clustering point: runs within a scenario are correlated.

    Build a suite where each scenario is all-pass or all-fail. Within-scenario
    variance is zero, so run-level resampling sees a huge sample of
    independent coins and reports a tight interval. Scenario-level resampling
    sees ten coins and reports an honest, wide one.
    """
    runs = suite({f"s{i}": [i % 2 == 0] * 20 for i in range(10)})
    clustered = bootstrap(runs)

    # Naive: pretend every run is its own scenario.
    naive_runs = [outcome(f"{o.scenario_id}-{o.run_index}", 0, o.passed) for o in runs]
    naive = bootstrap(naive_runs)

    assert clustered.point == naive.point == 0.5
    assert clustered.width > 2 * naive.width


def test_more_scenarios_narrow_the_interval():
    small = bootstrap(suite({f"s{i}": [i % 2 == 0] * 5 for i in range(6)}))
    large = bootstrap(suite({f"s{i}": [i % 2 == 0] * 5 for i in range(60)}))
    assert large.width < small.width


# --- breakdowns ---------------------------------------------------------


def test_per_category_splits_and_bootstraps_each():
    runs = suite({"a": [True] * 3}, Category.LOOKUP) + suite(
        {"b": [False] * 3}, Category.REFUSAL_TRAP
    )
    result = per_category(runs)
    assert set(result) == {"lookup", "refusal_trap"}
    assert result["lookup"].point == 1.0
    assert result["refusal_trap"].point == 0.0


def test_per_layer_isolates_where_failures_come_from():
    runs = [
        outcome("a", 0, False, end_state=True, trajectory=False, faithfulness=True),
        outcome("a", 1, False, end_state=True, trajectory=False, faithfulness=True),
        outcome("b", 0, True),
    ]
    result = per_layer(runs)
    assert result["end_state"].point == 1.0
    assert result["trajectory"].point == pytest.approx(1 / 3)
    assert result["faithfulness"].point == 1.0


def test_per_layer_skips_a_layer_that_never_ran():
    runs = [outcome("a", 0, True, faithfulness=None)]
    assert "faithfulness" not in per_layer(runs)


# --- paired comparison --------------------------------------------------


def test_paired_difference_on_identical_results_is_zero():
    a = suite({f"s{i}": [i % 2 == 0] * 3 for i in range(10)})
    estimate = paired_difference(a, a)
    assert estimate.point == 0.0
    assert estimate.low == 0.0 and estimate.high == 0.0


def test_paired_difference_sign_and_overlap():
    a = suite({f"s{i}": [True] * 3 for i in range(10)})
    b = suite({f"s{i}": [i % 2 == 0] * 3 for i in range(10)})
    estimate = paired_difference(a, b)
    assert estimate.point == pytest.approx(0.5)
    assert estimate.low > 0, "A is better than B and the interval should say so"


def test_paired_difference_uses_only_shared_scenarios():
    a = suite({"s1": [True], "s2": [True], "only_a": [True]})
    b = suite({"s1": [True], "s2": [False], "only_b": [False]})
    estimate = paired_difference(a, b)
    assert estimate.n_scenarios == 2
    assert estimate.point == pytest.approx(0.5)
