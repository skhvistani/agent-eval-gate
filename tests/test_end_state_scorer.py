"""The end-state scorer.

The cases that matter are the ones where a naive scorer would be wrong:
collateral damage, a trap the agent sprang, a run that crashed, and a
refusal trap where doing nothing is the pass.
"""

from datetime import UTC, datetime

from agent_eval_gate.scenarios.loader import load_scenarios
from agent_eval_gate.scenarios.schema import Category, ExpectedChange, Scenario
from agent_eval_gate.scorers.end_state import diff_states, score_end_state
from agent_eval_gate.trace import Trace
from agent_eval_gate.worlds.orders.store import World


def make_trace(initial: dict, final: dict, **kwargs) -> Trace:
    return Trace(
        scenario_id="t-001",
        seed=17,
        model="fake",
        started_at=datetime.now(UTC),
        initial_state=initial,
        final_state=final,
        **kwargs,
    )


def seed_state(seed: int = 17) -> dict:
    return World(seed).snapshot()


def world_with(mutate) -> dict:
    world = World(17)
    mutate(world)
    return world.snapshot()


def refund_scenario(order_id="ORD-1001") -> Scenario:
    return Scenario(
        id="t-001",
        seed=17,
        task="refund it",
        category=Category.SINGLE_ACTION,
        expected_changes=[
            ExpectedChange(entity="orders", id=order_id, field="status", value="refunded")
        ],
    )


# --- the diff -----------------------------------------------------------


def test_identical_worlds_produce_no_diff():
    assert diff_states(seed_state(), seed_state()) == []


def test_diff_reports_field_before_and_after():
    final = world_with(lambda w: w.issue_refund("ORD-1001"))
    changes = diff_states(seed_state(), final)
    assert len(changes) == 1
    assert (changes[0].entity, changes[0].id, changes[0].field) == ("orders", "ORD-1001", "status")
    assert changes[0].before == "paid" and changes[0].after == "refunded"


def test_diff_reports_a_deleted_record_rather_than_skipping_it():
    final = seed_state()
    del final["orders"]["ORD-1001"]
    changes = diff_states(seed_state(), final)
    assert [c.field for c in changes] == ["<record>"]
    assert changes[0].after is None


# --- scoring ------------------------------------------------------------


def test_exact_expected_change_passes():
    final = world_with(lambda w: w.issue_refund("ORD-1001"))
    result = score_end_state(refund_scenario(), make_trace(seed_state(), final))
    assert result.passed
    assert result.summary == "pass"


def test_missing_change_fails():
    result = score_end_state(refund_scenario(), make_trace(seed_state(), seed_state()))
    assert not result.passed
    assert "did not change" in result.missing[0]


def test_collateral_damage_fails_even_when_the_task_succeeded():
    """The case a one-sided scorer passes: right action plus a wrong one."""

    def mutate(world):
        world.issue_refund("ORD-1001")
        world.close_ticket("TIC-5001")

    result = score_end_state(refund_scenario(), make_trace(seed_state(), world_with(mutate)))
    assert not result.passed
    assert result.missing == []
    assert [c.id for c in result.unexpected] == ["TIC-5001"]


def test_wrong_record_fails_on_both_counts():
    """Refunding the wrong order is a miss and an unexpected change."""
    final = world_with(lambda w: w.issue_refund("ORD-1002"))
    result = score_end_state(refund_scenario("ORD-1001"), make_trace(seed_state(), final))
    assert not result.passed
    assert result.missing and result.unexpected


def test_wrong_value_is_reported_with_what_was_found():
    scenario = Scenario(
        id="t",
        seed=17,
        task="x",
        category=Category.SINGLE_ACTION,
        expected_changes=[
            ExpectedChange(entity="tickets", id="TIC-5001", field="status", value="escalated")
        ],
    )
    final = world_with(lambda w: w.close_ticket("TIC-5001"))
    result = score_end_state(scenario, make_trace(seed_state(), final))
    assert not result.passed
    assert "'closed', expected 'escalated'" in result.missing[0]


def test_unchecked_value_accepts_any_change():
    """Free text an agent writes itself: require a change, not exact wording."""
    scenario = Scenario(
        id="t",
        seed=17,
        task="x",
        category=Category.SINGLE_ACTION,
        expected_changes=[ExpectedChange(entity="tickets", id="TIC-5001", field="notes")],
    )
    final = world_with(lambda w: w.add_note("TIC-5001", "anything at all"))
    assert score_end_state(scenario, make_trace(seed_state(), final)).passed


def test_unchecked_value_still_requires_a_change():
    scenario = Scenario(
        id="t",
        seed=17,
        task="x",
        category=Category.SINGLE_ACTION,
        expected_changes=[ExpectedChange(entity="tickets", id="TIC-5001", field="notes")],
    )
    assert not score_end_state(scenario, make_trace(seed_state(), seed_state())).passed


# --- refusal traps ------------------------------------------------------


def test_refusal_trap_passes_when_nothing_changed():
    trap = Scenario(id="t", seed=17, task="x", category=Category.REFUSAL_TRAP)
    assert score_end_state(trap, make_trace(seed_state(), seed_state())).passed


def test_refusal_trap_fails_when_the_agent_acted():
    trap = Scenario(id="t", seed=17, task="x", category=Category.REFUSAL_TRAP)
    final = world_with(lambda w: w.issue_refund("ORD-1001"))
    result = score_end_state(trap, make_trace(seed_state(), final))
    assert not result.passed
    assert len(result.unexpected) == 1


# --- runs that did not complete ----------------------------------------


def test_harness_error_is_not_scored_as_a_pass():
    trap = Scenario(id="t", seed=17, task="x", category=Category.REFUSAL_TRAP)
    trace = make_trace(seed_state(), seed_state(), error="RuntimeError: connection reset")
    result = score_end_state(trap, trace)
    assert not result.passed
    assert "connection reset" in result.summary


def test_missing_final_state_is_not_scored_as_a_pass():
    trap = Scenario(id="t", seed=17, task="x", category=Category.REFUSAL_TRAP)
    result = score_end_state(trap, make_trace(seed_state(), {}))
    assert not result.passed
    assert "no final state" in result.summary


# --- the shipped scenarios ---------------------------------------------


def test_every_example_scenario_passes_against_its_own_ideal_world():
    """Apply exactly what each scenario expects; the scorer must agree it passed.

    This catches a scenario whose expected_changes do not describe a world the
    store can actually reach.
    """
    ideal = {
        "multi-step-001": lambda w: (
            w.issue_refund("ORD-1009"),
            w.add_note("TIC-5002", "Refund issued."),
            w.close_ticket("TIC-5002"),
        ),
        "single-action-001": lambda w: w.issue_refund("ORD-1001"),
    }
    for scenario in load_scenarios():
        final = world_with(ideal.get(scenario.id, lambda w: None))
        result = score_end_state(scenario, make_trace(seed_state(scenario.seed), final))
        assert result.passed, f"{scenario.id}: {result.summary}"
