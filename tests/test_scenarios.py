"""The scenario schema and loader.

Two jobs here. One: the shipped example scenarios are valid and describe the
seeded world accurately — a scenario that references a nonexistent order would
fail at scoring time for the wrong reason. Two: bad input is rejected loudly.
"""

import pytest
from pydantic import ValidationError

from agent_eval_gate.scenarios.loader import ScenarioLoadError, load_scenario, load_scenarios
from agent_eval_gate.scenarios.schema import Category, ExpectedChange, Scenario
from agent_eval_gate.worlds.orders.store import World


def test_examples_load_and_validate():
    scenarios = load_scenarios()
    assert len(scenarios) == 5


def test_every_category_has_an_example():
    categories = {scenario.category for scenario in load_scenarios()}
    assert categories == set(Category)


def test_scenario_ids_are_unique():
    ids = [scenario.id for scenario in load_scenarios()]
    assert len(ids) == len(set(ids))


def test_expected_changes_reference_real_records():
    """A scenario that points at a nonexistent record is a broken test, not a failing agent."""
    for scenario in load_scenarios():
        world = World(scenario.seed)
        collections = {"orders": world.orders, "tickets": world.tickets}
        for change in scenario.expected_changes:
            assert change.entity in collections, f"{scenario.id}: unknown entity {change.entity}"
            assert change.id in collections[change.entity], f"{scenario.id}: unknown {change.id}"


def test_expected_changes_actually_differ_from_the_seed_world():
    """If the expected value already holds at seed time, the scenario proves nothing."""
    for scenario in load_scenarios():
        world = World(scenario.seed)
        collections = {"orders": world.orders, "tickets": world.tickets}
        for change in scenario.expected_changes:
            record = collections[change.entity][change.id]
            assert getattr(record, change.field) != change.value, (
                f"{scenario.id}: {change.id}.{change.field} is already {change.value!r} at seed"
            )


def test_refusal_trap_expects_an_untouched_world():
    traps = [s for s in load_scenarios() if s.category is Category.REFUSAL_TRAP]
    assert traps
    for trap in traps:
        assert trap.expected_changes == []


def test_unknown_category_is_rejected():
    with pytest.raises(ValidationError):
        Scenario(id="x", seed=17, task="t", category="lookupp")


def test_non_integer_seed_is_rejected():
    with pytest.raises(ValidationError):
        Scenario(id="x", seed="seventeen", task="t", category=Category.LOOKUP)


def test_defaults_are_not_shared_between_instances():
    """Pydantic copies mutable defaults; a plain class would share one list."""
    a = Scenario(id="a", seed=17, task="t", category=Category.LOOKUP)
    b = Scenario(id="b", seed=17, task="t", category=Category.LOOKUP)
    a.expected_changes.append(
        ExpectedChange(entity="orders", id="ORD-1001", field="status", value="refunded")
    )
    assert b.expected_changes == []


def test_malformed_yaml_names_the_file(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text("id: x\n  seed: [unclosed\n")
    with pytest.raises(ScenarioLoadError, match="broken.yaml"):
        load_scenario(bad)


def test_invalid_scenario_names_the_file(tmp_path):
    bad = tmp_path / "wrong-category.yaml"
    bad.write_text("id: x\nseed: 17\ntask: t\ncategory: nonsense\n")
    with pytest.raises(ScenarioLoadError, match="wrong-category.yaml"):
        load_scenario(bad)
