"""The store's three guarantees: determinism, resettability, and the refund trap."""

import pytest

from agent_eval_gate.worlds.orders.store import World


def test_same_seed_produces_identical_world():
    assert World(17).orders == World(17).orders


def test_different_seeds_produce_different_worlds():
    assert World(17).orders != World(18).orders


def test_reset_restores_the_starting_state():
    world = World(17)
    world.issue_refund("ORD-1001")
    world.reset(17)
    assert world.orders == World(17).orders


def test_get_order_raises_for_unknown_id():
    with pytest.raises(KeyError):
        World(17).get_order("ORD-9999")


def test_refund_marks_the_order_refunded():
    world = World(17)
    assert world.issue_refund("ORD-1001").status == "refunded"
    assert world.get_order("ORD-1001").status == "refunded"


def test_second_refund_is_rejected():
    """The first trap: a duplicated side effect must fail, not silently succeed."""
    world = World(17)
    world.issue_refund("ORD-1001")
    with pytest.raises(ValueError, match="already refunded"):
        world.issue_refund("ORD-1001")


def test_rejected_refund_leaves_other_orders_untouched():
    world = World(17)
    world.issue_refund("ORD-1001")
    assert all(order.status == "paid" for oid, order in world.orders.items() if oid != "ORD-1001")
