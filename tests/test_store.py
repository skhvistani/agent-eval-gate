"""The store's guarantees: determinism, resettability, and the traps.

These run against the store directly rather than over MCP, so they cost
microseconds. The protocol layer is smoke-tested separately in test_server.py.
"""

import pytest

from agent_eval_gate.worlds.orders.store import N_PRE_CLOSED, N_PRE_REFUNDED, World


def snapshot(world: World) -> tuple[dict, dict]:
    """Everything that defines world state, for whole-world comparisons."""
    return (dict(world.orders), dict(world.tickets))


# --- determinism --------------------------------------------------------


def test_same_seed_produces_identical_world():
    assert snapshot(World(17)) == snapshot(World(17))


def test_different_seeds_produce_different_worlds():
    assert snapshot(World(17)) != snapshot(World(18))


def test_reset_restores_the_starting_state():
    world = World(17)
    world.issue_refund("ORD-1001")
    world.close_ticket("TIC-5001")
    world.add_note("TIC-5002", "touched")
    world.reset(17)
    assert snapshot(world) == snapshot(World(17))


# --- seed world shape ---------------------------------------------------


def test_seed_world_contains_already_refunded_orders():
    """Refusal traps need a world where the wrong answer is already true."""
    refunded = [o for o in World(17).orders.values() if o.status == "refunded"]
    assert len(refunded) == N_PRE_REFUNDED


def test_seed_world_contains_already_closed_tickets():
    closed = [t for t in World(17).tickets.values() if t.status == "closed"]
    assert len(closed) == N_PRE_CLOSED


def test_every_ticket_points_at_a_real_order():
    world = World(17)
    assert all(ticket.order_id in world.orders for ticket in world.tickets.values())


def test_tickets_do_not_share_a_notes_list():
    """default_factory, not a bare []: otherwise one note lands on every ticket."""
    world = World(17)
    world.add_note("TIC-5001", "only here")
    assert world.get_ticket("TIC-5002").notes == []


# --- orders -------------------------------------------------------------


def test_get_order_raises_for_unknown_id():
    with pytest.raises(KeyError):
        World(17).get_order("ORD-9999")


def test_refund_marks_the_order_refunded():
    world = World(17)
    assert world.issue_refund("ORD-1001").status == "refunded"
    assert world.get_order("ORD-1001").status == "refunded"


def test_second_refund_is_rejected():
    """The core trap: a duplicated side effect must fail, not silently succeed."""
    world = World(17)
    world.issue_refund("ORD-1001")
    with pytest.raises(ValueError, match="already refunded"):
        world.issue_refund("ORD-1001")


def test_rejected_refund_changes_nothing():
    """A refused action must leave the world byte-identical, not partially applied."""
    world = World(17)
    already_refunded = next(o.id for o in world.orders.values() if o.status == "refunded")
    before = snapshot(World(17))
    with pytest.raises(ValueError):
        world.issue_refund(already_refunded)
    assert snapshot(world) == before


def test_refund_touches_only_the_named_order():
    world = World(17)
    before = snapshot(World(17))
    world.issue_refund("ORD-1001")
    after_orders, after_tickets = snapshot(world)
    assert after_tickets == before[1]
    changed = {oid for oid in after_orders if after_orders[oid] != before[0][oid]}
    assert changed == {"ORD-1001"}


# --- tickets ------------------------------------------------------------


def test_get_ticket_raises_for_unknown_id():
    with pytest.raises(KeyError):
        World(17).get_ticket("TIC-9999")


def test_close_ticket_marks_it_closed():
    world = World(17)
    assert world.close_ticket("TIC-5001").status == "closed"


def test_second_close_is_rejected():
    world = World(17)
    already_closed = next(t.id for t in world.tickets.values() if t.status == "closed")
    with pytest.raises(ValueError, match="already closed"):
        world.close_ticket(already_closed)


def test_add_note_appends_in_order():
    world = World(17)
    world.add_note("TIC-5001", "first")
    world.add_note("TIC-5001", "second")
    assert world.get_ticket("TIC-5001").notes == ["first", "second"]


def test_empty_note_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        World(17).add_note("TIC-5001", "   ")
