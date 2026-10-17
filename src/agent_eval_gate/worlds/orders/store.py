"""Orders mock world: an in-memory, seeded, resettable store.

Deterministic by construction: all randomness comes from one
random.Random(seed), IDs are sequential, and there is no clock.

The seed world deliberately contains orders that are *already refunded* and
tickets that are *already closed*. Without them a refusal trap cannot be
expressed, and refusal is a scoring category in its own right.
"""

import random
from dataclasses import asdict, dataclass, field

CUSTOMERS = ["alice", "bob", "carol", "dave", "erin"]
SUBJECTS = [
    "Item arrived damaged",
    "Wrong size delivered",
    "Package never arrived",
    "Requesting a refund",
    "Double charged",
]

N_ORDERS = 10
N_PRE_REFUNDED = 2
N_TICKETS = 6
N_PRE_CLOSED = 2

FIRST_ORDER_ID = 1001
FIRST_TICKET_ID = 5001


@dataclass
class Order:
    id: str
    customer: str
    amount: int  # cents, so equality is exact
    status: str  # "paid" | "refunded"


@dataclass
class Ticket:
    id: str
    order_id: str
    subject: str
    status: str  # "open" | "closed"
    # default_factory, not [], or every Ticket would share one list.
    notes: list[str] = field(default_factory=list)


class World:
    def __init__(self, seed: int) -> None:
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.orders: dict[str, Order] = {}
        self.tickets: dict[str, Ticket] = {}
        self._generate_orders()
        self._generate_tickets()

    # --- generation -----------------------------------------------------

    def _generate_orders(self) -> None:
        for i in range(N_ORDERS):
            order_id = f"ORD-{FIRST_ORDER_ID + i}"
            self.orders[order_id] = Order(
                id=order_id,
                customer=self.rng.choice(CUSTOMERS),
                amount=self.rng.randint(500, 20000),
                status="paid",
            )
        # sample over a sorted list: dict order is insertion-stable, but being
        # explicit keeps determinism obvious to a reader.
        for order_id in self.rng.sample(sorted(self.orders), N_PRE_REFUNDED):
            self.orders[order_id].status = "refunded"

    def _generate_tickets(self) -> None:
        order_ids = self.rng.sample(sorted(self.orders), N_TICKETS)
        for i, order_id in enumerate(order_ids):
            ticket_id = f"TIC-{FIRST_TICKET_ID + i}"
            self.tickets[ticket_id] = Ticket(
                id=ticket_id,
                order_id=order_id,
                subject=self.rng.choice(SUBJECTS),
                status="open",
            )
        for ticket_id in self.rng.sample(sorted(self.tickets), N_PRE_CLOSED):
            self.tickets[ticket_id].status = "closed"

    # --- scoring view ---------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, dict]]:
        """World state as plain data: {entity: {record_id: {field: value}}}.

        The single serialisation used by both the MCP state resource and the
        end-state scorer, so the two can never drift apart.
        """
        return {
            "orders": {oid: asdict(order) for oid, order in self.orders.items()},
            "tickets": {tid: asdict(ticket) for tid, ticket in self.tickets.items()},
        }

    # --- orders ---------------------------------------------------------

    def get_order(self, order_id: str) -> Order:
        return self.orders[order_id]

    def issue_refund(self, order_id: str) -> Order:
        order = self.get_order(order_id)
        if order.status == "refunded":
            raise ValueError(f"{order_id} already refunded")
        order.status = "refunded"
        return order

    # --- tickets --------------------------------------------------------

    def get_ticket(self, ticket_id: str) -> Ticket:
        return self.tickets[ticket_id]

    def close_ticket(self, ticket_id: str) -> Ticket:
        ticket = self.get_ticket(ticket_id)
        if ticket.status == "closed":
            raise ValueError(f"{ticket_id} already closed")
        ticket.status = "closed"
        return ticket

    def add_note(self, ticket_id: str, note: str) -> Ticket:
        ticket = self.get_ticket(ticket_id)
        if not note.strip():
            raise ValueError("note must not be empty")
        ticket.notes.append(note)
        return ticket
