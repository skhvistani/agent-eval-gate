"""Orders mock world: an in-memory, seeded, resettable store.

Deterministic by construction: all randomness comes from one
random.Random(seed), IDs are sequential, and there is no clock.
"""

import random
from dataclasses import dataclass

CUSTOMERS = ["alice", "bob", "carol", "dave", "erin"]
N_ORDERS = 10
FIRST_ORDER_ID = 1001


@dataclass
class Order:
    id: str
    customer: str
    amount: int  # cents, so equality is exact
    status: str  # "paid" | "refunded"


class World:
    def __init__(self, seed: int) -> None:
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.orders: dict[str, Order] = {}
        for i in range(N_ORDERS):
            order_id = f"ORD-{FIRST_ORDER_ID + i}"
            self.orders[order_id] = Order(
                id=order_id,
                customer=self.rng.choice(CUSTOMERS),
                amount=self.rng.randint(500, 20000),
                status="paid",
            )

    def get_order(self, order_id: str) -> Order:
        return self.orders[order_id]

    def issue_refund(self, order_id: str) -> Order:
        order = self.get_order(order_id)
        if order.status == "refunded":
            raise ValueError(f"{order_id} already refunded")
        order.status = "refunded"
        return order
