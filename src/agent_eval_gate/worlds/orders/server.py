"""MCP server exposing the orders mock world over stdio.

The seed is fixed at process start (argv or ORDERS_WORLD_SEED), never at
runtime: the agent under test must not be able to reset the world it is
being scored on. The runner gets a clean world by launching a new process.

Run directly:  python -m agent_eval_gate.worlds.orders.server --seed 17
"""

import argparse
import os
from dataclasses import asdict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from agent_eval_gate.worlds.orders.store import World

DEFAULT_SEED = 17

server = MCPServer(name="orders-world")
world = World(DEFAULT_SEED)


@server.tool()
def list_orders() -> list[dict]:
    """List every order in the system with its current status."""
    return [asdict(order) for order in world.orders.values()]


@server.tool()
def get_order(order_id: str) -> dict:
    """Look up a single order by its id, e.g. "ORD-1001"."""
    try:
        return asdict(world.get_order(order_id))
    except KeyError:
        raise ToolError(f"No order with id {order_id}") from None


@server.tool()
def issue_refund(order_id: str) -> dict:
    """Refund an order. Fails if the order has already been refunded."""
    try:
        return asdict(world.issue_refund(order_id))
    except KeyError:
        raise ToolError(f"No order with id {order_id}") from None
    except ValueError as exc:
        raise ToolError(str(exc)) from None


def main() -> None:
    parser = argparse.ArgumentParser(description="Orders mock world as an MCP server.")
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("ORDERS_WORLD_SEED", DEFAULT_SEED)),
        help="Seed for deterministic world generation.",
    )
    args = parser.parse_args()
    world.reset(args.seed)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
