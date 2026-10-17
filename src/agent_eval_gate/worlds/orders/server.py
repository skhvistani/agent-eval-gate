"""MCP server exposing the orders mock world over stdio.

The seed is fixed at process start (argv or ORDERS_WORLD_SEED), never at
runtime: the agent under test must not be able to reset the world it is
being scored on. The runner gets a clean world by launching a new process.

This layer holds no business logic. It translates the store's ordinary Python
exceptions into ToolError, which is what puts a readable reason in front of
the model instead of an opaque "Error executing tool".

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


# --- orders -------------------------------------------------------------


@server.tool()
def list_orders() -> list[dict]:
    """List every order with its customer, amount in cents, and status."""
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


# --- tickets ------------------------------------------------------------


@server.tool()
def list_tickets() -> list[dict]:
    """List every support ticket with the order it concerns and its status."""
    return [asdict(ticket) for ticket in world.tickets.values()]


@server.tool()
def get_ticket(ticket_id: str) -> dict:
    """Look up a single support ticket by its id, e.g. "TIC-5001"."""
    try:
        return asdict(world.get_ticket(ticket_id))
    except KeyError:
        raise ToolError(f"No ticket with id {ticket_id}") from None


@server.tool()
def close_ticket(ticket_id: str) -> dict:
    """Close a support ticket. Fails if the ticket is already closed."""
    try:
        return asdict(world.close_ticket(ticket_id))
    except KeyError:
        raise ToolError(f"No ticket with id {ticket_id}") from None
    except ValueError as exc:
        raise ToolError(str(exc)) from None


@server.tool()
def add_note(ticket_id: str, note: str) -> dict:
    """Append a note to a support ticket."""
    try:
        return asdict(world.add_note(ticket_id, note))
    except KeyError:
        raise ToolError(f"No ticket with id {ticket_id}") from None
    except ValueError as exc:
        raise ToolError(str(exc)) from None


# --- harness read channel -----------------------------------------------
#
# MCP separates tools (what the agent may do) from resources (what a client
# may read). Our adapter converts only tools into model-visible definitions,
# so this resource gives the scorer the final world state over the same
# connection without ever appearing in the agent's tool list.


@server.resource("world://state", mime_type="application/json")
def world_state() -> dict:
    """Full world state, for end-state scoring. Never exposed to the agent."""
    return world.snapshot()


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
