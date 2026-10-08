"""End-to-end smoke test: launch the MCP server as a subprocess and talk to it.

This is the test that proves an arbitrary MCP client can reach the world.
Everything else about the world is tested directly in test_store.py, where it
costs microseconds instead of a process launch.
"""

import json
import sys
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

SEED = 17


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def running_server(seed: int = SEED):
    """Launch the server on stdio and yield an initialized client session."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_eval_gate.worlds.orders.server", "--seed", str(seed)],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def payload(result):
    """Parse the JSON a successful tool call returns."""
    return json.loads(result.content[0].text)


@pytest.mark.anyio
async def test_server_exposes_the_expected_tools():
    async with running_server() as session:
        names = {tool.name for tool in (await session.list_tools()).tools}
    assert names == {"list_orders", "get_order", "issue_refund"}


@pytest.mark.anyio
async def test_reset_is_not_exposed_as_a_tool():
    """The agent under test must not be able to reset the world it is scored on."""
    async with running_server() as session:
        names = {tool.name for tool in (await session.list_tools()).tools}
    assert not {name for name in names if "reset" in name}


@pytest.mark.anyio
async def test_get_order_returns_the_seeded_order():
    async with running_server() as session:
        result = await session.call_tool("get_order", {"order_id": "ORD-1001"})
    assert not result.is_error
    assert payload(result)["id"] == "ORD-1001"
    assert payload(result)["status"] == "paid"


@pytest.mark.anyio
async def test_seed_is_honored_over_stdio():
    """Two processes on the same seed must serve byte-identical worlds."""
    async with running_server(SEED) as session:
        first = payload(await session.call_tool("list_orders", {}))
    async with running_server(SEED) as session:
        second = payload(await session.call_tool("list_orders", {}))
    assert first == second


@pytest.mark.anyio
async def test_double_refund_returns_a_readable_error():
    """The trap must reach the model as text it can act on, not an opaque failure."""
    async with running_server() as session:
        ok = await session.call_tool("issue_refund", {"order_id": "ORD-1001"})
        assert not ok.is_error
        assert payload(ok)["status"] == "refunded"

        again = await session.call_tool("issue_refund", {"order_id": "ORD-1001"})

    assert again.is_error
    assert "already refunded" in again.content[0].text


@pytest.mark.anyio
async def test_unknown_order_returns_a_readable_error():
    async with running_server() as session:
        result = await session.call_tool("get_order", {"order_id": "ORD-9999"})
    assert result.is_error
    assert "ORD-9999" in result.content[0].text
