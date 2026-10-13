"""The tool-use loop, exercised against a scripted fake model.

No API calls: the loop's mechanics are what matter here, and they have to be
testable in CI without a key or a bill. The one thing a fake cannot check is
whether a real model behaves sensibly, which is what the evaluation run is
for.
"""

from types import SimpleNamespace

import pytest

from agent_eval_gate.adapters.claude import (
    anthropic_tools,
    orders_world,
    read_world_state,
    run_scenario,
)
from agent_eval_gate.scenarios.schema import Category, Scenario


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --- fake model ---------------------------------------------------------


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(block_id, name, args):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=args)


def response(blocks, stop_reason):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


class FakeClient:
    """Replays a scripted list of responses and records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.script:
            return self.script.pop(0)
        return response([text_block("done")], "end_turn")


def scenario(task="do the thing", seed=17):
    return Scenario(id="t-001", seed=seed, task=task, category=Category.SINGLE_ACTION)


# --- MCP wiring ---------------------------------------------------------


@pytest.mark.anyio
async def test_tool_definitions_have_the_shape_the_api_expects():
    async with orders_world(17) as session:
        tools = await anthropic_tools(session)
    assert tools
    for tool in tools:
        assert set(tool) == {"name", "description", "input_schema"}
        assert tool["description"], f"{tool['name']} has no description for the model to read"
        assert tool["input_schema"]["type"] == "object"


@pytest.mark.anyio
async def test_state_resource_is_readable_but_not_a_tool():
    """The harness reads world state over a channel the agent never sees."""
    async with orders_world(17) as session:
        tool_names = {tool["name"] for tool in await anthropic_tools(session)}
        state = await read_world_state(session)
    assert "orders" in state and "tickets" in state
    assert not any("state" in name for name in tool_names)


# --- the loop -----------------------------------------------------------


@pytest.mark.anyio
async def test_loop_stops_on_end_turn_and_records_the_final_message():
    client = FakeClient([response([text_block("All done.")], "end_turn")])
    trace = await run_scenario(scenario(), client=client)
    assert trace.error is None
    assert trace.final_message == "All done."
    assert len(trace.steps) == 1
    assert trace.tool_calls == []


@pytest.mark.anyio
async def test_loop_executes_a_tool_and_feeds_the_result_back():
    client = FakeClient(
        [
            response([tool_block("tu_1", "get_order", {"order_id": "ORD-1001"})], "tool_use"),
            response([text_block("ORD-1001 is paid.")], "end_turn"),
        ]
    )
    trace = await run_scenario(scenario(), client=client)

    assert len(trace.tool_calls) == 1
    call = trace.tool_calls[0]
    assert call.name == "get_order" and call.id == "tu_1" and not call.is_error
    assert "ORD-1001" in call.result
    assert trace.final_message == "ORD-1001 is paid."


@pytest.mark.anyio
async def test_parallel_results_go_back_in_one_user_message():
    """Splitting them teaches the model to stop making parallel calls."""
    client = FakeClient(
        [
            response(
                [
                    tool_block("tu_1", "get_order", {"order_id": "ORD-1001"}),
                    tool_block("tu_2", "get_order", {"order_id": "ORD-1002"}),
                ],
                "tool_use",
            ),
            response([text_block("Both checked.")], "end_turn"),
        ]
    )
    trace = await run_scenario(scenario(), client=client)

    assert len(trace.tool_calls) == 2
    sent = client.requests[-1]["messages"]
    results = [m for m in sent if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(results) == 1
    assert [b["tool_use_id"] for b in results[0]["content"]] == ["tu_1", "tu_2"]


@pytest.mark.anyio
async def test_assistant_turn_is_echoed_back_whole():
    """The tool_use blocks must survive in history or the next call is invalid."""
    blocks = [tool_block("tu_1", "get_order", {"order_id": "ORD-1001"})]
    client = FakeClient([response(blocks, "tool_use"), response([text_block("ok")], "end_turn")])
    await run_scenario(scenario(), client=client)

    sent = client.requests[-1]["messages"]
    assistant = [m for m in sent if m["role"] == "assistant"]
    assert assistant and assistant[0]["content"] is blocks


@pytest.mark.anyio
async def test_failed_tool_is_recorded_and_the_loop_continues():
    """A refusal must reach the model as readable text, not a dropped block."""
    client = FakeClient(
        [
            response([tool_block("tu_1", "get_order", {"order_id": "ORD-9999"})], "tool_use"),
            response([text_block("No such order.")], "end_turn"),
        ]
    )
    trace = await run_scenario(scenario(), client=client)

    call = trace.tool_calls[0]
    assert call.is_error
    assert "ORD-9999" in call.result

    sent = client.requests[-1]["messages"]
    result_block = sent[-1]["content"][0]
    assert result_block["is_error"] is True
    assert "ORD-9999" in result_block["content"]


@pytest.mark.anyio
async def test_step_limit_is_recorded_as_a_failure():
    """A truncated run is a failed run, never a quiet pass."""
    loop_forever = [
        response([tool_block(f"tu_{i}", "list_orders", {})], "tool_use") for i in range(10)
    ]
    trace = await run_scenario(scenario(), client=FakeClient(loop_forever), max_steps=3)

    assert trace.hit_step_limit
    assert len(trace.steps) == 3


@pytest.mark.anyio
async def test_mutations_land_in_the_final_state():
    client = FakeClient(
        [
            response([tool_block("tu_1", "issue_refund", {"order_id": "ORD-1001"})], "tool_use"),
            response([text_block("Refunded.")], "end_turn"),
        ]
    )
    trace = await run_scenario(scenario(), client=client)
    assert trace.final_state["orders"]["ORD-1001"]["status"] == "refunded"


@pytest.mark.anyio
async def test_each_run_starts_from_a_clean_world():
    """k runs of one scenario must not contaminate each other."""
    script = [
        response([tool_block("tu_1", "issue_refund", {"order_id": "ORD-1001"})], "tool_use"),
        response([text_block("Refunded.")], "end_turn"),
    ]
    first = await run_scenario(scenario(), client=FakeClient(script))
    second = await run_scenario(scenario(), client=FakeClient(list(script)))

    assert first.final_state["orders"]["ORD-1001"]["status"] == "refunded"
    assert second.tool_calls[0].is_error is False
    assert second.final_state == first.final_state


@pytest.mark.anyio
async def test_token_usage_is_summed_across_steps():
    client = FakeClient(
        [
            response([tool_block("tu_1", "list_orders", {})], "tool_use"),
            response([text_block("ok")], "end_turn"),
        ]
    )
    trace = await run_scenario(scenario(), client=client)
    assert trace.input_tokens == 200
    assert trace.output_tokens == 40


@pytest.mark.anyio
async def test_harness_failure_is_recorded_not_raised():
    """A crash must produce a trace marked failed, not kill the whole suite."""

    class Exploding(FakeClient):
        async def _create(self, **kwargs):
            raise RuntimeError("connection reset")

    trace = await run_scenario(scenario(), client=Exploding([]))
    assert trace.error is not None
    assert "connection reset" in trace.error
