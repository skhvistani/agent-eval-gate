"""Claude adapter: a hand-written tool-use loop on the Messages API.

Deliberately not the SDK's tool runner. The loop is the thing being
instrumented — every model call, every tool call, every error has to land in
the trace — and owning it makes that explicit rather than a callback.

The loop, in four steps:

  1. Send `messages` plus `tools`.
  2. If `stop_reason == "tool_use"`, the response contains one or more
     `tool_use` blocks: {id, name, input}.
  3. Append the *entire* response content to the history (so tool_use and
     thinking blocks survive), run each tool, and append ONE user message
     whose content is the list of matching `tool_result` blocks.
  4. Repeat until the model stops asking, or the step cap is hit.

Two things that are easy to get wrong and are load-bearing here:

  - All tool results for one turn go back in a single user message. Splitting
    them teaches the model to stop making parallel calls.
  - A failed tool returns a `tool_result` with `is_error: True`, never a
    dropped block. An agent that cannot read why it failed cannot recover,
    and the faithfulness judge needs the failure text in the trace.
"""

import json
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from agent_eval_gate.trace import Step, ToolCall, Trace

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_STEPS = 12
STATE_RESOURCE = "world://state"

SYSTEM_PROMPT = """\
You are a customer support agent for an online store. You have tools for \
looking up orders and support tickets, issuing refunds, adding notes, and \
closing tickets.

Rules:
- Check the current state before acting. Do not assume.
- Only take actions the request actually calls for.
- If a request is ambiguous, ask which record is meant instead of guessing.
- If an action cannot be completed, say so plainly. Never report an action \
you did not successfully perform.

When you are done, reply with a short summary of exactly what you did."""


def describe_exception(exc: BaseException) -> str:
    """Flatten an exception into something readable in a results table.

    The MCP client runs inside an anyio task group, so anything raised in the
    loop resurfaces as `ExceptionGroup: unhandled errors in a TaskGroup`,
    which names no cause at all. Across a few hundred runs that is useless,
    so we unwrap to the leaf exceptions.
    """
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(describe_exception(sub) for sub in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def orders_world(seed: int):
    """Launch the orders MCP server on a fresh process and yield a session."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_eval_gate.worlds.orders.server", "--seed", str(seed)],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def anthropic_tools(session: ClientSession) -> list[dict]:
    """Convert MCP tool definitions into Anthropic tool definitions.

    Only tools are converted. Resources stay invisible to the model, which is
    what keeps the harness's state-read channel out of the agent's reach.
    """
    listed = await session.list_tools()
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.input_schema,
        }
        for tool in listed.tools
    ]


async def read_world_state(session: ClientSession) -> dict:
    """Read the server's state resource for end-state scoring."""
    result = await session.read_resource(STATE_RESOURCE)
    return json.loads(result.contents[0].text)


async def run_tool(session: ClientSession, name: str, arguments: dict) -> ToolCall:
    """Execute one tool call and record it, turning failures into readable text."""
    started = time.perf_counter()
    try:
        result = await session.call_tool(name, arguments)
        text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
        is_error = bool(result.is_error)
    except Exception as exc:  # transport-level failure, not a tool refusal
        text = f"Tool call failed: {exc}"
        is_error = True
    return ToolCall(
        id="",  # filled in by the caller, which owns the tool_use id
        name=name,
        arguments=arguments,
        result=text,
        is_error=is_error,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


async def run_scenario(
    scenario,
    *,
    client: AsyncAnthropic | None = None,
    model: str = DEFAULT_MODEL,
    run_index: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Trace:
    """Run one scenario end to end and return its trace."""
    client = client or AsyncAnthropic()
    trace = Trace(
        scenario_id=scenario.id,
        seed=scenario.seed,
        model=model,
        run_index=run_index,
        started_at=datetime.now(UTC),
    )
    started = time.perf_counter()

    try:
        async with orders_world(scenario.seed) as session:
            tools = await anthropic_tools(session)
            trace.initial_state = await read_world_state(session)
            messages: list[dict] = [{"role": "user", "content": scenario.task}]

            for index in range(max_steps):
                call_started = time.perf_counter()
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=messages,
                )
                step = Step(
                    index=index,
                    text="".join(b.text for b in response.content if b.type == "text"),
                    stop_reason=response.stop_reason,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=(time.perf_counter() - call_started) * 1000,
                )
                trace.stop_reason = response.stop_reason

                if response.stop_reason != "tool_use":
                    trace.steps.append(step)
                    trace.final_message = step.text
                    break

                # The whole content block list goes back, not just the text.
                messages.append({"role": "assistant", "content": response.content})

                results = []
                for block in (b for b in response.content if b.type == "tool_use"):
                    call = await run_tool(session, block.name, dict(block.input))
                    call.id = block.id
                    step.tool_calls.append(call)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": call.result,
                            "is_error": call.is_error,
                        }
                    )

                # All results for one turn in ONE user message.
                messages.append({"role": "user", "content": results})
                trace.steps.append(step)
            else:
                trace.hit_step_limit = True

            trace.final_state = await read_world_state(session)

    except BaseException as exc:
        # BaseException, not Exception: BaseExceptionGroup is not an Exception
        # subclass unless every member is, so a plain `except Exception` here
        # lets harness crashes escape and take the whole suite down.
        trace.error = describe_exception(exc)

    trace.duration_ms = (time.perf_counter() - started) * 1000
    return trace
