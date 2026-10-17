"""The trace: the complete record of one agent run.

This is the raw evidence every scorer reads, so it is defined once here and
is the only thing a scorer is allowed to look at:

  - end-state diff     reads `final_state`
  - trajectory rules   read `tool_calls` (names, arguments, duplicates)
  - faithfulness judge reads `final_message` against `tool_calls`

Keeping it a pydantic model rather than loose dicts means a trace can be
written to disk, reloaded weeks later, and still be the same shape the
scorers expect.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One tool invocation: what the model asked for and what came back."""

    id: str = Field(description="The tool_use block id, used to match the result.")
    name: str
    arguments: dict[str, Any]
    result: str = Field(description="Raw text returned to the model.")
    is_error: bool = Field(
        default=False,
        description="True when the tool refused or failed. The model still sees the text.",
    )
    latency_ms: float


class Step(BaseModel):
    """One turn of the loop: a single model call plus the tools it requested."""

    index: int
    text: str = Field(default="", description="Assistant prose emitted in this turn.")
    tool_calls: list[ToolCall] = Field(default=[])
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class Trace(BaseModel):
    """Everything that happened in one run of one scenario."""

    scenario_id: str
    seed: int
    model: str
    run_index: int = Field(default=0, description="Which of the k runs this is.")
    started_at: datetime

    steps: list[Step] = Field(default=[])
    final_message: str = Field(
        default="",
        description="The agent's closing claim. The faithfulness judge scores this.",
    )
    read_only_tools: list[str] = Field(
        default=[],
        description="Tools the server declared read-only (MCP readOnlyHint). Recorded "
        "so the trajectory scorer can tell a lookup from a side effect without "
        "hardcoding tool names for a particular world.",
    )
    initial_state: dict[str, Any] = Field(
        default={},
        description="World state before the agent ran. Recorded so the end-state "
        "scorer can diff without knowing anything about which world this is.",
    )
    final_state: dict[str, Any] = Field(
        default={},
        description="World state after the run, read from the server's state resource.",
    )

    stop_reason: str | None = None
    hit_step_limit: bool = Field(
        default=False,
        description="True when the loop was cut off, which is a failed run, not a pass.",
    )
    error: str | None = Field(
        default=None,
        description="Harness failure (network, crash). Distinct from an agent failure.",
    )
    duration_ms: float = 0.0

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Every tool call across all steps, in order."""
        return [call for step in self.steps for call in step.tool_calls]

    @property
    def input_tokens(self) -> int:
        return sum(step.input_tokens for step in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(step.output_tokens for step in self.steps)
