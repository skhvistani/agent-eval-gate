"""The scenario schema: one test case for a tool-using agent.

A scenario is arrange / act / assert. It names a seed world (arrange), the task
handed to the agent (act), and what must be true afterwards (assert).

The assertion is deliberately a *change set*, not a world snapshot. We record
only what should differ from the seed world; the scorer verifies those changes
happened and that nothing else moved. Two consequences:

  - Agents may reach the end state by any route. We never match trajectories.
  - A refusal trap is just a scenario with an empty change set: the correct
    behaviour is that the world is untouched. No special case needed.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class Category(StrEnum):
    """Scenario kinds, reported as separate rows in the results table.

    An enum rather than a free string: a typo must not silently create a
    sixth category and split the per-category pass rates.
    """

    LOOKUP = "lookup"
    SINGLE_ACTION = "single_action"
    MULTI_STEP = "multi_step"
    REFUSAL_TRAP = "refusal_trap"
    AMBIGUITY = "ambiguity"


class ExpectedChange(BaseModel):
    """One field that must differ from the seed world once the agent is done.

    `value` omitted means "this field must change, but the value is not
    checked". That is for free text an agent writes itself, such as a note on
    a ticket: requiring it to change is meaningful, requiring exact wording
    is not.
    """

    entity: str = Field(description='Collection name, e.g. "orders".')
    id: str = Field(description='Record id within the collection, e.g. "ORD-1001".')
    field: str = Field(description='Attribute that must change, e.g. "status".')
    value: str | int | None = Field(
        default=None,
        description='Value it must hold at the end, e.g. "refunded". '
        "Omit to require only that the field changed.",
    )

    @property
    def checks_value(self) -> bool:
        return self.value is not None


class ForbiddenAction(BaseModel):
    """A tool call that must never appear in the trace.

    With `args` omitted the tool is banned outright. With `args` set, only that
    argument combination is banned, so a scenario can allow `issue_refund` on
    one order while forbidding it on another.
    """

    tool: str = Field(description="Tool name as exposed by the MCP server.")
    args: dict | None = Field(
        default=None,
        description="Ban only this argument combination. None bans the tool entirely.",
    )


class Scenario(BaseModel):
    """One evaluation case, run k times against a freshly seeded world."""

    id: str = Field(description='Stable identifier, e.g. "refund-001".')
    seed: int = Field(description="Seed the world is generated from before each run.")
    task: str = Field(description="The instruction handed to the agent.")
    category: Category

    expected_changes: list[ExpectedChange] = Field(
        default=[],
        description="Everything that must differ from the seed world. Empty means the "
        "world must be untouched, which is how refusal traps are expressed.",
    )
    forbidden_actions: list[ForbiddenAction] = Field(
        default=[],
        description="Tool calls that must not appear in the trace.",
    )
    reference_answer: str | None = Field(
        default=None,
        description="Optional gold final message, used when calibrating the judge.",
    )
    tags: list[str] = Field(default=[], description="Free-form labels for slicing results.")
