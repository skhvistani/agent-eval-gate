"""Results: one record per run, one file per suite execution.

A trace is the raw evidence. A result is the verdict on it: which layers
passed, and a short reason when one did not. Results are what the statistics
read and what the CI gate compares against a baseline, so they are small,
flat, and serialisable.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from agent_eval_gate.scenarios.schema import Category


class RunOutcome(BaseModel):
    """The verdict on one run of one scenario."""

    scenario_id: str
    category: Category
    run_index: int

    end_state: bool
    trajectory: bool
    faithfulness: bool | None = Field(
        default=None, description="None when the judge was not run (no key, or --no-judge)."
    )
    lexical: bool | None = Field(
        default=None, description="The no-model baseline's verdict, for calibration."
    )
    passed: bool = Field(description="Every layer that ran passed.")

    error: str | None = Field(default=None, description="Harness failure, if any.")
    detail: str = Field(default="", description="Why it failed, one line per layer.")

    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0


class SuiteResult(BaseModel):
    """Everything from one execution of the suite against one model."""

    model: str
    judge_model: str | None
    k: int
    started_at: datetime
    scenario_count: int
    outcomes: list[RunOutcome] = Field(default=[])

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def errors(self) -> int:
        return sum(1 for o in self.outcomes if o.error)

    @property
    def input_tokens(self) -> int:
        return sum(o.input_tokens for o in self.outcomes)

    @property
    def output_tokens(self) -> int:
        return sum(o.output_tokens for o in self.outcomes)
