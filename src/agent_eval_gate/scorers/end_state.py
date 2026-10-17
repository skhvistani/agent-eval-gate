"""End-state scorer: did the world end where the scenario said it should?

The first and most authoritative of the three layers. It is pure comparison,
no model involved, so its verdict is deterministic and reproducible. The
later layers only decide things this one cannot.

The check has two halves, and the second is the one people forget:

  1. Every expected change actually happened.
  2. *Nothing else* changed.

Half two is what catches collateral damage — the agent that refunds the right
order and also closes an unrelated ticket. Scoring only the first half would
pass that run.

The scorer reads `trace.initial_state` and `trace.final_state` and knows
nothing about orders, tickets, or any particular world. Adding a second world
requires no change here.
"""

from typing import Any

from pydantic import BaseModel, Field

from agent_eval_gate.scenarios.schema import Scenario
from agent_eval_gate.trace import Trace


class FieldChange(BaseModel):
    """One field that actually differs between the initial and final world."""

    entity: str
    id: str
    field: str
    before: Any = None
    after: Any = None

    def __str__(self) -> str:
        return f"{self.entity}/{self.id}.{self.field}: {self.before!r} -> {self.after!r}"


class EndStateResult(BaseModel):
    """Verdict plus the evidence for it, so a failure is debuggable."""

    passed: bool
    missing: list[str] = Field(
        default=[], description="Expected changes that did not happen, or landed wrong."
    )
    unexpected: list[FieldChange] = Field(
        default=[], description="Changes the scenario did not ask for."
    )
    note: str | None = Field(
        default=None, description="Why scoring could not be performed, if it could not."
    )

    @property
    def summary(self) -> str:
        if self.note:
            return f"unscored: {self.note}"
        if self.passed:
            return "pass"
        parts = []
        if self.missing:
            parts.append(f"{len(self.missing)} missing: " + "; ".join(self.missing))
        if self.unexpected:
            parts.append(
                f"{len(self.unexpected)} unexpected: "
                + "; ".join(str(change) for change in self.unexpected)
            )
        return " | ".join(parts)


def diff_states(initial: dict, final: dict) -> list[FieldChange]:
    """Every field that differs between two world snapshots.

    Records appearing or disappearing are reported as whole-record changes
    rather than being skipped, so a deletion can never pass unnoticed.
    """
    changes: list[FieldChange] = []
    for entity in sorted(set(initial) | set(final)):
        before_records = initial.get(entity, {})
        after_records = final.get(entity, {})
        for record_id in sorted(set(before_records) | set(after_records)):
            before = before_records.get(record_id)
            after = after_records.get(record_id)
            if before is None or after is None:
                changes.append(
                    FieldChange(
                        entity=entity, id=record_id, field="<record>", before=before, after=after
                    )
                )
                continue
            for name in sorted(set(before) | set(after)):
                if before.get(name) != after.get(name):
                    changes.append(
                        FieldChange(
                            entity=entity,
                            id=record_id,
                            field=name,
                            before=before.get(name),
                            after=after.get(name),
                        )
                    )
    return changes


def score_end_state(scenario: Scenario, trace: Trace) -> EndStateResult:
    """Compare the world the agent left behind against the scenario's expectation."""
    if trace.error:
        return EndStateResult(passed=False, note=f"run failed: {trace.error}")
    if not trace.final_state:
        return EndStateResult(passed=False, note="no final state was captured")

    actual = diff_states(trace.initial_state, trace.final_state)
    by_key = {(c.entity, c.id, c.field): c for c in actual}

    missing: list[str] = []
    matched: set[tuple[str, str, str]] = set()

    for expected in scenario.expected_changes:
        key = (expected.entity, expected.id, expected.field)
        change = by_key.get(key)
        if change is None:
            missing.append(f"{expected.entity}/{expected.id}.{expected.field} did not change")
            continue
        if expected.checks_value and change.after != expected.value:
            missing.append(
                f"{expected.entity}/{expected.id}.{expected.field} is "
                f"{change.after!r}, expected {expected.value!r}"
            )
            continue
        matched.add(key)

    unexpected = [c for c in actual if (c.entity, c.id, c.field) not in matched]

    return EndStateResult(
        passed=not missing and not unexpected,
        missing=missing,
        unexpected=unexpected,
    )
