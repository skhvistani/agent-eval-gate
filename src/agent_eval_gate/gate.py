"""The gate: does this run clear the baseline?

A report tells you a number. A gate makes a decision, and the decision has
to be conservative in the right direction: a change to a prompt, a tool, or
a model must not slip through because one run got lucky.

The rule:

    FAIL  when  lower_bound(candidate − baseline)  <  −tolerance

where the difference is a *paired* bootstrap over the scenarios both runs
share. Two runs on the same scenarios are a paired design; resampling the
same scenario for both sides cancels scenario-difficulty noise, so the
interval is about the change, not about which scenarios happened to be hard.

Why paired and not "candidate lower bound vs baseline point", which is how
the charter first phrased it: that rule compares an interval to a point, and
with few scenarios it cannot be cleared even by an identical run. On the
first five-scenario baseline the candidate interval was [88%, 100%] against
a 91% threshold — the same run, failing itself — while the paired difference
was exactly 0.0% [0.0%, 0.0%]. The paired rule says what is true: no
regression. The absolute rule is still reported, for context.

Reading the *lower bound*, not the point estimate, is the whole design in
either form. A candidate whose difference is centred on zero but whose
interval reaches well below it is not evidence of "no regression"; it is
evidence of "cannot tell", and the gate treats that as a failure to clear.
A narrower gate is earned with more scenarios, not a looser tolerance.

Two further checks, because the comparison is only meaningful if the two
runs measured the same thing:

  - The scenario sets must overlap substantially. A run on a different
    suite is not comparable, and the gate says so instead of guessing.
  - Harness errors are named in the verdict. A run where half the calls
    failed at the API has a low pass rate for a reason that has nothing to
    do with the agent, and that should be visible.
"""

from pydantic import BaseModel, Field

from agent_eval_gate.results import SuiteResult
from agent_eval_gate.stats import Estimate, bootstrap, paired_difference, pass_rate

DEFAULT_TOLERANCE = 0.05
MIN_SCENARIO_OVERLAP = 0.8


class GateVerdict(BaseModel):
    passed: bool
    reason: str

    tolerance: float
    paired: Estimate | None = Field(
        default=None, description="Candidate − baseline on shared scenarios. The gated quantity."
    )
    baseline_pass_rate: float
    candidate: Estimate = Field(description="Candidate pass rate on its own, for context.")
    candidate_errors: int
    scenario_overlap: float

    @property
    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [f"gate: {verdict} — {self.reason}"]
        if self.paired is not None:
            lines.append(
                f"  paired difference (candidate − baseline) {self.paired}; "
                f"lower bound must be ≥ −{self.tolerance:.1%}"
            )
        lines.append(
            f"  candidate {self.candidate}  ·  baseline {self.baseline_pass_rate:.1%}  ·  "
            f"scenario overlap {self.scenario_overlap:.0%}"
        )
        if self.candidate_errors:
            lines.append(f"  candidate had {self.candidate_errors} harness errors")
        return "\n".join(lines)


def evaluate(
    candidate: SuiteResult,
    baseline: SuiteResult,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    seed: int = 0,
) -> GateVerdict:
    """Decide whether `candidate` clears `baseline`."""
    cand_ids = {o.scenario_id for o in candidate.outcomes}
    base_ids = {o.scenario_id for o in baseline.outcomes}
    overlap = len(cand_ids & base_ids) / len(base_ids) if base_ids else 0.0

    estimate = bootstrap(candidate.outcomes, seed=seed)
    paired = paired_difference(candidate.outcomes, baseline.outcomes, seed=seed)
    errors = candidate.errors

    common = dict(
        tolerance=tolerance,
        paired=paired if cand_ids & base_ids else None,
        baseline_pass_rate=pass_rate(baseline.outcomes),
        candidate=estimate,
        candidate_errors=errors,
        scenario_overlap=overlap,
    )

    if not candidate.outcomes:
        return GateVerdict(passed=False, reason="candidate has no runs", **common)
    if overlap < MIN_SCENARIO_OVERLAP:
        return GateVerdict(
            passed=False,
            reason=f"candidate covers only {overlap:.0%} of baseline scenarios; not comparable",
            **common,
        )
    if paired.low < -tolerance:
        why = f"paired difference lower bound {paired.low:+.1%} is below −{tolerance:.1%}"
        if errors:
            why += f" ({errors} harness errors may account for some of this)"
        return GateVerdict(passed=False, reason=why, **common)

    return GateVerdict(
        passed=True,
        reason=f"paired difference lower bound {paired.low:+.1%} is within −{tolerance:.1%}",
        **common,
    )
