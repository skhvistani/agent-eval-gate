"""Judge calibration: how often does the judge agree with a human?

A judge is a model and can be wrong. Publishing its agreement with hand
labels is what makes it trustworthy enough to gate on. Two numbers:

  accuracy   Fraction of cases where judge and human agree. Easy to read,
             easy to inflate: if 90% of cases are faithful, a judge that
             says "faithful" to everything scores 90%.
  kappa      Cohen's kappa: agreement corrected for the agreement two
             labellers would reach by chance. 0 is chance, 1 is perfect,
             and the usual reading is >0.6 substantial, >0.8 near-perfect.

Both are reported, plus the confusion matrix, because a single number hides
*which way* the judge is wrong. A judge that misses unfaithful reports is a
much bigger problem for a gate than one that is over-suspicious.

Labels live in a YAML file a human writes by hand after reading the final
message and the tool log — never the expected answer, which would turn a
faithfulness label into a correctness label.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class HumanLabel(BaseModel):
    """One hand-labelled (final message, tool log) pair."""

    trace_id: str = Field(description="scenario_id:run_index, matching a saved trace.")
    faithful: bool
    note: str = Field(default="", description="Why, in the labeller's words.")


class Agreement(BaseModel):
    n: int
    accuracy: float
    kappa: float
    true_positive: int = Field(description="Both say unfaithful.")
    true_negative: int = Field(description="Both say faithful.")
    false_positive: int = Field(description="Judge says unfaithful, human says faithful.")
    false_negative: int = Field(description="Judge says faithful, human says unfaithful.")

    @property
    def missed_unfaithful_rate(self) -> float:
        """Of the reports a human called unfaithful, how many did the judge miss?

        The number that matters most for a gate.
        """
        unfaithful = self.true_positive + self.false_negative
        return self.false_negative / unfaithful if unfaithful else 0.0

    @property
    def summary(self) -> str:
        return (
            f"n={self.n} accuracy={self.accuracy:.2f} kappa={self.kappa:.2f} "
            f"missed_unfaithful={self.missed_unfaithful_rate:.2f} "
            f"[tp={self.true_positive} tn={self.true_negative} "
            f"fp={self.false_positive} fn={self.false_negative}]"
        )


def load_labels(path: Path) -> list[HumanLabel]:
    raw = yaml.safe_load(path.read_text()) or []
    return [HumanLabel(**item) for item in raw]


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Kappa for binary labels. `pairs` is (judge_says, human_says)."""
    n = len(pairs)
    if n == 0:
        return 0.0
    observed = sum(a == b for a, b in pairs) / n
    judge_yes = sum(a for a, _ in pairs) / n
    human_yes = sum(b for _, b in pairs) / n
    expected = judge_yes * human_yes + (1 - judge_yes) * (1 - human_yes)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def agreement(judge: dict[str, bool], human: dict[str, bool]) -> Agreement:
    """Compare judge verdicts to human labels on the traces both have.

    Both dicts map trace_id -> faithful. Traces present in only one are
    ignored, so a partially-labelled set can still be scored.
    """
    ids = sorted(set(judge) & set(human))
    pairs = [(judge[i], human[i]) for i in ids]

    tp = sum(1 for j, h in pairs if not j and not h)
    tn = sum(1 for j, h in pairs if j and h)
    fp = sum(1 for j, h in pairs if not j and h)
    fn = sum(1 for j, h in pairs if j and not h)

    return Agreement(
        n=len(pairs),
        accuracy=(tp + tn) / len(pairs) if pairs else 0.0,
        kappa=cohens_kappa(pairs),
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )
