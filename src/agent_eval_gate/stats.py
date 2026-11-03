"""Pass rates with uncertainty.

One run of a stochastic agent is noise, not evidence. Every number here
carries an interval, and every interval comes from the same bootstrap:

    resample *scenarios* with replacement, recompute the statistic, repeat,
    take the 2.5th and 97.5th percentiles.

The unit of resampling is the scenario, not the run. The k runs of one
scenario share a task and are correlated; treating them as independent
samples would make the intervals too narrow. This is the clustered-error
argument from Miller (2024), applied.

Pure Python on purpose: a few thousand resamples over a few hundred runs is
nothing, and it keeps the dependency list short. Resampling uses its own
seeded `random.Random`, like the mock world, so a report is reproducible.
"""

import random
from collections import defaultdict
from collections.abc import Callable

from pydantic import BaseModel

from agent_eval_gate.results import RunOutcome

N_RESAMPLES = 2000
CI = 0.95


class Estimate(BaseModel):
    """A point estimate with its bootstrap confidence interval."""

    point: float
    low: float
    high: float
    n_scenarios: int
    n_runs: int

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"

    @property
    def width(self) -> float:
        return self.high - self.low


def by_scenario(outcomes: list[RunOutcome]) -> dict[str, list[RunOutcome]]:
    groups: dict[str, list[RunOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[outcome.scenario_id].append(outcome)
    return dict(groups)


def pass_rate(outcomes: list[RunOutcome]) -> float:
    return sum(o.passed for o in outcomes) / len(outcomes) if outcomes else 0.0


def pass_power_k(outcomes: list[RunOutcome]) -> float:
    """Fraction of scenarios where *every* run passed: reliability, not luck.

    tau-bench's pass^k. A 90% agent at k=5 is reliable only ~59% of the time.
    """
    groups = by_scenario(outcomes)
    if not groups:
        return 0.0
    return sum(all(o.passed for o in runs) for runs in groups.values()) / len(groups)


def bootstrap(
    outcomes: list[RunOutcome],
    statistic: Callable[[list[RunOutcome]], float] = pass_rate,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
) -> Estimate:
    """Resample scenarios with replacement and take percentiles of `statistic`."""
    groups = by_scenario(outcomes)
    scenario_ids = sorted(groups)
    if not scenario_ids:
        return Estimate(point=0.0, low=0.0, high=0.0, n_scenarios=0, n_runs=0)

    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        chosen = rng.choices(scenario_ids, k=len(scenario_ids))
        resampled = [o for sid in chosen for o in groups[sid]]
        samples.append(statistic(resampled))
    samples.sort()

    alpha = (1 - CI) / 2
    lo_idx = int(alpha * n_resamples)
    hi_idx = min(int((1 - alpha) * n_resamples), n_resamples - 1)
    return Estimate(
        point=statistic(outcomes),
        low=samples[lo_idx],
        high=samples[hi_idx],
        n_scenarios=len(scenario_ids),
        n_runs=len(outcomes),
    )


def per_category(outcomes: list[RunOutcome], **kwargs) -> dict[str, Estimate]:
    """One estimate per category, each bootstrapped within the category."""
    groups: dict[str, list[RunOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[outcome.category.value].append(outcome)
    return {category: bootstrap(runs, **kwargs) for category, runs in sorted(groups.items())}


def per_layer(outcomes: list[RunOutcome]) -> dict[str, Estimate]:
    """Pass rate of each scoring layer on its own, to see where failures come from."""
    layers: dict[str, Callable[[RunOutcome], bool | None]] = {
        "end_state": lambda o: o.end_state,
        "trajectory": lambda o: o.trajectory,
        "faithfulness": lambda o: o.faithfulness,
    }
    result = {}
    for name, pick in layers.items():
        scored = [o for o in outcomes if pick(o) is not None]
        if not scored:
            continue
        result[name] = bootstrap(
            scored, lambda runs, p=pick: sum(bool(p(o)) for o in runs) / len(runs)
        )
    return result


def paired_difference(
    a: list[RunOutcome],
    b: list[RunOutcome],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
) -> Estimate:
    """Pass rate of A minus B, resampling the *same* scenarios for both.

    Two models run on identical scenarios are a paired design. Bootstrapping
    the difference scenario-by-scenario cancels the scenario-difficulty noise
    that two independent intervals would each carry, which is why the paired
    interval is the one to read when asking "is A better than B".
    """
    groups_a, groups_b = by_scenario(a), by_scenario(b)
    shared = sorted(set(groups_a) & set(groups_b))
    if not shared:
        return Estimate(point=0.0, low=0.0, high=0.0, n_scenarios=0, n_runs=0)

    def diff(ids: list[str]) -> float:
        runs_a = [o for sid in ids for o in groups_a[sid]]
        runs_b = [o for sid in ids for o in groups_b[sid]]
        return pass_rate(runs_a) - pass_rate(runs_b)

    rng = random.Random(seed)
    samples = sorted(diff(rng.choices(shared, k=len(shared))) for _ in range(n_resamples))
    alpha = (1 - CI) / 2
    return Estimate(
        point=diff(shared),
        low=samples[int(alpha * n_resamples)],
        high=samples[min(int((1 - alpha) * n_resamples), n_resamples - 1)],
        n_scenarios=len(shared),
        n_runs=sum(len(groups_a[s]) + len(groups_b[s]) for s in shared),
    )
