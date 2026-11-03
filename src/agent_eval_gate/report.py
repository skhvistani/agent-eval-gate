"""Render a suite result as the markdown table the README publishes."""

from agent_eval_gate.results import SuiteResult
from agent_eval_gate.stats import bootstrap, pass_power_k, per_category, per_layer


def render(suite: SuiteResult) -> str:
    outcomes = suite.outcomes
    overall = bootstrap(outcomes)
    reliability = bootstrap(outcomes, pass_power_k)

    lines = [
        f"**Model:** `{suite.model}` · **Judge:** `{suite.judge_model or 'not run'}` · "
        f"**k = {suite.k}** · {suite.scenario_count} scenarios · {len(outcomes)} runs · "
        f"{suite.errors} harness errors",
        "",
        "| Metric | Estimate (95% bootstrap CI) |",
        "|---|---|",
        f"| Pass rate | {overall} |",
        f"| pass^{suite.k} (all runs pass) | {reliability} |",
        "",
        "| Category | Scenarios | Pass rate (95% CI) |",
        "|---|---|---|",
    ]
    for category, estimate in per_category(outcomes).items():
        lines.append(f"| {category} | {estimate.n_scenarios} | {estimate} |")

    lines += ["", "| Layer | Pass rate (95% CI) |", "|---|---|"]
    for layer, estimate in per_layer(outcomes).items():
        lines.append(f"| {layer} | {estimate} |")

    failures = [o for o in outcomes if not o.passed]
    if failures:
        lines += ["", "| Failed run | Why |", "|---|---|"]
        for o in failures:
            why = o.error or o.detail
            lines.append(f"| {o.scenario_id} #{o.run_index} | {why[:160]} |")

    lines += [
        "",
        f"Tokens: {suite.input_tokens:,} in / {suite.output_tokens:,} out.",
    ]
    return "\n".join(lines)
