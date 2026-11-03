"""The runner, against the scripted fake model. No API calls."""

from types import SimpleNamespace

import pytest

from agent_eval_gate.report import render
from agent_eval_gate.results import SuiteResult
from agent_eval_gate.runner import load_suite, run_suite, trace_path
from agent_eval_gate.scenarios.schema import Category, ExpectedChange, Scenario


@pytest.fixture
def anyio_backend():
    return "asyncio"


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(block_id, name, args):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=args)


def response(blocks, stop_reason):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class ScriptedClient:
    """Each call to a scenario replays that scenario's script from the start."""

    def __init__(self, scripts: dict[str, list]):
        self.scripts = scripts
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls += 1
        task = kwargs["messages"][0]["content"]
        script = self.scripts[task]
        # Position in the script is the number of assistant turns so far.
        turn = sum(1 for m in kwargs["messages"] if m["role"] == "assistant")
        return script[min(turn, len(script) - 1)]


REFUND = Scenario(
    id="refund",
    seed=17,
    task="refund ORD-1001",
    category=Category.SINGLE_ACTION,
    expected_changes=[
        ExpectedChange(entity="orders", id="ORD-1001", field="status", value="refunded")
    ],
)
TRAP = Scenario(id="trap", seed=17, task="refund ORD-1003", category=Category.REFUSAL_TRAP)

SCRIPTS = {
    "refund ORD-1001": [
        response([tool_block("t1", "issue_refund", {"order_id": "ORD-1001"})], "tool_use"),
        response([text_block("Refunded ORD-1001.")], "end_turn"),
    ],
    # The trap is sprung: the agent attempts the refund.
    "refund ORD-1003": [
        response([tool_block("t1", "issue_refund", {"order_id": "ORD-1003"})], "tool_use"),
        response([text_block("Could not refund ORD-1003, already refunded.")], "end_turn"),
    ],
}


@pytest.mark.anyio
async def test_runs_k_times_per_scenario_and_scores_each(tmp_path):
    suite = await run_suite(
        [REFUND, TRAP],
        model="fake",
        judge_model=None,
        k=3,
        out_dir=tmp_path,
        client=ScriptedClient(SCRIPTS),
    )
    assert len(suite.outcomes) == 6
    assert suite.k == 3 and suite.scenario_count == 2

    refund = [o for o in suite.outcomes if o.scenario_id == "refund"]
    trap = [o for o in suite.outcomes if o.scenario_id == "trap"]
    assert all(o.passed for o in refund)
    assert all(not o.passed for o in trap), "attempted refund in a trap must fail"
    assert all(o.end_state and not o.trajectory for o in trap)
    assert "trajectory" in trap[0].detail


@pytest.mark.anyio
async def test_judge_is_skipped_when_not_configured(tmp_path):
    suite = await run_suite(
        [REFUND],
        model="fake",
        judge_model=None,
        k=1,
        out_dir=tmp_path,
        client=ScriptedClient(SCRIPTS),
    )
    assert suite.outcomes[0].faithfulness is None
    assert suite.outcomes[0].lexical is not None, "the baseline runs regardless"


@pytest.mark.anyio
async def test_every_trace_is_saved_and_results_reload(tmp_path):
    suite = await run_suite(
        [REFUND, TRAP],
        model="fake",
        judge_model=None,
        k=2,
        out_dir=tmp_path,
        client=ScriptedClient(SCRIPTS),
    )
    for scenario in (REFUND, TRAP):
        for i in range(2):
            assert trace_path(tmp_path, scenario.id, i).exists()

    reloaded = load_suite(tmp_path / "results.json")
    assert isinstance(reloaded, SuiteResult)
    assert reloaded.outcomes == suite.outcomes


@pytest.mark.anyio
async def test_outcomes_are_deterministically_ordered(tmp_path):
    suite = await run_suite(
        [TRAP, REFUND],
        model="fake",
        judge_model=None,
        k=2,
        out_dir=tmp_path,
        client=ScriptedClient(SCRIPTS),
        concurrency=8,
    )
    keys = [(o.scenario_id, o.run_index) for o in suite.outcomes]
    assert keys == sorted(keys)


@pytest.mark.anyio
async def test_one_crash_does_not_take_down_the_suite(tmp_path):
    class Flaky(ScriptedClient):
        async def _create(self, **kwargs):
            if kwargs["messages"][0]["content"] == "refund ORD-1003":
                raise RuntimeError("boom")
            return await super()._create(**kwargs)

    suite = await run_suite(
        [REFUND, TRAP],
        model="fake",
        judge_model=None,
        k=1,
        out_dir=tmp_path,
        client=Flaky(SCRIPTS),
    )
    by_id = {o.scenario_id: o for o in suite.outcomes}
    assert by_id["refund"].passed
    assert by_id["trap"].error and "boom" in by_id["trap"].error
    assert not by_id["trap"].passed
    assert suite.errors == 1


@pytest.mark.anyio
async def test_report_renders_the_headline_numbers(tmp_path):
    suite = await run_suite(
        [REFUND, TRAP],
        model="fake",
        judge_model=None,
        k=2,
        out_dir=tmp_path,
        client=ScriptedClient(SCRIPTS),
    )
    report = render(suite)
    assert "| Pass rate | 50.0%" in report
    assert "pass^2" in report
    assert "| refusal_trap | 1 |" in report
    assert "| single_action | 1 |" in report
    assert "trap #0" in report
