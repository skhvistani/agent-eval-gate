"""The faithfulness judge's plumbing, against a scripted fake model.

What a fake cannot test is whether the real judge is any good. That is what
calibration against hand labels is for, and it is a published number, not a
unit test.
"""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agent_eval_gate.scorers.faithfulness import (
    FaithfulnessVerdict,
    UnsupportedClaim,
    judge_prompt,
    render_tool_log,
    score_faithfulness,
)
from agent_eval_gate.trace import Step, ToolCall, Trace


@pytest.fixture
def anyio_backend():
    return "asyncio"


def call(name, arguments, result="{}", is_error=False):
    return ToolCall(
        id="tu", name=name, arguments=arguments, result=result, is_error=is_error, latency_ms=1
    )


def make_trace(final_message, calls=(), *, model="claude-opus-5", **kwargs):
    return Trace(
        scenario_id="t",
        seed=17,
        model=model,
        started_at=datetime.now(UTC),
        steps=[Step(index=0, tool_calls=list(calls))],
        final_message=final_message,
        **kwargs,
    )


class FakeJudge:
    """Returns a fixed verdict and records the prompt it was shown."""

    def __init__(self, verdict, stop_reason="end_turn"):
        self.verdict = verdict
        self.stop_reason = stop_reason
        self.requests = []
        self.messages = SimpleNamespace(parse=self._parse)

    async def _parse(self, **kwargs):
        self.requests.append(kwargs)
        block = SimpleNamespace(type="text", text="", parsed_output=self.verdict)
        return SimpleNamespace(content=[block], stop_reason=self.stop_reason)


FAITHFUL = FaithfulnessVerdict(faithful=True, reasoning="all supported")
UNFAITHFUL = FaithfulnessVerdict(
    faithful=False,
    unsupported_claims=[UnsupportedClaim(claim="I closed the ticket", reason="no close call")],
    reasoning="claims a close that never happened",
)


# --- what the judge is shown -------------------------------------------


def test_tool_log_marks_failures_unmistakably():
    trace = make_trace(
        "x",
        [
            call("get_order", {"order_id": "ORD-1"}, '{"status": "paid"}'),
            call("issue_refund", {"order_id": "ORD-1"}, "already refunded", is_error=True),
        ],
    )
    log = render_tool_log(trace)
    assert "1. get_order" in log and "-> OK" in log
    assert "2. issue_refund" in log and "-> FAILED: already refunded" in log


def test_empty_log_is_stated_not_blank():
    assert "no tool calls" in render_tool_log(make_trace("x"))


def test_judge_never_sees_expected_answer():
    """Faithfulness is not correctness: the prompt carries task, log, message only."""
    trace = make_trace("I refunded ORD-1.", [call("issue_refund", {"order_id": "ORD-1"})])
    prompt = judge_prompt("Refund ORD-1.", trace)
    assert "Refund ORD-1." in prompt
    assert "I refunded ORD-1." in prompt
    assert "expected" not in prompt.lower()
    assert "reference" not in prompt.lower()


# --- verdict handling ---------------------------------------------------


@pytest.mark.anyio
async def test_faithful_verdict_passes():
    judge = FakeJudge(FAITHFUL)
    result = await score_faithfulness("t", make_trace("done"), client=judge, judge_model="other")
    assert result.passed
    assert result.judge_model == "other"
    assert result.summary == "pass"


@pytest.mark.anyio
async def test_unfaithful_verdict_fails_and_quotes_the_claim():
    judge = FakeJudge(UNFAITHFUL)
    result = await score_faithfulness("t", make_trace("done"), client=judge, judge_model="other")
    assert not result.passed
    assert "I closed the ticket" in result.summary


@pytest.mark.anyio
async def test_listed_claims_override_a_true_faithful_flag():
    """A verdict that says faithful but lists unsupported claims is not faithful."""
    contradictory = FaithfulnessVerdict(
        faithful=True,
        unsupported_claims=[UnsupportedClaim(claim="x", reason="y")],
        reasoning="",
    )
    result = await score_faithfulness(
        "t", make_trace("done"), client=FakeJudge(contradictory), judge_model="other"
    )
    assert not result.passed


@pytest.mark.anyio
async def test_structured_output_is_requested():
    judge = FakeJudge(FAITHFUL)
    await score_faithfulness("t", make_trace("done"), client=judge, judge_model="other")
    assert judge.requests[0]["output_format"] is FaithfulnessVerdict


# --- safeguards ---------------------------------------------------------


@pytest.mark.anyio
async def test_refuses_to_judge_its_own_model():
    """Never grade your own work: the minimum defence against self-preference."""
    judge = FakeJudge(FAITHFUL)
    trace = make_trace("done", model="claude-opus-5")
    result = await score_faithfulness("t", trace, client=judge, judge_model="claude-opus-5")
    assert not result.passed
    assert "model under test" in result.summary
    assert judge.requests == [], "must not have called the model"


@pytest.mark.anyio
async def test_harness_error_is_not_judged():
    judge = FakeJudge(FAITHFUL)
    trace = make_trace("done", error="RuntimeError: boom")
    result = await score_faithfulness("t", trace, client=judge, judge_model="other")
    assert not result.passed and "boom" in result.summary
    assert judge.requests == []


@pytest.mark.anyio
async def test_empty_final_message_is_not_a_pass():
    result = await score_faithfulness(
        "t", make_trace("   "), client=FakeJudge(FAITHFUL), judge_model="other"
    )
    assert not result.passed
    assert "no final message" in result.summary


@pytest.mark.anyio
async def test_judge_refusal_is_recorded_not_passed():
    judge = FakeJudge(FAITHFUL, stop_reason="refusal")
    result = await score_faithfulness("t", make_trace("done"), client=judge, judge_model="other")
    assert not result.passed and "refused" in result.summary


@pytest.mark.anyio
async def test_judge_crash_is_recorded_not_raised():
    class Exploding(FakeJudge):
        async def _parse(self, **kwargs):
            raise RuntimeError("rate limited")

    result = await score_faithfulness(
        "t", make_trace("done"), client=Exploding(FAITHFUL), judge_model="other"
    )
    assert not result.passed and "rate limited" in result.summary
