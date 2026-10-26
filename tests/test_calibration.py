"""Agreement and Cohen's kappa. Small hand-checkable cases."""

import pytest

from agent_eval_gate.scorers.calibration import agreement, cohens_kappa, load_labels


def test_perfect_agreement():
    pairs = [(True, True), (False, False), (True, True), (False, False)]
    assert cohens_kappa(pairs) == pytest.approx(1.0)


def test_chance_agreement_is_zero():
    """Judge says True half the time regardless of the human: kappa 0."""
    pairs = [(True, True), (False, True), (True, False), (False, False)]
    assert cohens_kappa(pairs) == pytest.approx(0.0)


def test_always_faithful_judge_gets_high_accuracy_but_zero_kappa():
    """The inflation accuracy hides and kappa exposes."""
    human = {f"t{i}": True for i in range(9)} | {"t9": False}
    judge = {k: True for k in human}
    result = agreement(judge, human)
    assert result.accuracy == pytest.approx(0.9)
    assert result.kappa == pytest.approx(0.0)
    assert result.missed_unfaithful_rate == pytest.approx(1.0)


def test_confusion_matrix_orientation():
    """Positive class is 'unfaithful', since that is what a gate must catch."""
    judge = {"a": False, "b": True, "c": False, "d": True}
    human = {"a": False, "b": True, "c": True, "d": False}
    result = agreement(judge, human)
    assert result.true_positive == 1  # a
    assert result.true_negative == 1  # b
    assert result.false_positive == 1  # c: judge unfaithful, human faithful
    assert result.false_negative == 1  # d: judge faithful, human unfaithful


def test_only_overlapping_ids_are_compared():
    result = agreement({"a": True, "x": True}, {"a": True, "y": False})
    assert result.n == 1


def test_empty_is_safe():
    assert agreement({}, {}).n == 0
    assert cohens_kappa([]) == 0.0


def test_labels_load_from_yaml(tmp_path):
    path = tmp_path / "labels.yaml"
    path.write_text(
        "- trace_id: refusal-trap-001:0\n  faithful: true\n  note: declined and said so\n"
        "- trace_id: single-action-001:2\n  faithful: false\n"
    )
    labels = load_labels(path)
    assert [label.trace_id for label in labels] == ["refusal-trap-001:0", "single-action-001:2"]
    assert labels[1].faithful is False
