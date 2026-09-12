"""The gate is what makes the measurement load-bearing, so its failure modes
are pinned here rather than its success.

A gate that cannot fail is decoration. Each test below is a way quality can get
worse that the build has to notice: a headline number slipping, one question
collapsing while the aggregate covers for it, and a question entering the set
without anyone recording what it costs.
"""

import pytest

from rag_contract.gate import (
    ThresholdError,
    Thresholds,
    failures,
    load_thresholds,
    run_gate,
)

THRESHOLDS_YAML = """
aggregate:
  recall_at_1: 0.60
  mrr: 0.75
max_rank:
  q01: 3
  q02: 3
"""


def thresholds(aggregate=None, max_rank=None):
    return Thresholds(
        aggregate=aggregate if aggregate is not None else {"recall_at_1": 0.6},
        max_rank=max_rank if max_rank is not None else {"q01": 3, "q02": 3},
        measured={},
    )


def report(ranks=(1, 1), recall_at_1=1.0, mrr=1.0, unanswerable=("u01",)):
    questions = [
        {
            "id": f"q0{i + 1}",
            "answerable": True,
            "rank": rank,
            "top_section": "rfc9110#1",
        }
        for i, rank in enumerate(ranks)
    ]
    questions += [
        {"id": qid, "answerable": False, "rank": None, "top_section": "rfc9110#1"}
        for qid in unanswerable
    ]
    return {
        "index_version": "0fc1763d6701",
        "top_k": 10,
        "metrics": {"recall_at_1": recall_at_1, "mrr": mrr},
        "questions": questions,
    }


def test_a_passing_report_produces_no_failures():
    assert failures(run_gate(report(), thresholds())) == []


def test_an_aggregate_below_its_floor_fails():
    checks = failures(run_gate(report(recall_at_1=0.55), thresholds()))
    assert [c.name for c in checks] == ["recall_at_1"]


def test_a_metric_exactly_at_its_floor_passes():
    assert failures(run_gate(report(recall_at_1=0.6), thresholds())) == []


def test_one_question_collapsing_fails_even_when_the_aggregate_improves():
    # q01 falls from rank 1 to rank 8 while q02 is unchanged; recall@1 and MRR
    # are handed in higher than the floors. The aggregate is content and the
    # gate is not, which is the reason per-question ceilings exist at all.
    checks = failures(
        run_gate(report(ranks=(8, 1), recall_at_1=0.9, mrr=0.9), thresholds())
    )
    assert [c.name for c in checks] == ["rank q01"]
    assert "first hit at 8" in checks[0].detail


def test_a_question_with_no_expected_section_in_the_window_fails():
    checks = failures(run_gate(report(ranks=(None, 1)), thresholds()))
    assert [c.name for c in checks] == ["rank q01"]
    assert "top 10" in checks[0].detail


def test_an_answerable_question_with_no_ceiling_fails_the_gate():
    checks = failures(run_gate(report(ranks=(1, 1, 1)), thresholds()))
    assert [c.name for c in checks] == ["ungated questions"]
    assert "q03" in checks[0].detail


def test_a_ceiling_for_a_question_that_no_longer_exists_fails_the_gate():
    checks = failures(
        run_gate(report(), thresholds(max_rank={"q01": 3, "q02": 3, "q09": 3}))
    )
    assert [c.name for c in checks] == ["stale ceilings"]
    assert "q09" in checks[0].detail


def test_unanswerable_questions_are_not_gated():
    # P02 measured the overlap that makes a score band on refusal questions
    # indefensible. Nothing here may quietly reintroduce one.
    checks = run_gate(report(unanswerable=("u01", "u02")), thresholds())
    assert not any("u0" in c.name for c in checks)


def test_a_floor_on_a_metric_the_eval_does_not_report_is_an_error():
    with pytest.raises(ThresholdError, match="did not produce"):
        run_gate(report(), thresholds(aggregate={"recall_at_5": 0.9}))


def write(tmp_path, text):
    path = tmp_path / "thresholds.yaml"
    path.write_text(text)
    return path


def test_thresholds_load_from_the_committed_file():
    loaded = load_thresholds()
    assert loaded.aggregate["recall_at_10"] == 1.0
    assert len(loaded.max_rank) == 20
    assert loaded.measured["index_version"] == "0fc1763d6701"


def test_the_committed_thresholds_sit_below_the_measured_numbers():
    # Floors pinned to the measured value would fail on the first legitimate
    # chunk-parameter change and the gate would be switched off. Headroom is a
    # property of the file, not an accident of the numbers in it.
    loaded = load_thresholds()
    measured = {"recall_at_1": 0.70, "recall_at_5": 1.00, "mrr": 0.8292}
    for name, value in measured.items():
        assert loaded.aggregate[name] < value, f"{name} has no headroom"


def test_a_missing_threshold_file_is_an_error(tmp_path):
    with pytest.raises(ThresholdError, match="no bar to hold"):
        load_thresholds(tmp_path / "absent.yaml")


def test_a_file_with_no_floors_is_an_error(tmp_path):
    with pytest.raises(ThresholdError, match="no aggregate floors"):
        load_thresholds(write(tmp_path, "max_rank:\n  q01: 3\n"))


def test_a_file_with_no_rank_ceilings_is_an_error(tmp_path):
    with pytest.raises(ThresholdError, match="no per-question rank ceilings"):
        load_thresholds(write(tmp_path, "aggregate:\n  mrr: 0.75\n"))


def test_a_floor_on_an_unknown_metric_is_rejected_at_load(tmp_path):
    with pytest.raises(ThresholdError, match="precision_at_1"):
        load_thresholds(write(tmp_path, THRESHOLDS_YAML + "  precision_at_1: 0.9\n"))


def test_a_nonsensical_rank_ceiling_is_rejected_at_load(tmp_path):
    with pytest.raises(ThresholdError, match="q02"):
        load_thresholds(write(tmp_path, THRESHOLDS_YAML.replace("q02: 3", "q02: 0")))
