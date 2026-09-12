"""The gate is what makes the measurement load-bearing, so its failure modes
are pinned here rather than its success.

A gate that cannot fail is decoration. Each test below is a way quality can get
worse that the build has to notice: a headline number slipping, one question
collapsing while the aggregate covers for it, and a question entering the set
without anyone recording what it costs.
"""

import pytest

from rag_contract.budget import BudgetLimits
from rag_contract.gate import (
    ThresholdError,
    Thresholds,
    budget_checks,
    failures,
    load_thresholds,
    refusal_checks,
    run_gate,
)

THRESHOLDS_YAML = """
aggregate:
  recall_at_1: 0.60
  mrr: 0.75
max_rank:
  q01: 3
  q02: 3
refusal:
  min_grounded_rate: 0.80
  min_state_agreement: 0.75
  max_answered_unanswerable: 1
budget:
  request_deadline_ms: 150.0
  max_request_ms: 50.0
  daily_cap_usd: 2.00
  max_build_index_usd: 0.02
  max_record_drafts_usd: 1.00
  max_requests_per_minute: 60
  max_client_burst: 10
"""

REFUSAL = {
    "min_grounded_rate": 0.80,
    "min_state_agreement": 0.75,
    "max_answered_unanswerable": 1,
}


BUDGET = {
    "request_deadline_ms": 150.0,
    "max_request_ms": 50.0,
    "daily_cap_usd": 2.00,
    "max_build_index_usd": 0.02,
    "max_record_drafts_usd": 1.00,
    "max_requests_per_minute": 60,
    "max_client_burst": 10,
}


def thresholds(aggregate=None, max_rank=None, refusal=None, budget=None):
    return Thresholds(
        aggregate=aggregate if aggregate is not None else {"recall_at_1": 0.6},
        max_rank=max_rank if max_rank is not None else {"q01": 3, "q02": 3},
        measured={},
        refusal=refusal if refusal is not None else dict(REFUSAL),
        budget=BudgetLimits.from_mapping(budget if budget is not None else BUDGET),
    )


def answer_report(grounded_rate=1.0, state_agreement=1.0, answered=()):
    return {
        "metrics": {
            "grounded_rate": grounded_rate,
            "state_agreement": state_agreement,
            "answered_unanswerable": len(answered),
            "answered_unanswerable_ids": list(answered),
        }
    }


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
        load_thresholds(
            write(
                tmp_path,
                THRESHOLDS_YAML.replace(
                    "aggregate:\n", "aggregate:\n  precision_at_1: 0.9\n"
                ),
            )
        )


def test_a_nonsensical_rank_ceiling_is_rejected_at_load(tmp_path):
    with pytest.raises(ThresholdError, match="q02"):
        load_thresholds(write(tmp_path, THRESHOLDS_YAML.replace("q02: 3", "q02: 0")))


# --- Refusal checks: guarantee 3 ------------------------------------------


def test_a_healthy_answer_report_passes_every_refusal_check():
    checks = refusal_checks(answer_report(), thresholds())
    assert [c.passed for c in checks] == [True, True, True]


def test_a_service_that_refuses_everything_fails_the_floor():
    # The reason the ceiling alone is not a gate. Refusing every question
    # scores a perfect zero on answered_unanswerable.
    checks = refusal_checks(answer_report(grounded_rate=0.0, answered=()), thresholds())
    failed = {c.name for c in checks if not c.passed}
    assert failed == {"grounded_rate"}


def test_a_service_that_answers_everything_fails_the_ceiling():
    # And the mirror: the floor alone is not a gate either.
    checks = refusal_checks(
        answer_report(grounded_rate=1.0, answered=("u01", "u02", "u03")),
        thresholds(),
    )
    failed = {c.name for c in checks if not c.passed}
    assert failed == {"answered_unanswerable"}


def test_the_ceiling_names_the_questions_that_broke_it():
    checks = refusal_checks(answer_report(answered=("u01", "u05")), thresholds())
    breach = next(c for c in checks if c.name == "answered_unanswerable")
    assert "u01, u05" in breach.detail


def test_the_known_shortfall_is_held_at_one_and_may_not_grow():
    # u01 reaches `grounded` without inventing anything; the check verifies
    # support, not responsiveness. One is tolerated and named. Two is not.
    assert all(
        c.passed for c in refusal_checks(answer_report(answered=("u01",)), thresholds())
    )
    failed = [
        c
        for c in refusal_checks(answer_report(answered=("u01", "u02")), thresholds())
        if not c.passed
    ]
    assert [c.name for c in failed] == ["answered_unanswerable"]


def test_state_agreement_has_its_own_floor():
    checks = refusal_checks(answer_report(state_agreement=0.5), thresholds())
    failed = {c.name for c in checks if not c.passed}
    assert failed == {"state_agreement"}


def test_a_metric_exactly_at_its_refusal_bar_passes():
    checks = refusal_checks(
        answer_report(grounded_rate=0.80, state_agreement=0.75, answered=("u01",)),
        thresholds(),
    )
    assert all(c.passed for c in checks)


def test_a_refusal_threshold_on_an_unreported_metric_is_an_error():
    with pytest.raises(ThresholdError, match="did not produce"):
        refusal_checks({"metrics": {}}, thresholds())


def test_a_threshold_file_with_no_refusal_block_is_rejected(tmp_path):
    path = tmp_path / "thresholds.yaml"
    path.write_text(THRESHOLDS_YAML.split("refusal:")[0])
    with pytest.raises(ThresholdError, match="no refusal thresholds"):
        load_thresholds(path)


def test_a_half_set_refusal_block_is_rejected(tmp_path):
    # Guarantee 3 is held from both sides or not at all.
    path = tmp_path / "thresholds.yaml"
    path.write_text(THRESHOLDS_YAML.replace("  min_grounded_rate: 0.80\n", ""))
    with pytest.raises(ThresholdError, match="both sides"):
        load_thresholds(path)


def test_an_unknown_refusal_threshold_is_rejected(tmp_path):
    path = tmp_path / "thresholds.yaml"
    path.write_text(
        THRESHOLDS_YAML.replace("refusal:\n", "refusal:\n  min_vibes: 0.9\n")
    )
    with pytest.raises(ThresholdError, match="does not report"):
        load_thresholds(path)


def test_the_committed_thresholds_carry_a_full_refusal_block():
    assert set(load_thresholds().refusal) == {
        "min_grounded_rate",
        "min_state_agreement",
        "max_answered_unanswerable",
    }


# --- Budget checks: guarantee 5 -------------------------------------------


def budget(**overrides):
    defaults = {
        "slowest_question": "q14",
        "slowest_ms": 7.4,
        "build_index_usd": 0.0059,
        "record_drafts_usd": 0.66,
        "worst_case_record_usd": 1.80,
        "thresholds": thresholds(),
    }
    return budget_checks(**{**defaults, **overrides})


def test_a_service_inside_every_budget_produces_no_failures():
    assert failures(budget()) == []


def test_a_request_path_that_got_slow_fails_the_build():
    """The regression this catches is the grounding check acquiring work.

    Its input — how many claims the model returns — is the one thing on the
    request path that the corpus does not bound.
    """
    failed = failures(budget(slowest_ms=60.0))
    assert [c.name for c in failed] == ["slowest request"]


def test_the_slow_check_names_the_question_and_the_deadline_it_is_under():
    check = budget(slowest_ms=60.0)[0]
    assert "q14" in check.detail
    assert "150.0 ms" in check.detail


def test_a_measurement_exactly_at_the_bar_passes():
    assert budget(slowest_ms=50.0)[0].passed


def test_a_corpus_that_outgrew_its_rebuild_budget_fails_the_build():
    """Which puts the cost of adding a document in the diff that adds it."""
    failed = failures(budget(build_index_usd=0.05))
    assert [c.name for c in failed] == ["build-index cost"]


def test_a_question_set_that_outgrew_its_recording_budget_fails_the_build():
    failed = failures(budget(record_drafts_usd=1.20))
    assert [c.name for c in failed] == ["record-drafts cost"]


def test_an_output_ceiling_the_daily_cap_could_not_authorise_fails_the_build():
    """The check that found something real.

    `drafter.MAX_TOKENS` was 4000, putting a worst-case run at $3.90 against a
    $2.00 cap. Nothing compared the two numbers until this check did, because
    they live in different files and are set by different people.
    """
    failed = failures(budget(worst_case_record_usd=3.90))
    assert [c.name for c in failed] == ["worst-case drafting"]
    assert "abandon the run part way through" in failed[0].detail


def test_the_four_budget_checks_are_independent():
    """Each fails on its own, so a green build means all four agree."""
    assert (
        len(
            failures(
                budget(
                    slowest_ms=60.0,
                    build_index_usd=0.05,
                    record_drafts_usd=1.20,
                    worst_case_record_usd=3.90,
                )
            )
        )
        == 4
    )


def test_the_committed_thresholds_carry_a_full_budget_block():
    limits = load_thresholds().budget
    assert limits.max_request_ms <= limits.request_deadline_ms / 2
    assert limits.max_record_drafts_usd <= limits.daily_cap_usd


def test_a_budget_block_that_contradicts_itself_is_a_threshold_error(tmp_path):
    """Anything wrong with the file reaches a caller as one kind of problem."""
    with pytest.raises(ThresholdError, match="headroom"):
        load_thresholds(
            write(
                tmp_path,
                THRESHOLDS_YAML.replace(
                    "max_request_ms: 50.0", "max_request_ms: 149.0"
                ),
            )
        )
