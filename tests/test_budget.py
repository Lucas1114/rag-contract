"""Guarantee 5's latency half: the deadline, and what it does when it fires.

The deadline cannot fire against this corpus. The slowest question measures
around 7 ms against a 150 ms ceiling, so every test here drives the clock
rather than the workload. That is deliberate and is the point of injecting one:
a test that made the deadline fire by sleeping would be testing `time.sleep`,
and making it fire by doing real work would mean putting a slow path into the
service to have something to catch — which is the thing `budget.py` refuses to
do.

What the tests therefore have to pin is the *behaviour*, not the timing: where
the deadline is checked, and that a request which hits it produces no answer
rather than a cheap one.
"""

import pytest

from rag_contract.answering import AnswerState, Passage, decide
from rag_contract.budget import (
    Budget,
    BudgetError,
    BudgetLimits,
    DeadlineExceeded,
    measure_request_ms,
)
from rag_contract.grounding import Claim, check_claims
from rag_contract.registry import IndexRegistry
from rag_contract.service import Service

from .synthetic import make_index

LIMITS = {
    "request_deadline_ms": 150.0,
    "max_request_ms": 50.0,
}


class Clock:
    """A clock that moves only when told to."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


class TickingClock:
    """A clock that advances a fixed amount every time it is read.

    For driving the deadline from inside a loop, where the test cannot reach
    between iterations to move a manual clock.
    """

    def __init__(self, ms_per_read: float):
        self.ms_per_read = ms_per_read
        self.now = 0.0
        self.reads = 0

    def __call__(self) -> float:
        value = self.now
        self.reads += 1
        self.now += self.ms_per_read / 1000.0
        return value


# --- The committed limits have to be self-consistent ----------------------


def test_the_limits_load_from_a_well_formed_block():
    limits = BudgetLimits.from_mapping(LIMITS)
    assert limits.request_deadline_ms == 150.0
    assert limits.max_request_ms == 50.0


@pytest.mark.parametrize("missing", sorted(LIMITS))
def test_every_limit_is_required(missing):
    with pytest.raises(BudgetError, match=missing):
        BudgetLimits.from_mapping({k: v for k, v in LIMITS.items() if k != missing})


def test_no_budget_block_at_all_is_refused():
    with pytest.raises(BudgetError, match="nobody reviews"):
        BudgetLimits.from_mapping(None)


def test_a_limit_nothing_reads_is_refused():
    with pytest.raises(BudgetError, match="max_vibes"):
        BudgetLimits.from_mapping({**LIMITS, "max_vibes": 1.0})


@pytest.mark.parametrize("name", sorted(LIMITS))
def test_a_non_positive_limit_is_refused(name):
    with pytest.raises(BudgetError, match=name):
        BudgetLimits.from_mapping({**LIMITS, name: 0})


def test_a_gate_bar_without_headroom_under_the_deadline_is_refused():
    """A bar at the deadline is the deadline with extra steps.

    The build would be green on the last commit before requests start being
    abandoned, which is the one commit where it needed to be red.
    """
    with pytest.raises(BudgetError, match="headroom"):
        BudgetLimits.from_mapping({**LIMITS, "max_request_ms": 100.0})


def test_a_bar_at_exactly_half_the_deadline_is_allowed():
    assert (
        BudgetLimits.from_mapping({**LIMITS, "max_request_ms": 75.0}).max_request_ms
        == 75.0
    )


# --- What a request spends ------------------------------------------------


def test_stages_are_billed_to_the_stage_that_spent_them():
    clock = Clock()
    spend = Budget(deadline_ms=150.0, clock=clock).start()

    clock.advance_ms(2.0)
    spend.checkpoint("retrieval")
    clock.advance_ms(5.0)
    spend.checkpoint("grounding")

    assert spend.stages == {"retrieval": 2.0, "grounding": 5.0}
    assert spend.elapsed_ms == pytest.approx(7.0)
    assert not spend.exceeded


def test_a_stage_checkpointed_twice_accumulates():
    """The grounding stage is checkpointed once per claim, not once."""
    clock = Clock()
    spend = Budget(deadline_ms=150.0, clock=clock).start()
    for _ in range(3):
        clock.advance_ms(4.0)
        spend.checkpoint("grounding")
    assert spend.stages["grounding"] == pytest.approx(12.0)


def test_crossing_the_deadline_raises_naming_the_stage_that_crossed_it():
    clock = Clock()
    spend = Budget(deadline_ms=10.0, clock=clock).start()
    clock.advance_ms(4.0)
    spend.checkpoint("retrieval")

    clock.advance_ms(20.0)
    with pytest.raises(DeadlineExceeded) as raised:
        spend.checkpoint("grounding")

    assert raised.value.stage == "grounding"
    assert raised.value.spend is spend
    assert "grounding" in str(raised.value)
    assert "24.0 ms" in str(raised.value)


def test_the_report_carries_the_deadline_it_was_held_to():
    clock = Clock()
    spend = Budget(deadline_ms=150.0, clock=clock).start()
    clock.advance_ms(3.0)
    spend.checkpoint("retrieval")

    payload = spend.to_dict()
    assert payload["deadline_ms"] == 150.0
    assert payload["exceeded"] is False
    assert payload["stages"] == {"retrieval": 3.0}


# --- The deadline reaches the only unbounded loop --------------------------


PASSAGES = [
    Passage(
        section_id="rfc9110#1",
        citation="RFC9110 Section 1",
        rank=1,
        score=0.9,
        text="body of rfc9110#1 part 0",
    )
]


def claims(count: int) -> list[Claim]:
    return [Claim(text="body of part 0", citation="rfc9110#1") for _ in range(count)]


def test_the_grounding_check_stops_part_way_rather_than_finishing():
    """The claim loop is the one unbounded thing on the request path.

    A deadline checked only after `check_claims` returned would report the
    overrun having already paid for it in full, which for the failure this
    ceiling exists to catch — a model returning far more claims than usual — is
    the whole of the cost.
    """
    clock = TickingClock(ms_per_read=30.0)
    spend = Budget(deadline_ms=100.0, clock=clock).start()

    with pytest.raises(DeadlineExceeded) as raised:
        check_claims(
            claims(50),
            {"rfc9110#1": PASSAGES[0].text},
            0.67,
            checkpoint=spend.stage("grounding"),
        )

    # It gave up in the first handful of claims rather than checking all fifty.
    assert raised.value.stage == "grounding"
    assert clock.reads < 20


def test_no_partial_verdict_list_escapes_the_check():
    """A truncated grounding check is unsound, not merely weaker.

    Claims it never reached are not claims that passed, so there must be no
    value for a caller to be tempted into answering from. The exception
    carries the spend and nothing else.
    """
    clock = TickingClock(ms_per_read=30.0)
    spend = Budget(deadline_ms=100.0, clock=clock).start()

    with pytest.raises(DeadlineExceeded) as raised:
        check_claims(
            claims(50),
            {"rfc9110#1": PASSAGES[0].text},
            0.67,
            checkpoint=spend.stage("grounding"),
        )

    assert not hasattr(raised.value, "verdicts")
    assert not hasattr(raised.value, "supported")


def test_decide_is_unaffected_when_no_deadline_is_supplied():
    """The evals call `decide` directly and must stay free of a wall clock."""
    answer = decide(
        question="q",
        index_version="0" * 12,
        passages=PASSAGES,
        claims=claims(3),
    )
    assert answer.state is AnswerState.GROUNDED
    assert answer.spend is None
    assert "budget" not in answer.to_dict()


def test_an_answer_served_under_a_budget_reports_what_it_spent():
    answer = decide(
        question="q",
        index_version="0" * 12,
        passages=PASSAGES,
        claims=claims(3),
        spend=Budget(deadline_ms=150.0, clock=Clock()).start(),
    )
    payload = answer.to_dict()
    assert payload["budget"]["deadline_ms"] == 150.0
    assert payload["budget"]["stages"]["grounding"] == 0.0


# --- The served request ----------------------------------------------------


class OneClaim:
    def draft(self, question, passages):
        return claims(3)


def service(deadline_ms=150.0, clock=None):
    index = make_index(
        ["rfc9110#1"],
        [[1.0, 0.0]],
        question_ids=["qa"],
        question_vectors=[[1.0, 0.0]],
    )

    class Q:
        id = "qa"
        question = "q"
        answerable = True

    return Service(
        registry=IndexRegistry(index),
        questions={"qa": Q()},
        drafter=OneClaim(),
        budget=Budget(deadline_ms=deadline_ms, clock=clock or Clock()),
    )


def test_a_served_request_carries_its_budget():
    answer = service().answer("qa")
    assert answer.spend is not None
    assert set(answer.spend.stages) >= {"retrieval", "drafting", "grounding"}


def test_a_served_request_that_runs_out_of_time_is_abandoned():
    with pytest.raises(DeadlineExceeded):
        service(deadline_ms=1.0, clock=TickingClock(ms_per_read=5.0)).answer("qa")


def test_a_service_with_no_budget_reports_none():
    """What the cutover tests and the evals use. No clock, nothing to report."""
    svc = service()
    unbudgeted = Service(
        registry=svc.registry, questions=svc.questions, drafter=svc.drafter
    )
    assert unbudgeted.answer("qa").spend is None


# --- What the gate measures ------------------------------------------------


def test_measure_reports_the_worst_question_by_median():
    seen = []

    def answer(question_id):
        seen.append(question_id)

    slowest, slowest_ms, medians = measure_request_ms(answer, ["qa", "qb"], repeats=3)
    assert seen == ["qa"] * 3 + ["qb"] * 3
    assert set(medians) == {"qa", "qb"}
    assert slowest in {"qa", "qb"}
    assert slowest_ms == max(medians.values())
