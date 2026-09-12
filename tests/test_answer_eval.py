"""The answer eval, on a hand-built index and hand-written drafts.

The point of these tests is the shape of the measurement, not the corpus: the
two metrics that gate guarantee 3 have to move in the ways the gate assumes,
and a refuse-everything service has to be visibly distinguishable from a
working one.
"""

import numpy as np

from rag_contract.answer_eval import evaluate_answers
from rag_contract.answering import AnswerState
from rag_contract.evalset import Question
from rag_contract.grounding import Claim

from .synthetic import make_index

SAFE = "the safe methods are GET HEAD OPTIONS TRACE"
MENTION = "HTTP/2 introduced a multiplexed session layer over TCP"


def index():
    # Two chunks, two questions, each question nearest its own chunk.
    return make_index(
        ["rfc9110#9.2.1", "rfc9110#1.2"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        question_ids=["q04", "u01"],
        question_vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def questions():
    return [
        Question(
            id="q04",
            question="Which methods are safe?",
            answerable=True,
            expected=("rfc9110#9.2.1",),
        ),
        Question(
            id="u01",
            question="How does HTTP/2 multiplex?",
            answerable=False,
            expected_state=AnswerState.PARTIAL,
            state_reason="mentioned, not explained",
            absent="RFC 9113 is not in the corpus",
        ),
    ]


class StubDrafter:
    """Stands in for a recorded model. Keyed the same way a fixture is."""

    def __init__(self, by_question: dict[str, list[Claim]]):
        self._by_question = by_question

    def draft(self, question: str, passages):
        del passages
        return list(self._by_question[question])


def report(drafts):
    # Chunk text is "body of <section> part <n>", so claims are written to
    # ground or not against that rather than against real RFC prose.
    return evaluate_answers(index(), questions(), StubDrafter(drafts), top_k=2)


GROUNDS = Claim("body of rfc9110#9.2.1 part 0", "rfc9110#9.2.1")
INVENTED = Claim("A HEADERS frame carries a stream dependency.", "rfc9113#6.2")
MENTION_CLAIM = Claim("body of rfc9110#1.2 part 1", "rfc9110#1.2")


class TestTheTwoGatedMetrics:
    def test_a_working_service_grounds_the_answerable_and_refuses_the_rest(self):
        metrics = report(
            {
                "Which methods are safe?": [GROUNDS],
                "How does HTTP/2 multiplex?": [MENTION_CLAIM, INVENTED],
            }
        )["metrics"]
        assert metrics["answered_unanswerable"] == 0
        assert metrics["grounded_rate"] == 1.0
        assert metrics["state_agreement"] == 1.0

    def test_refusing_everything_keeps_the_zero_but_collapses_the_floor(self):
        # The reason guarantee 3 needs two-sided gating. A service that refuses
        # every question scores perfectly on the metric that must be zero.
        metrics = report(
            {
                "Which methods are safe?": [],
                "How does HTTP/2 multiplex?": [],
            }
        )["metrics"]
        assert metrics["answered_unanswerable"] == 0
        assert metrics["grounded_rate"] == 0.0

    def test_answering_an_unanswerable_question_is_counted_and_named(self):
        metrics = report(
            {
                "Which methods are safe?": [GROUNDS],
                "How does HTTP/2 multiplex?": [MENTION_CLAIM],
            }
        )["metrics"]
        assert metrics["answered_unanswerable"] == 1
        assert metrics["answered_unanswerable_ids"] == ["u01"]


class TestAgreement:
    def test_a_disagreement_names_expected_and_observed(self):
        metrics = report(
            {
                "Which methods are safe?": [GROUNDS],
                "How does HTTP/2 multiplex?": [INVENTED],
            }
        )["metrics"]
        assert metrics["disagreements"] == [
            {"id": "u01", "expected": "partial", "observed": "unsupported"}
        ]
        assert metrics["state_agreement"] == 0.0

    def test_answerable_questions_are_not_counted_in_agreement(self):
        # Agreement is about the refusal annotations. An answerable question
        # failing to ground shows up in grounded_rate instead.
        metrics = report(
            {
                "Which methods are safe?": [INVENTED],
                "How does HTTP/2 multiplex?": [MENTION_CLAIM, INVENTED],
            }
        )["metrics"]
        assert metrics["state_agreement"] == 1.0
        assert metrics["grounded_rate"] == 0.0


class TestReportShape:
    def test_states_are_counted_across_every_question(self):
        metrics = report(
            {
                "Which methods are safe?": [GROUNDS],
                "How does HTTP/2 multiplex?": [MENTION_CLAIM, INVENTED],
            }
        )["metrics"]
        assert metrics["states"] == {"grounded": 1, "partial": 1}

    def test_withdrawn_claims_are_reported_as_evidence_the_check_fires(self):
        metrics = report(
            {
                "Which methods are safe?": [GROUNDS],
                "How does HTTP/2 multiplex?": [MENTION_CLAIM, INVENTED],
            }
        )["metrics"]
        assert metrics["withdrawn_claims"] == 1
        assert metrics["supported_claims"] == 2

    def test_every_question_carries_its_expectation_and_verdict(self):
        payload = report(
            {
                "Which methods are safe?": [GROUNDS],
                "How does HTTP/2 multiplex?": [MENTION_CLAIM, INVENTED],
            }
        )
        u01 = next(q for q in payload["questions"] if q["id"] == "u01")
        assert u01["expected_state"] == "partial"
        assert u01["state"] == "partial"
        assert u01["agrees"] is True

    def test_the_coverage_floor_is_recorded_in_the_report(self):
        # It decides every verdict below, so a report that did not say which
        # floor produced it would not be reproducible.
        assert report(
            {"Which methods are safe?": [GROUNDS], "How does HTTP/2 multiplex?": []}
        )["coverage_floor"]
