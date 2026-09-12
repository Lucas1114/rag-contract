"""Scoring the failure behaviour: the measurement behind guarantee 3.

Guarantee 1's eval asks whether retrieval found the right passage. This one
asks what the service then *did* — which of the four states each question ended
in, and whether that is the state `eval/questions.yaml` says it must end in.

Like the retrieval eval it is deterministic and keyless, and for the same kind
of reason: the drafts are committed, so the only thing that varies between runs
is the code being measured. Retrieval, the grounding check and the state
machine all rerun; only the model's words are replayed.

What the report is built to support
-----------------------------------

Guarantee 3 has to be gated from two sides, because either alone is trivially
satisfiable.

Gate only the refusals and a service that refuses everything scores perfectly.
`answered_unanswerable` is the metric that must be zero, and `grounded_
answerable` is what stops zero being achievable by refusing the whole question
set. A change that tightens the coverage floor until nothing grounds moves the
first metric not at all and the second straight through the floor.

That is the same argument the retrieval gate makes about aggregates and
per-question ceilings, arriving at the same shape from the other direction: one
number pinned at zero for the outcome that must never happen, one floor under
the behaviour that makes the zero meaningful.

`withdrawn_claims` is reported and not gated. It counts the claims the check
actually rejected — if it were ever zero across the whole question set the
check would be untested by its own eval, which is worth seeing, but a model
that simply never overreaches is a legitimate way to reach zero and not a
regression to fail a build on.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .answering import Answer, AnswerState, decide, passages_from_hits
from .drafter import Drafter, FixtureDrafter
from .evalset import Question
from .evaluate import TOP_K
from .grounding import COVERAGE_FLOOR
from .index import Index
from .retrieval import search


@dataclass(frozen=True)
class AnswerResult:
    question: Question
    answer: Answer
    drift: tuple[str, ...] = ()

    @property
    def agrees(self) -> bool:
        return self.answer.state is self.question.expected_state

    def to_dict(self) -> dict:
        payload = self.answer.to_dict()
        payload.update(
            {
                "id": self.question.id,
                "answerable": self.question.answerable,
                "expected_state": str(self.question.expected_state),
                "agrees": self.agrees,
                "drift": list(self.drift),
            }
        )
        return payload


def answer_question(
    index: Index,
    question: Question,
    drafter: Drafter,
    top_k: int = TOP_K,
    coverage_floor: float = COVERAGE_FLOOR,
) -> AnswerResult:
    hits = search(index, index.question_vector(question.id), k=top_k)
    passages = passages_from_hits(hits)
    claims = drafter.draft(question.question, passages)
    answer = decide(
        question=question.question,
        index_version=index.version,
        passages=passages,
        claims=claims,
        coverage_floor=coverage_floor,
    )
    drift = (
        drafter.drift(question.question, passages)
        if isinstance(drafter, FixtureDrafter)
        else ()
    )
    return AnswerResult(question=question, answer=answer, drift=drift)


def evaluate_answers(
    index: Index,
    questions: list[Question],
    drafter: Drafter,
    top_k: int = TOP_K,
    coverage_floor: float = COVERAGE_FLOOR,
) -> dict:
    results = [
        answer_question(index, q, drafter, top_k, coverage_floor) for q in questions
    ]
    answerable = [r for r in results if r.question.answerable]
    unanswerable = [r for r in results if not r.question.answerable]

    grounded_answerable = sum(
        r.answer.state is AnswerState.GROUNDED for r in answerable
    )
    # The outcome guarantee 3 exists to prevent: a confident, fully grounded
    # answer to a question the corpus cannot answer.
    answered_unanswerable = [
        r.question.id for r in unanswerable if r.answer.state is AnswerState.GROUNDED
    ]
    agreeing = [r for r in unanswerable if r.agrees]

    return {
        "index_version": index.version,
        "coverage_floor": coverage_floor,
        "top_k": top_k,
        "drafts": {
            "models": sorted(
                {drafter.recording(r.question.question).model for r in results}
            )
            if isinstance(drafter, FixtureDrafter)
            else [],
            "drifted_questions": [r.question.id for r in results if r.drift],
        },
        "metrics": {
            "answerable_questions": len(answerable),
            "unanswerable_questions": len(unanswerable),
            "states": dict(Counter(str(r.answer.state) for r in results).most_common()),
            # Pinned at zero by the gate. Nothing else here can substitute.
            "answered_unanswerable": len(answered_unanswerable),
            "answered_unanswerable_ids": answered_unanswerable,
            # The floor that stops zero being reachable by refusing everything.
            "grounded_answerable": grounded_answerable,
            "grounded_rate": round(grounded_answerable / len(answerable), 4)
            if answerable
            else 0.0,
            "state_agreement": round(len(agreeing) / len(unanswerable), 4)
            if unanswerable
            else 0.0,
            "disagreements": [
                {
                    "id": r.question.id,
                    "expected": str(r.question.expected_state),
                    "observed": str(r.answer.state),
                }
                for r in unanswerable
                if not r.agrees
            ],
            # Evidence the check fires at all. Reported, not gated.
            "withdrawn_claims": sum(len(r.answer.withdrawn) for r in results),
            "supported_claims": sum(len(r.answer.supported) for r in results),
        },
        "questions": [r.to_dict() for r in results],
    }
