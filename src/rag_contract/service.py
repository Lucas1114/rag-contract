"""The request path: one question in, one attributable answer out.

This is where the index reference is acquired, and it is the only place that
calls `registry.acquire()`. Everything downstream — the query vector, the
chunks, the passages, the version reported in the response — comes from the
object returned by that one call, so the answer is attributable to exactly one
index whatever happens to the registry while the request is running.

The question set is fixed, and that is a constraint rather than a limitation.
There is no open-ended prompt input anywhere in this service: a request names
one of the questions in `eval/questions.yaml`, and anything else is a 404. That
is what lets the answer step replay a committed draft instead of calling a
model, which in turn is what lets the whole grounding check run with no key and
no cost. A service that accepted arbitrary text would need a live model on the
request path and would be a different project.
"""

from __future__ import annotations

from dataclasses import dataclass

from .answering import (
    NO_CONTEXT_MESSAGE,
    Answer,
    AnswerState,
    decide,
    passages_from_hits,
)
from .drafter import Drafter, FixtureDrafter
from .evalset import Question, load_questions
from .evaluate import TOP_K
from .registry import IndexRegistry
from .retrieval import search


class UnknownQuestion(KeyError):
    """No question by that id. Not a failure state — a request for nothing."""


@dataclass(frozen=True)
class Service:
    """Retrieval, drafting and the grounding check over one acquired index."""

    registry: IndexRegistry
    questions: dict[str, Question]
    drafter: Drafter
    top_k: int = TOP_K

    @classmethod
    def from_committed(cls, registry: IndexRegistry, top_k: int = TOP_K) -> Service:
        """The service as it is actually served: committed index, committed drafts."""
        return cls(
            registry=registry,
            questions={q.id: q for q in load_questions()},
            drafter=FixtureDrafter.from_directory(),
            top_k=top_k,
        )

    def question(self, question_id: str) -> Question:
        try:
            return self.questions[question_id]
        except KeyError as exc:
            raise UnknownQuestion(question_id) from exc

    def answer(self, question_id: str) -> Answer:
        """One request. The index is acquired once, on the line below.

        Everything after that line reads `index`, never the registry. A cutover
        landing anywhere in this method is therefore invisible to it: the
        request keeps the object it acquired, finishes against those vectors,
        and reports that version — which is the guarantee, expressed as the
        absence of a second acquire rather than as machinery.
        """
        question = self.question(question_id)
        index = self.registry.acquire()

        if index is None:
            # The index is unavailable. Nothing was retrieved, so there is
            # nothing to refuse *from*: this is a service error, not a
            # decision the service made about the corpus.
            return Answer(
                question=question.question,
                state=AnswerState.NO_CONTEXT,
                index_version="",
                message=NO_CONTEXT_MESSAGE,
            )

        hits = search(index, index.question_vector(question.id), k=self.top_k)
        passages = passages_from_hits(hits)
        return decide(
            question=question.question,
            index_version=index.version,
            passages=passages,
            claims=self.drafter.draft(question.question, passages),
        )
