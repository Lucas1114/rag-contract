"""Scoring retrieval against the annotated question set.

Guarantee 1 is that retrieval quality is measured rather than claimed, and this
is the measurement. It is deterministic numpy over committed vectors: same
index, same questions, same numbers, on any machine and in CI, with no network
call and no key.

Scoring rule, as stated in `eval/questions.yaml`: a question is a hit at k when
any chunk in the top k carries one of its expected sections. Rank is the
position of the first such chunk.

Refusal questions carry no expected sections and are not scored for recall.
What is reported for them is their top similarity score, because the boundary
between the three failure states of guarantee 3 is a threshold on exactly that
number, and it was deliberately left unannotated until a real distribution
existed. This report is that distribution.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from .evalset import Question, questions_fingerprint
from .index import Index, IndexError_
from .retrieval import Hit, search

RECALL_AT = (1, 3, 5, 10)
TOP_K = max(RECALL_AT)


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float
    chunk_id: str
    section_id: str
    expected: bool


@dataclass(frozen=True)
class QuestionResult:
    id: str
    question: str
    answerable: bool
    expected: list[str]
    rank: int | None  # first rank carrying an expected section, None on a miss
    top_score: float
    top_section: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


def _score_summary(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0}
    ordered = sorted(scores)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "median": round(statistics.median(ordered), 4),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
    }


def evaluate_question(
    index: Index, question: Question, top_k: int = TOP_K
) -> QuestionResult:
    hits: list[Hit] = search(index, index.question_vector(question.id), k=top_k)
    expected = set(question.expected)

    rank = next((h.rank for h in hits if h.section_id in expected), None)
    return QuestionResult(
        id=question.id,
        question=question.question,
        answerable=question.answerable,
        expected=list(question.expected),
        rank=rank,
        top_score=round(hits[0].score, 4) if hits else 0.0,
        top_section=hits[0].section_id if hits else "",
        retrieved=[
            RetrievedChunk(
                rank=h.rank,
                score=round(h.score, 4),
                chunk_id=h.chunk.id,
                section_id=h.section_id,
                expected=h.section_id in expected,
            )
            for h in hits
        ],
    )


def evaluate(index: Index, questions: list[Question], top_k: int = TOP_K) -> dict:
    """Run the whole question set and return the machine-readable report."""
    current = questions_fingerprint()
    if current != index.meta.questions_fingerprint:
        raise IndexError_(
            "the question set has changed since the index was built "
            f"(questions.yaml is {current[:12]}, index was built against "
            f"{index.meta.questions_fingerprint[:12]}). "
            "Rerun `rag-contract build-index` so the committed question "
            "vectors match the questions being scored."
        )

    results = [evaluate_question(index, q, top_k) for q in questions]
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    recall = {
        f"recall_at_{k}": round(
            sum(r.hit_at(k) for r in answerable) / len(answerable), 4
        )
        for k in RECALL_AT
        if answerable
    }
    reciprocal = [1.0 / r.rank for r in answerable if r.rank is not None]
    mrr = round(sum(reciprocal) / len(answerable), 4) if answerable else 0.0

    return {
        "index_version": index.version,
        "embedding_model": index.meta.embedding_model,
        "chunk_words": index.meta.chunk_words,
        "overlap_words": index.meta.overlap_words,
        "questions_fingerprint": current,
        "top_k": top_k,
        "metrics": {
            "answerable_questions": len(answerable),
            "unanswerable_questions": len(unanswerable),
            **recall,
            "mrr": mrr,
            "misses": [r.id for r in answerable if r.rank is None],
        },
        # The input to the failure-state threshold deferred in questions.yaml:
        # how far apart answerable and unanswerable questions actually score.
        "score_distribution": {
            "answerable": _score_summary([r.top_score for r in answerable]),
            "unanswerable": _score_summary([r.top_score for r in unanswerable]),
        },
        "questions": [asdict(r) for r in results],
    }
