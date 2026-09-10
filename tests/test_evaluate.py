"""Scoring is the measurement guarantee 1 rests on, so the rule is pinned here.

A question is a hit at k when any chunk in the top k carries an expected
section; rank is the position of the first such chunk. The synthetic index
makes those positions arithmetic rather than a property of an embedding model.
"""

import numpy as np
import pytest

from rag_contract.evalset import Question, questions_fingerprint
from rag_contract.evaluate import evaluate, evaluate_question
from rag_contract.index import IndexError_

from .synthetic import make_index


def question(qid, expected=(), answerable=True):
    return Question(
        id=qid,
        question=f"question {qid}",
        answerable=answerable,
        expected=tuple(expected),
        absent="" if answerable else "not in this corpus",
    )


def test_rank_is_the_first_chunk_carrying_an_expected_section(tiny_index):
    result = evaluate_question(tiny_index, question("qa", ["rfc9111#3"]), top_k=4)
    assert result.rank == 3
    assert result.hit_at(3) and result.hit_at(10)
    assert not result.hit_at(2)


def test_a_miss_has_no_rank(tiny_index):
    result = evaluate_question(tiny_index, question("qa", ["rfc6265#9"]), top_k=4)
    assert result.rank is None
    assert not result.hit_at(10)


def test_only_chunks_inside_the_window_can_hit(tiny_index):
    result = evaluate_question(tiny_index, question("qa", ["rfc9111#4"]), top_k=2)
    assert result.rank is None


def test_retrieved_chunks_are_marked_expected(tiny_index):
    result = evaluate_question(tiny_index, question("qa", ["rfc9110#2"]), top_k=4)
    assert [r.expected for r in result.retrieved] == [False, True, False, False]
    assert result.top_section == "rfc9110#1"


def test_report_aggregates_recall_and_mrr():
    # Two chunks; question qa ranks section A first, qb ranks section B first.
    index = make_index(
        ["rfcA#1", "rfcB#1"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        question_ids=["qa", "qb", "qc"],
        question_vectors=np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
        ),
        questions_fingerprint=questions_fingerprint(),
    )

    report = evaluate(
        index,
        [
            question("qa", ["rfcA#1"]),  # rank 1
            question("qb", ["rfcA#1"]),  # rank 2
            question("qc", answerable=False),
        ],
    )
    metrics = report["metrics"]
    assert metrics["answerable_questions"] == 2
    assert metrics["unanswerable_questions"] == 1
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr"] == 0.75  # (1/1 + 1/2) / 2
    assert metrics["misses"] == []


def test_refusal_questions_are_not_scored_for_recall_but_report_their_score():
    index = make_index(
        ["rfcA#1"],
        np.array([[1.0, 0.0]], dtype=np.float32),
        question_ids=["u01"],
        question_vectors=np.array([[0.6, 0.8]], dtype=np.float32),
        questions_fingerprint=questions_fingerprint(),
    )

    report = evaluate(index, [question("u01", answerable=False)])
    assert "recall_at_1" not in report["metrics"]
    distribution = report["score_distribution"]["unanswerable"]
    assert distribution["count"] == 1
    assert distribution["max"] == pytest.approx(0.6, abs=1e-3)


def test_an_edited_question_set_refuses_to_score_against_stale_vectors(tiny_index):
    # tiny_index carries a placeholder fingerprint, not the real file's.
    with pytest.raises(IndexError_, match="build-index"):
        evaluate(tiny_index, [question("qa", ["rfc9110#1"])])
