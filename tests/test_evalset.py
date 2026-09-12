"""Loading and validating the question set.

The question set defines what "correct" means for the whole project, so a
half-edited annotation has to fail at load time rather than quietly change a
number. These tests are about the failing, not the loading.
"""

import textwrap

import pytest
import yaml

from rag_contract.answering import AnswerState
from rag_contract.evalset import (
    QuestionSetError,
    load_questions,
    questions_fingerprint,
)

ANSWERABLE = """
  - id: q01
    question: "Which characters make up the unreserved set in a URI?"
    answerable: true
    expected: ["rfc3986#2.3"]
"""
REFUSAL = """
  - id: u01
    question: "How does HTTP/2 multiplex requests?"
    answerable: false
    expected_state: partial
    state_reason: "RFC 9110 Section 1.2 says that it does and never says how."
    absent: "HTTP/2 is RFC 9113, not in the corpus."
"""


def write(tmp_path, *entries):
    path = tmp_path / "questions.yaml"
    path.write_text("questions:\n" + "".join(textwrap.dedent(e) for e in entries))
    return path


class TestRefusalAnnotations:
    def test_a_refusal_question_must_say_which_state_it_lands_in(self, tmp_path):
        entry = REFUSAL.replace("    expected_state: partial\n", "")
        with pytest.raises(QuestionSetError, match="which failure state"):
            load_questions(write(tmp_path, ANSWERABLE, entry))

    def test_a_refusal_question_must_say_why_it_lands_there(self, tmp_path):
        entry = "\n".join(
            line for line in REFUSAL.splitlines() if "state_reason" not in line
        )
        with pytest.raises(QuestionSetError, match="does not say why"):
            load_questions(write(tmp_path, ANSWERABLE, entry))

    def test_grounded_is_rejected_on_a_question_the_corpus_cannot_answer(
        self, tmp_path
    ):
        # The one annotation that cannot be made. It would specify as correct
        # the exact outcome guarantee 3 exists to prevent.
        entry = REFUSAL.replace("expected_state: partial", "expected_state: grounded")
        with pytest.raises(QuestionSetError, match="grounded"):
            load_questions(write(tmp_path, ANSWERABLE, entry))

    def test_an_unknown_state_is_rejected(self, tmp_path):
        entry = REFUSAL.replace("expected_state: partial", "expected_state: maybe")
        with pytest.raises(QuestionSetError, match="unknown expected_state"):
            load_questions(write(tmp_path, ANSWERABLE, entry))

    def test_an_answerable_question_may_not_annotate_a_state(self, tmp_path):
        entry = ANSWERABLE + "    expected_state: grounded\n"
        with pytest.raises(QuestionSetError, match="do not annotate"):
            load_questions(write(tmp_path, entry, REFUSAL))

    def test_an_answerable_question_is_grounded_by_default(self, tmp_path):
        questions = load_questions(write(tmp_path, ANSWERABLE, REFUSAL))
        assert questions[0].expected_state is AnswerState.GROUNDED

    def test_the_annotation_is_read_as_a_state(self, tmp_path):
        questions = load_questions(write(tmp_path, ANSWERABLE, REFUSAL))
        assert questions[1].expected_state is AnswerState.PARTIAL


class TestCommittedQuestionSet:
    def test_every_refusal_question_is_annotated_and_reasoned(self):
        refusals = [q for q in load_questions() if not q.answerable]
        assert len(refusals) == 8
        for question in refusals:
            assert question.expected_state is not AnswerState.GROUNDED
            assert question.state_reason

    def test_no_question_expects_no_context(self):
        # Brute-force cosine over 868 chunks always returns ten, so no question
        # against this corpus can reach that state. It is held by the unit
        # tests in test_answering.py, and questions.yaml says as much.
        states = {q.expected_state for q in load_questions()}
        assert AnswerState.NO_CONTEXT not in states

    def test_both_refusal_states_are_exercised(self):
        states = {q.expected_state for q in load_questions() if not q.answerable}
        assert states == {AnswerState.PARTIAL, AnswerState.UNSUPPORTED}


class TestFingerprint:
    def test_it_covers_the_question_ids_and_texts(self, tmp_path):
        before = questions_fingerprint(write(tmp_path, ANSWERABLE, REFUSAL))
        edited = ANSWERABLE.replace("unreserved set", "reserved set")
        assert questions_fingerprint(write(tmp_path, edited, REFUSAL)) != before

    def test_it_ignores_what_cannot_change_a_vector(self, tmp_path):
        # Annotations and comments change what a question *means* to the
        # scorer without changing what was embedded. Hashing them would make
        # every annotation edit demand a paid rebuild of 868 chunk vectors to
        # protect against a change that cannot affect them.
        before = questions_fingerprint(write(tmp_path, ANSWERABLE, REFUSAL))
        annotated = ANSWERABLE + '    note: "ALPHA, DIGIT and four marks."\n'
        assert questions_fingerprint(write(tmp_path, annotated, REFUSAL)) == before

    def test_reordering_questions_changes_it(self, tmp_path):
        # Order is the contract between the vectors and the ids that index them.
        before = questions_fingerprint(write(tmp_path, ANSWERABLE, REFUSAL))
        assert questions_fingerprint(write(tmp_path, REFUSAL, ANSWERABLE)) != before

    def test_the_committed_index_matches_the_committed_question_set(self):
        from rag_contract.index import load_index

        assert load_index().meta.questions_fingerprint == questions_fingerprint()


class TestStructuralValidation:
    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda e: e.replace("- id: q01\n", "- id:\n"), "no id"),
            (
                lambda e: e.replace('question: "Which', 'question: ""  # "Which'),
                "no question text",
            ),
            (lambda e: e.replace("    answerable: true\n", ""), "no answerable flag"),
            (
                lambda e: e.replace('expected: ["rfc3986#2.3"]', "expected: []"),
                "no expected sections",
            ),
        ],
    )
    def test_a_malformed_answerable_question_fails_loudly(
        self, tmp_path, mutate, message
    ):
        with pytest.raises(QuestionSetError, match=message):
            load_questions(write(tmp_path, mutate(ANSWERABLE), REFUSAL))

    def test_a_duplicate_id_fails(self, tmp_path):
        with pytest.raises(QuestionSetError, match="duplicate question id"):
            load_questions(write(tmp_path, ANSWERABLE, ANSWERABLE, REFUSAL))

    def test_a_refusal_question_may_not_carry_expected_sections(self, tmp_path):
        entry = REFUSAL + '    expected: ["rfc9110#1.2"]\n'
        with pytest.raises(QuestionSetError, match="carries expected sections"):
            load_questions(write(tmp_path, ANSWERABLE, entry))

    def test_an_empty_file_fails(self, tmp_path):
        path = tmp_path / "questions.yaml"
        path.write_text(yaml.safe_dump({"questions": []}))
        with pytest.raises(QuestionSetError, match="defines no questions"):
            load_questions(path)
