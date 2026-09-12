"""Loading `eval/questions.yaml`.

The question set defines what "correct" means for the whole project, so it is
loaded through a validating reader rather than a bare `yaml.safe_load`: a
malformed or half-edited annotation must fail loudly at load time, not quietly
change a recall number.

Refusal questions additionally annotate which of guarantee 3's failure states
they must land in, and the loader treats that as mandatory: a question the
corpus cannot answer is not fully specified by saying it must be refused, only
by saying how. Annotating `grounded` on one is rejected outright — that is the
outcome the guarantee exists to prevent.

The file is also content-addressed. Question vectors are a committed build
artefact just as chunk vectors are, and the eval refuses to score against
vectors built from a different revision of this file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .answering import AnswerState

QUESTIONS_PATH = Path(__file__).resolve().parents[2] / "eval" / "questions.yaml"


class QuestionSetError(RuntimeError):
    """The question set is malformed."""


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answerable: bool
    expected: tuple[str, ...] = ()  # section ids; answerable questions only
    nearest: tuple[str, ...] = ()  # metadata for refusal questions, not scored
    note: str = ""
    absent: str = ""
    # Which failure state a refusal question must land in. Answerable
    # questions are GROUNDED by definition and do not annotate it.
    expected_state: AnswerState = AnswerState.GROUNDED
    state_reason: str = ""
    extra: dict = field(default_factory=dict, repr=False)


def _validate(entry: dict, seen: set[str]) -> None:
    qid = entry.get("id")
    if not qid:
        raise QuestionSetError(f"question with no id: {entry!r}")
    if qid in seen:
        raise QuestionSetError(f"duplicate question id: {qid}")
    if not entry.get("question"):
        raise QuestionSetError(f"{qid}: no question text")
    if "answerable" not in entry:
        raise QuestionSetError(f"{qid}: no answerable flag")
    if entry["answerable"]:
        if not entry.get("expected"):
            raise QuestionSetError(f"{qid}: answerable but no expected sections")
        if entry.get("expected_state"):
            # An answerable question expecting anything but `grounded` would be
            # a contradiction in the annotation, and one annotating `grounded`
            # would be restating the answerable flag in a second place that can
            # drift from it.
            raise QuestionSetError(
                f"{qid}: answerable questions do not annotate expected_state"
            )
    else:
        if entry.get("expected"):
            raise QuestionSetError(
                f"{qid}: not answerable but carries expected sections"
            )
        if not entry.get("absent"):
            raise QuestionSetError(f"{qid}: not answerable but does not say why")
        state = entry.get("expected_state")
        if not state:
            raise QuestionSetError(
                f"{qid}: refuses but does not say which failure state it lands in"
            )
        if state == AnswerState.GROUNDED:
            raise QuestionSetError(
                f"{qid}: expected_state grounded on a question the corpus "
                "cannot answer. That is the outcome guarantee 3 exists to "
                "prevent, so it cannot be the annotated expectation."
            )
        if state not in set(AnswerState):
            raise QuestionSetError(
                f"{qid}: unknown expected_state {state!r}; "
                f"expected one of {', '.join(sorted(set(AnswerState)))}"
            )
        if not entry.get("state_reason"):
            raise QuestionSetError(
                f"{qid}: annotates expected_state but does not say why"
            )


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    raw = yaml.safe_load(path.read_text())
    entries = raw.get("questions") if isinstance(raw, dict) else None
    if not entries:
        raise QuestionSetError(f"{path} defines no questions")

    seen: set[str] = set()
    questions = []
    for entry in entries:
        _validate(entry, seen)
        seen.add(entry["id"])
        questions.append(
            Question(
                id=entry["id"],
                question=entry["question"],
                answerable=bool(entry["answerable"]),
                expected=tuple(entry.get("expected", ())),
                nearest=tuple(entry.get("nearest", ())),
                note=entry.get("note", ""),
                absent=entry.get("absent", ""),
                expected_state=AnswerState(
                    entry.get("expected_state") or AnswerState.GROUNDED
                ),
                state_reason=entry.get("state_reason", ""),
            )
        )
    return questions


def questions_fingerprint(path: Path = QUESTIONS_PATH) -> str:
    """sha256 of the question ids and texts, in file order.

    Recorded alongside the committed question vectors so a stale vector file
    cannot be scored against an edited question set.

    It covers the id and the text of each question and nothing else. Those are
    the only inputs to the vectors it protects: expected sections, notes,
    `expected_state` and the file's comments all change what a question *means*
    to the scorer without changing what was embedded. Hashing the whole file
    instead — which is what this did originally — made every annotation and
    every comment edit demand a rebuild of 868 chunk vectors through a paid
    API, to protect against a change that cannot affect them. A check with a
    cost that large and a yield that small is one that eventually gets deleted.
    """
    material = "\n".join(f"{q.id}\t{q.question}" for q in load_questions(path))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
