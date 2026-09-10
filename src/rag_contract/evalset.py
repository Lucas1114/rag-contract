"""Loading `eval/questions.yaml`.

The question set defines what "correct" means for the whole project, so it is
loaded through a validating reader rather than a bare `yaml.safe_load`: a
malformed or half-edited annotation must fail loudly at load time, not quietly
change a recall number.

The file is also content-addressed. Question vectors are a committed build
artefact just as chunk vectors are, and the eval refuses to score against
vectors built from a different revision of this file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

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
    else:
        if entry.get("expected"):
            raise QuestionSetError(
                f"{qid}: not answerable but carries expected sections"
            )
        if not entry.get("absent"):
            raise QuestionSetError(f"{qid}: not answerable but does not say why")


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
            )
        )
    return questions


def questions_fingerprint(path: Path = QUESTIONS_PATH) -> str:
    """sha256 of the question set as committed.

    Recorded alongside the question vectors so a stale vector file cannot be
    scored against an edited question set.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()
