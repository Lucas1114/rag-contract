"""Where the model goes, and how it is kept out of CI.

Everything else in this project is arithmetic. This is the one step that is
not: a model reads the retrieved passages and writes the claims the grounding
check then judges. That makes it the one step CI must never run, and the reason
the interface below exists at all.

    Drafter        what the service needs: question + passages -> claims
    LiveDrafter    calls the API. Local only, and the only class here that does
    FixtureDrafter replays a committed recording. What CI uses

The split mirrors the index. Chunk vectors are computed once by a command that
calls an API and committed as a build artefact; drafts are recorded once by a
command that calls an API and committed the same way. Both leave the evaluation
deterministic, free and keyless.

What a fixture does and does not freeze
---------------------------------------

A fixture records the drafter's output and nothing else. Retrieval, the
grounding check and the state machine all rerun on every replay, against the
live index. So a change to the coverage floor, to the chunk parameters, or to
the state rules is *measured* by the answer eval rather than frozen out of it —
which is the property that lets guarantee 3 be gated at all. Freezing the
verdicts instead would produce an eval that could never fail.

The passages the draft was recorded against are stored with it, so a replay
whose retrieval no longer matches can say so rather than quietly scoring a
model response to a prompt it was never shown.

About the prompt
----------------

It asks for grounded claims and tells the model to decline when the passages do
not answer the question, because that is what a competently written RAG prompt
does and anything less would be arguing against a strawman. The guarantee does
not rest on the model following it. Asking is not verifying, and the grounding
check is the verifying.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .answering import Passage
from .grounding import Claim

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "eval" / "fixtures" / "drafts"
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_KEY_NAME = "ANTHROPIC_API_KEY"

MODEL = "claude-opus-5"
MAX_TOKENS = 4000

SYSTEM = """\
You answer questions strictly from the passages you are given, which are \
sections of IETF RFCs.

Return your answer as a list of claims. Each claim is one self-contained \
sentence, and each names the section id of the single passage that supports \
it. Use only the section ids you were given, exactly as written.

If the passages do not answer the question, return an empty list of claims \
rather than answering from your own knowledge. Do not pad an answer with \
background that the passages do not contain.\
"""


class DrafterError(RuntimeError):
    """A draft could not be produced or replayed."""


def _api_key() -> str:
    """The answer key, read the same way the embedding key is.

    Deliberately duplicated rather than shared: each network-calling module
    owns its own credential, so the import graph that proves the gate reaches
    no HTTP client also proves it reaches no key.
    """
    key = os.environ.get(_KEY_NAME, "").strip()
    if not key and _ENV_PATH.is_file():
        for line in _ENV_PATH.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == _KEY_NAME:
                key = value.strip()
                break
    if not key:
        raise DrafterError(
            f"{_KEY_NAME} is not set. Drafts are recorded once and committed; "
            "the answer eval replays those fixtures and needs no key."
        )
    return key


class Drafter(Protocol):
    """Question and passages in, claims out."""

    def draft(self, question: str, passages: list[Passage]) -> list[Claim]: ...


def _prompt(question: str, passages: list[Passage]) -> str:
    blocks = [f"[{p.section_id}] {p.citation}\n{p.text}" for p in passages]
    return "Passages:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQuestion: {question}"


@dataclass(frozen=True)
class LiveDrafter:
    """The only class in the project that calls an answer model.

    Imported inside the recording command rather than at module scope, the way
    the embedding client is, so that `tests/test_ci_contract.py` can keep
    asserting by import graph that the gate reaches no HTTP client.
    """

    model: str = MODEL

    def draft(self, question: str, passages: list[Passage]) -> list[Claim]:
        import anthropic
        from pydantic import BaseModel

        class DraftedClaim(BaseModel):
            text: str
            citation: str

        class DraftedAnswer(BaseModel):
            claims: list[DraftedClaim]

        response = anthropic.Anthropic(api_key=_api_key()).messages.parse(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": _prompt(question, passages)}],
            output_format=DraftedAnswer,
        )
        if response.parsed_output is None:
            raise DrafterError(
                f"the model returned no parsable draft for {question!r} "
                f"(stop_reason {response.stop_reason})"
            )
        return [
            Claim(text=c.text, citation=c.citation)
            for c in response.parsed_output.claims
        ]


@dataclass(frozen=True)
class Recording:
    """One committed draft, and what it was drafted against."""

    question_id: str
    question: str
    model: str
    index_version: str
    recorded_at: str
    passages: tuple[str, ...]  # section ids shown to the model, in rank order
    claims: tuple[Claim, ...]

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "model": self.model,
            "index_version": self.index_version,
            "recorded_at": self.recorded_at,
            "passages": list(self.passages),
            "claims": [{"text": c.text, "citation": c.citation} for c in self.claims],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Recording:
        try:
            return cls(
                question_id=raw["question_id"],
                question=raw["question"],
                model=raw["model"],
                index_version=raw["index_version"],
                recorded_at=raw["recorded_at"],
                passages=tuple(raw["passages"]),
                claims=tuple(
                    Claim(text=c["text"], citation=c["citation"]) for c in raw["claims"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise DrafterError(f"malformed draft fixture: {exc}") from exc


def record(
    *,
    question_id: str,
    question: str,
    passages: list[Passage],
    claims: list[Claim],
    index_version: str,
    model: str,
) -> Recording:
    return Recording(
        question_id=question_id,
        question=question,
        model=model,
        index_version=index_version,
        recorded_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        passages=tuple(p.section_id for p in passages),
        claims=tuple(claims),
    )


def write_recording(recording: Recording, directory: Path = FIXTURES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{recording.question_id}.json"
    path.write_text(json.dumps(recording.to_dict(), indent=2) + "\n")
    return path


def load_recordings(directory: Path = FIXTURES_DIR) -> list[Recording]:
    if not directory.is_dir():
        raise DrafterError(
            f"no draft fixtures at {directory}. Run "
            "`rag-contract record-drafts` (the one command here that calls an "
            "answer model) to record them."
        )
    return [
        Recording.from_dict(json.loads(path.read_text()))
        for path in sorted(directory.glob("*.json"))
    ]


class FixtureDrafter:
    """Replays committed drafts. The drafter CI uses.

    Lookup is by question *text*, not by id. A fixture whose question has been
    reworded no longer answers the question being asked, and replaying it would
    score a model response to a prompt nobody sent; keying on the text turns
    that into a loud failure instead.
    """

    def __init__(self, recordings: list[Recording]):
        self._by_question = {r.question: r for r in recordings}
        if len(self._by_question) != len(recordings):
            raise DrafterError("two draft fixtures record the same question")

    @classmethod
    def from_directory(cls, directory: Path = FIXTURES_DIR) -> FixtureDrafter:
        return cls(load_recordings(directory))

    def recording(self, question: str) -> Recording:
        try:
            return self._by_question[question]
        except KeyError as exc:
            raise DrafterError(
                f"no committed draft for {question!r}. Either the question was "
                "edited after the drafts were recorded, or it is new; rerun "
                "`rag-contract record-drafts`."
            ) from exc

    def draft(self, question: str, passages: list[Passage]) -> list[Claim]:
        del passages  # the recording holds what was actually shown
        return list(self.recording(question).claims)

    def drift(self, question: str, passages: list[Passage]) -> tuple[str, ...]:
        """Sections retrieved now that were not shown when this was recorded.

        Reported rather than raised. Retrieval legitimately moves with the
        chunk parameters, and a replay across such a change is still worth
        scoring — it just has to say that the model never saw these.
        """
        recorded = set(self.recording(question).passages)
        return tuple(p.section_id for p in passages if p.section_id not in recorded)
