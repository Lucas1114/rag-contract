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

The drafting model is the same vendor as the embedding model, reached the same
way — raw httpx against the REST endpoint, as `embedding.py` documents. That is
one credential for the whole project, used by exactly two commands, both of
which run by hand and commit what they produce. The model is pinned to a dated
snapshot rather than a floating alias for the same reason the index is
content-addressed: a fixture recording `gpt-5.5` could not be re-recorded
reproducibly once that alias moves.

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
model response to a prompt it was never shown. The prompt itself is stored the
same way, as a fingerprint: retrieval moving is a legitimate change that a
replay reports and keeps scoring, but an edited `SYSTEM` means the committed
claims answer a question that was never asked in that form, and a fixture of a
different experiment is not a fixture of this one. That one is refused rather
than reported.

About the prompt
----------------

It asks for grounded claims and tells the model to decline when the passages do
not answer the question, because that is what a competently written RAG prompt
does and anything less would be arguing against a strawman. The guarantee does
not rest on the model following it. Asking is not verifying, and the grounding
check is the verifying.
"""

from __future__ import annotations

import hashlib
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
_KEY_NAME = "OPENAI_API_KEY"

MODEL = "gpt-5.5-2026-04-23"
ENDPOINT = "https://api.openai.com/v1/chat/completions"
MAX_TOKENS = 4000
TIMEOUT_SECONDS = 180

# The drafter must return claims, not prose, or there is nothing to check per
# claim. Enforced by the API rather than by parsing whatever came back.
CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citation": {"type": "string"},
                },
                "required": ["text", "citation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

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


def system_fingerprint(prompt: str = SYSTEM) -> str:
    """The content address of the drafting prompt, recorded with every draft.

    The same idea as the index version and the questions fingerprint: an
    artefact carries a hash of the input it was produced from, so it cannot be
    replayed against a different one without saying so.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


class DrafterError(RuntimeError):
    """A draft could not be produced or replayed."""


def _api_key() -> str:
    """The API key, read the same way `embedding.py` reads it.

    Deliberately duplicated rather than shared: each network-calling module
    reads its own credential at the point of use, so the import graph that
    proves the gate reaches no HTTP client also proves it reaches no key.
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

    `httpx` is imported inside `draft` rather than at module scope, the way
    `embedding.py`'s client is, so `tests/test_ci_contract.py` can keep
    asserting by import graph that nothing the gate touches reaches an HTTP
    client.

    The claim schema is enforced by the API's structured output rather than by
    parsing prose afterwards. A drafter that returns a paragraph gives the
    grounding check nothing to check per claim, so the structure is not a
    convenience — it is what makes the guarantee enforceable at all.
    """

    model: str = MODEL

    def draft(self, question: str, passages: list[Passage]) -> list[Claim]:
        import httpx

        response = httpx.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {_api_key()}"},
            timeout=TIMEOUT_SECONDS,
            json={
                "model": self.model,
                "max_completion_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _prompt(question, passages)},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "drafted_claims",
                        "strict": True,
                        "schema": CLAIM_SCHEMA,
                    },
                },
            },
        )
        if response.status_code != 200:
            raise DrafterError(
                f"drafting {question!r} failed with "
                f"{response.status_code}: {response.text[:400]}"
            )

        choice = response.json()["choices"][0]
        if choice.get("finish_reason") not in (None, "stop"):
            raise DrafterError(
                f"drafting {question!r} stopped on "
                f"{choice['finish_reason']}; the draft would be truncated"
            )
        try:
            drafted = json.loads(choice["message"]["content"])["claims"]
        except (KeyError, TypeError, ValueError) as exc:
            raise DrafterError(
                f"the model returned no parsable draft for {question!r}: {exc}"
            ) from exc
        return [Claim(text=c["text"], citation=c["citation"]) for c in drafted]


@dataclass(frozen=True)
class Recording:
    """One committed draft, and what it was drafted against."""

    question_id: str
    question: str
    model: str
    index_version: str
    system_fingerprint: str  # of the prompt the claims were drafted under
    recorded_at: str
    passages: tuple[str, ...]  # section ids shown to the model, in rank order
    claims: tuple[Claim, ...]

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "model": self.model,
            "index_version": self.index_version,
            "system_fingerprint": self.system_fingerprint,
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
                system_fingerprint=raw["system_fingerprint"],
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
        system_fingerprint=system_fingerprint(),
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

    The drafting prompt is held to the same rule. A recording made under a
    different `SYSTEM` is a recording of a different experiment, and replaying
    it would report this commit's prompt scoring claims that another one
    produced.
    """

    def __init__(self, recordings: list[Recording]):
        self._by_question = {r.question: r for r in recordings}
        if len(self._by_question) != len(recordings):
            raise DrafterError("two draft fixtures record the same question")
        current = system_fingerprint()
        stale = sorted(
            {
                r.system_fingerprint
                for r in recordings
                if r.system_fingerprint != current
            }
        )
        if stale:
            raise DrafterError(
                f"drafts were recorded under prompt {', '.join(stale)}; "
                f"drafter.SYSTEM is now {current}. Rerun "
                "`rag-contract record-drafts`: these claims were written "
                "against a prompt this commit no longer sends."
            )

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
