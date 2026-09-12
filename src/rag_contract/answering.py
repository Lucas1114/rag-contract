"""Failure behaviour: guarantee 3.

Three ways answering fails, three distinct responses, and one rule deciding
between them — the grounding check in `grounding.py`, not a number from the
retriever.

The states
----------

    no_context    nothing was retrieved. The service cannot try.
    unsupported   claims were drafted and none survived grounding.
    partial       claims were drafted and some survived.
    grounded      every claim survived. The success state.

They are distinguished by *where the failure is detected*, which is decidable
per request. The README originally named the third state "low-confidence
retrieval", and that state does not exist: it presumes a similarity band
separating answerable questions from unanswerable ones, and P02 measured the
two distributions overlapping. `partial` replaces it, and is a better state to
have. It is the case where the corpus *mentions* the subject without answering
it — RFC 9110 Section 1.2 says HTTP/2 introduced a multiplexed session layer
and never says how; RFC 9112 Section 9.7 mentions sending a ClientHello and
gives no handshake. Retrieval surfaces those passages with perfectly ordinary
scores, a drafter reads them and fills in the rest from memory, and no
confidence number anywhere in the pipeline is disturbed. That is the failure
this guarantee exists to catch.

The responses
-------------

`grounded` returns the claims, each with its citation.

`partial` returns *only* the claims that survived, and names the ones it
dropped along with the rule that dropped them. The service says less than it
was about to, and says so out loud. A user reading a partial response can see
both what the corpus supports and what the drafter tried to add.

`unsupported` returns no answer at all: a refusal, with the passages that were
consulted and the claims that were rejected. Naming what was consulted matters
more here than anywhere — it is the difference between "I will not answer" and
"here is what I looked at and why none of it answers you".

`no_context` is the only state that is a service error rather than an answer.
Nothing was retrieved, so there is nothing to refuse *from*; the request
failed rather than being declined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .grounding import COVERAGE_FLOOR, Claim, Verdict, check_claims
from .retrieval import Hit


class AnswerState(StrEnum):
    GROUNDED = "grounded"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    NO_CONTEXT = "no_context"

    @property
    def is_failure(self) -> bool:
        return self is not AnswerState.GROUNDED

    @property
    def http_status(self) -> int:
        """`no_context` is the only state that is not a valid answer.

        A refusal is something the service decided, so it is a 200 carrying
        `state: unsupported`. An empty retrieval is something that went wrong.
        """
        return 503 if self is AnswerState.NO_CONTEXT else 200


# What the service says in each failure state. Fixed strings rather than
# generated ones: a refusal written by the same model that just failed to
# ground its claims is not a refusal anyone should trust.
REFUSAL = (
    "This corpus does not answer that question. Nothing drafted from the "
    "passages below could be traced back to them, so no answer is given."
)
NO_CONTEXT_MESSAGE = (
    "No passages were retrieved for this question, so there is nothing to answer from."
)


@dataclass(frozen=True)
class Passage:
    """One retrieved section, as the drafter saw it."""

    section_id: str
    citation: str
    rank: int
    score: float
    text: str


def passages_from_hits(hits: list[Hit]) -> list[Passage]:
    """Collapse retrieved chunks to sections, keeping each section's best rank.

    Citations are section ids, so the drafter is shown sections. A section
    split across several retrieved chunks is one passage holding all of them in
    rank order — the grounding check must see exactly the text the drafter saw,
    or it would reject claims for quoting a part of the section that was on
    screen.
    """
    order: list[str] = []
    parts: dict[str, list[str]] = {}
    best: dict[str, Hit] = {}
    for hit in hits:
        section = hit.section_id
        if section not in parts:
            order.append(section)
            parts[section] = []
            best[section] = hit
        parts[section].append(hit.chunk.text)
    return [
        Passage(
            section_id=section,
            citation=best[section].chunk.citation,
            rank=best[section].rank,
            score=round(best[section].score, 4),
            text="\n\n".join(parts[section]),
        )
        for section in order
    ]


@dataclass(frozen=True)
class Answer:
    """One answered question, in whichever state it ended in."""

    question: str
    state: AnswerState
    index_version: str
    supported: list[Verdict] = field(default_factory=list)
    withdrawn: list[Verdict] = field(default_factory=list)
    consulted: list[Passage] = field(default_factory=list)
    message: str = ""

    @property
    def citations(self) -> list[str]:
        """The passages actually supporting what was said. Empty on refusal."""
        seen: list[str] = []
        for verdict in self.supported:
            if verdict.citation not in seen:
                seen.append(verdict.citation)
        return seen

    def text(self) -> str:
        """The answer as prose, each claim carrying its passage.

        Empty in the two states that return no answer. Nothing is ever said
        without the citation attached, because a claim separated from its
        citation is exactly the artefact the grounding check exists to prevent.
        """
        if not self.supported:
            return ""
        return " ".join(f"{v.text} [{v.citation}]" for v in self.supported)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "state": str(self.state),
            "http_status": self.state.http_status,
            "index_version": self.index_version,
            "answer": self.text() or None,
            "message": self.message,
            "citations": self.citations,
            "supported": [
                {
                    "text": v.text,
                    "citation": v.citation,
                    "coverage": round(v.coverage, 4),
                }
                for v in self.supported
            ],
            "withdrawn": [
                {
                    "text": v.text,
                    "citation": v.citation,
                    "rule": v.rule,
                    "detail": v.detail,
                    "coverage": round(v.coverage, 4),
                }
                for v in self.withdrawn
            ],
            "consulted": [
                {
                    "section_id": p.section_id,
                    "citation": p.citation,
                    "rank": p.rank,
                    "score": p.score,
                }
                for p in self.consulted
            ],
        }


def decide(
    *,
    question: str,
    index_version: str,
    passages: list[Passage],
    claims: list[Claim],
    coverage_floor: float = COVERAGE_FLOOR,
) -> Answer:
    """Grounding check first, state second. This is the whole guarantee.

    Nothing here consults a retrieval score. The state is a function of how
    many drafted claims survived being checked against the passages they
    themselves cited, which is why a question the corpus cannot answer is
    refused even when it retrieves as confidently as one the corpus answers.
    """
    if not passages:
        return Answer(
            question=question,
            state=AnswerState.NO_CONTEXT,
            index_version=index_version,
            message=NO_CONTEXT_MESSAGE,
        )

    texts = {p.section_id: p.text for p in passages}
    verdicts = check_claims(claims, texts, coverage_floor)
    supported = [v for v in verdicts if v.grounded]
    withdrawn = [v for v in verdicts if not v.grounded]

    if not supported:
        # Covers both a drafter that declined outright and one whose every
        # claim was rejected. Same response either way: the service and the
        # check agree there is nothing here to say.
        return Answer(
            question=question,
            state=AnswerState.UNSUPPORTED,
            index_version=index_version,
            withdrawn=withdrawn,
            consulted=passages,
            message=REFUSAL,
        )

    state = AnswerState.GROUNDED if not withdrawn else AnswerState.PARTIAL
    return Answer(
        question=question,
        state=state,
        index_version=index_version,
        supported=supported,
        withdrawn=withdrawn,
        consulted=passages,
        message=(
            ""
            if state is AnswerState.GROUNDED
            else (
                f"{len(withdrawn)} of {len(verdicts)} drafted claims were not "
                "supported by the passages they cited and have been withdrawn."
            )
        ),
    )
