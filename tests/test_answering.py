"""The three failure states and their three responses.

The property under test throughout is that the state is a function of the
grounding check and of nothing else. No test here sets a retrieval score that
changes an outcome, because no code path reads one.
"""

from rag_contract.answering import (
    NO_CONTEXT_MESSAGE,
    REFUSAL,
    Answer,
    AnswerState,
    Passage,
    decide,
    passages_from_hits,
)
from rag_contract.grounding import Claim
from rag_contract.retrieval import Hit

from .synthetic import make_chunk

SAFE_METHODS = (
    "Of the request methods defined by this specification, the GET, HEAD, "
    "OPTIONS, and TRACE methods are defined to be safe."
)
MULTIPLEX_MENTION = (
    "HTTP/2 introduced a multiplexed session layer on top of the existing TLS "
    "and TCP protocols for exchanging concurrent HTTP messages with efficient "
    "field compression and server push."
)


def passage(section_id: str, text: str, rank: int = 1, score: float = 0.7) -> Passage:
    return Passage(
        section_id=section_id,
        citation=f"RFC {section_id.split('#')[0][3:]} Section {section_id.split('#')[1]}",
        rank=rank,
        score=score,
        text=text,
    )


def answer_for(passages, claims) -> Answer:
    return decide(
        question="a question",
        index_version="0" * 12,
        passages=passages,
        claims=claims,
    )


class TestNoContext:
    def test_nothing_retrieved_is_not_an_answer_but_a_failure(self):
        result = answer_for([], [Claim("anything", "rfc9110#9.2.1")])
        assert result.state is AnswerState.NO_CONTEXT
        assert result.state.http_status == 503
        assert result.text() == ""
        assert result.message == NO_CONTEXT_MESSAGE

    def test_no_context_precedes_the_grounding_check(self):
        # With no passages every claim would fail rule 1, but reporting that
        # would blame the drafter for the retriever returning nothing.
        result = answer_for([], [Claim("anything", "rfc9110#9.2.1")])
        assert result.withdrawn == []
        assert result.consulted == []


class TestGrounded:
    def test_every_claim_supported_is_the_success_state(self):
        result = answer_for(
            [passage("rfc9110#9.2.1", SAFE_METHODS)],
            [
                Claim(
                    "The safe methods are GET, HEAD, OPTIONS, and TRACE.",
                    "rfc9110#9.2.1",
                )
            ],
        )
        assert result.state is AnswerState.GROUNDED
        assert not result.state.is_failure
        assert result.state.http_status == 200
        assert result.withdrawn == []
        assert result.message == ""

    def test_every_claim_carries_its_passage_in_the_text(self):
        result = answer_for(
            [passage("rfc9110#9.2.1", SAFE_METHODS)],
            [
                Claim(
                    "The safe methods are GET, HEAD, OPTIONS, and TRACE.",
                    "rfc9110#9.2.1",
                )
            ],
        )
        assert result.text().endswith("[rfc9110#9.2.1]")
        assert result.citations == ["rfc9110#9.2.1"]

    def test_citations_are_deduplicated(self):
        result = answer_for(
            [passage("rfc9110#9.2.1", SAFE_METHODS)],
            [
                Claim("The GET method is safe.", "rfc9110#9.2.1"),
                Claim("The HEAD method is safe.", "rfc9110#9.2.1"),
            ],
        )
        assert result.citations == ["rfc9110#9.2.1"]


class TestPartial:
    """The state that replaced "low-confidence retrieval"."""

    def result(self) -> Answer:
        # u01's shape exactly: the corpus says HTTP/2 multiplexes and never
        # says how, so the first claim grounds and the second is memory.
        return answer_for(
            [passage("rfc9110#1.2", MULTIPLEX_MENTION)],
            [
                Claim("HTTP/2 introduced a multiplexed session layer.", "rfc9110#1.2"),
                Claim(
                    "Each request is carried on its own stream, split into "
                    "frames with independent flow control windows.",
                    "rfc9110#1.2",
                ),
            ],
        )

    def test_some_claims_surviving_is_the_partial_state(self):
        assert self.result().state is AnswerState.PARTIAL
        assert self.result().state.is_failure

    def test_the_withdrawn_claim_never_reaches_the_answer_text(self):
        result = self.result()
        assert "frames" not in result.text()
        assert "multiplexed session layer" in result.text()

    def test_the_withdrawn_claim_is_named_with_the_rule_that_dropped_it(self):
        result = self.result()
        assert len(result.withdrawn) == 1
        assert result.withdrawn[0].rule == "coverage below floor"

    def test_the_response_says_out_loud_that_it_said_less(self):
        assert "1 of 2 drafted claims" in self.result().message


class TestUnsupported:
    def test_no_claim_surviving_is_a_refusal(self):
        result = answer_for(
            [passage("rfc9110#9.2.1", SAFE_METHODS)],
            [Claim("A HEADERS frame carries a header block fragment.", "rfc9113#6.2")],
        )
        assert result.state is AnswerState.UNSUPPORTED
        assert result.state.http_status == 200  # a decision, not an error
        assert result.text() == ""
        assert result.citations == []
        assert result.message == REFUSAL

    def test_a_refusal_names_what_it_consulted_and_what_it_rejected(self):
        result = answer_for(
            [passage("rfc9110#9.2.1", SAFE_METHODS)],
            [Claim("A HEADERS frame carries a header block fragment.", "rfc9113#6.2")],
        )
        assert [p.section_id for p in result.consulted] == ["rfc9110#9.2.1"]
        assert result.withdrawn[0].rule == "citation not retrieved"

    def test_a_drafter_that_declines_lands_in_the_same_state(self):
        # Zero claims and zero surviving claims are the same outcome: the
        # drafter and the check agree there is nothing to say.
        result = answer_for([passage("rfc9110#9.2.1", SAFE_METHODS)], [])
        assert result.state is AnswerState.UNSUPPORTED
        assert result.withdrawn == []

    def test_a_confident_retrieval_does_not_rescue_an_ungrounded_answer(self):
        # The whole design decision, as a test. P02 measured u02 retrieving at
        # 0.604 and q20 answering correctly at 0.592; a score band would have
        # to let this through.
        result = answer_for(
            [passage("rfc9110#9.2.1", SAFE_METHODS, rank=1, score=0.999)],
            [Claim("A HEADERS frame contains a stream dependency.", "rfc9110#9.2.1")],
        )
        assert result.state is AnswerState.UNSUPPORTED


class TestPassagesFromHits:
    def hits(self) -> list[Hit]:
        return [
            Hit(rank=1, score=0.8, chunk=make_chunk("rfc9110#9.2.1", 0)),
            Hit(rank=2, score=0.7, chunk=make_chunk("rfc9111#5.2", 0)),
            Hit(rank=3, score=0.6, chunk=make_chunk("rfc9110#9.2.1", 1)),
        ]

    def test_chunks_collapse_to_the_sections_citations_name(self):
        passages = passages_from_hits(self.hits())
        assert [p.section_id for p in passages] == ["rfc9110#9.2.1", "rfc9111#5.2"]

    def test_a_section_keeps_its_best_rank(self):
        passages = passages_from_hits(self.hits())
        assert passages[0].rank == 1

    def test_every_retrieved_chunk_of_a_section_is_in_its_text(self):
        # The check must see exactly what the drafter saw, or it rejects
        # claims for quoting a part of the section that was on screen.
        text = passages_from_hits(self.hits())[0].text
        assert "part 0" in text
        assert "part 1" in text

    def test_no_hits_gives_no_passages(self):
        assert passages_from_hits([]) == []


class TestSerialisation:
    def test_the_dict_carries_state_status_and_index_version(self):
        payload = answer_for(
            [passage("rfc9110#9.2.1", SAFE_METHODS)],
            [
                Claim(
                    "The safe methods are GET, HEAD, OPTIONS, and TRACE.",
                    "rfc9110#9.2.1",
                )
            ],
        ).to_dict()
        assert payload["state"] == "grounded"
        assert payload["http_status"] == 200
        assert payload["index_version"] == "0" * 12

    def test_a_refusal_serialises_a_null_answer(self):
        payload = answer_for(
            [passage("rfc9110#9.2.1", SAFE_METHODS)],
            [Claim("A HEADERS frame carries a fragment.", "rfc9113#6.2")],
        ).to_dict()
        assert payload["answer"] is None
        assert payload["citations"] == []
        assert payload["withdrawn"][0]["rule"] == "citation not retrieved"
