"""The grounding check, rule by rule.

Every case here is arithmetic over strings. There is no model in this file and
no network, which is the property that lets guarantee 3 be held in CI at all.
"""

from typing import ClassVar

import pytest

from rag_contract.grounding import (
    COVERAGE_FLOOR,
    Claim,
    check_claim,
    check_claims,
    content_words,
    coverage,
    literals,
    strip_citations,
)

SAFE_METHODS = (
    "Of the request methods defined by this specification, the GET, HEAD, "
    "OPTIONS, and TRACE methods are defined to be safe."
)
METHOD_NOT_ALLOWED = (
    "The 405 (Method Not Allowed) status code indicates that the method "
    "received in the request-line is known by the origin server but not "
    "supported by the target resource. The origin server MUST generate an "
    "Allow header field in a 405 response."
)


class TestLiterals:
    def test_numbers_are_literals(self):
        assert "405" in literals("the 405 status code")

    def test_hyphenated_tokens_are_literals(self):
        assert "Set-Cookie" in literals("the Set-Cookie header")

    def test_internal_capitals_are_literals(self):
        assert "HttpOnly" in literals("the HttpOnly attribute")

    def test_all_capitals_are_literals(self):
        assert set(literals("GET and HEAD are safe")) == {"GET", "HEAD"}

    def test_sentence_initial_capital_is_not_a_literal(self):
        # "Allow" here is indistinguishable from a capitalised ordinary word,
        # and treating it as a fact would reject correct claims on punctuation.
        assert literals("Allow header fields are required") == ()

    def test_stopwords_are_never_literals(self):
        assert literals("A response") == ()

    def test_literals_are_deduplicated_in_order(self):
        assert literals("405 then 200 then 405") == ("405", "200")


class TestSelfReferences:
    """A claim naming its own section is pointing, not asserting.

    Measured on the first recorded drafts: asked to cite, the model cited
    twice — once in the citation field and once inside the prose — and 20 of
    21 withdrawals were the check rejecting `rfc9112` and `6.3` for not
    appearing in RFC 9112's own text. Correct claims, rejected for being
    explicit about their source.
    """

    def test_a_section_id_in_the_prose_is_stripped(self):
        assert "rfc9112" not in strip_citations("In rfc9112#6.3, a recipient ...")

    def test_a_spelled_out_reference_is_stripped(self):
        assert "9112" not in strip_citations("RFC 9112 Section 7.1 states that ...")

    def test_a_bare_section_reference_is_stripped(self):
        assert "8" not in strip_citations("Section 8 defines incomplete messages")

    def test_what_the_claim_actually_asserts_survives(self):
        stripped = strip_citations("In rfc9111#3.5, a shared cache MUST NOT store")
        assert "shared cache MUST NOT store" in stripped

    def test_a_real_literal_beside_a_reference_still_counts(self):
        assert literals(
            strip_citations("Under RFC 9110 Section 15.5.6, the code is 405.")
        ) == ("405",)

    def test_a_claim_is_not_rejected_for_citing_itself_in_prose(self):
        passage = {"rfc9110#9.2.1": SAFE_METHODS}
        verdict = check_claim(
            Claim(
                "In rfc9110#9.2.1, the GET, HEAD, OPTIONS, and TRACE methods "
                "are defined to be safe.",
                "rfc9110#9.2.1",
            ),
            passage,
        )
        assert verdict.grounded, verdict.detail


class TestContentWords:
    def test_stopwords_are_dropped(self):
        assert content_words("the method is safe") == ["method", "safe"]

    def test_plurals_fold_to_singulars(self):
        assert content_words("characters") == content_words("character")

    def test_double_s_does_not_fold(self):
        assert content_words("address") == ["address"]

    def test_short_tokens_are_dropped_but_digits_are_kept(self):
        assert content_words("a 2 xx code") == ["2", "code"]


class TestCoverage:
    def test_full_coverage(self):
        assert coverage("safe methods", "the safe methods are listed") == 1.0

    def test_partial_coverage(self):
        assert coverage("safe idempotent methods", "safe methods") == pytest.approx(
            2 / 3
        )

    def test_a_claim_asserting_nothing_covers_vacuously(self):
        # Rules 1 and 2 have already had their say about such a claim.
        assert coverage("the is of", "unrelated text") == 1.0


class TestCheckClaim:
    passages: ClassVar[dict[str, str]] = {
        "rfc9110#9.2.1": SAFE_METHODS,
        "rfc9110#15.5.6": METHOD_NOT_ALLOWED,
    }

    def test_a_supported_claim_is_grounded(self):
        verdict = check_claim(
            Claim(
                "The safe methods are GET, HEAD, OPTIONS, and TRACE.", "rfc9110#9.2.1"
            ),
            self.passages,
        )
        assert verdict.grounded
        assert verdict.rule == ""
        assert verdict.coverage == 1.0

    def test_rule_1_rejects_a_citation_that_was_not_retrieved(self):
        verdict = check_claim(
            Claim("HTTP/2 multiplexes requests over streams.", "rfc9113#5.1"),
            self.passages,
        )
        assert not verdict.grounded
        assert verdict.rule == "citation not retrieved"
        assert "rfc9113#5.1" in verdict.detail

    def test_rule_2_rejects_an_invented_status_code(self):
        # The flagship case. The sentence is fluent, cites a real retrieved
        # passage, and asserts a number that passage never mentions.
        verdict = check_claim(
            Claim("The server responds with 406 in this case.", "rfc9110#15.5.6"),
            self.passages,
        )
        assert not verdict.grounded
        assert verdict.rule == "literal not in passage"
        assert verdict.missing_literals == ("406",)

    def test_rule_2_matches_literals_case_insensitively(self):
        verdict = check_claim(
            Claim(
                "An ALLOW header field is required in a 405 response.", "rfc9110#15.5.6"
            ),
            self.passages,
        )
        assert verdict.grounded, verdict.detail

    def test_rule_3_rejects_a_claim_the_passage_barely_touches(self):
        verdict = check_claim(
            Claim(
                "Safe methods enable prefetching without side effects on origin "
                "resources, so browsers speculatively issue them.",
                "rfc9110#9.2.1",
            ),
            self.passages,
        )
        assert not verdict.grounded
        assert verdict.rule == "coverage below floor"
        assert verdict.coverage < COVERAGE_FLOOR

    def test_rule_1_is_reported_before_rule_2(self):
        # An unretrieved citation makes the literal check meaningless; the
        # reported reason has to be the one a reader can act on.
        verdict = check_claim(
            Claim("The 999 code is defined here.", "rfc9113#5.1"), self.passages
        )
        assert verdict.rule == "citation not retrieved"

    def test_an_empty_passage_set_rejects_everything(self):
        verdict = check_claim(Claim("Anything at all.", "rfc9110#9.2.1"), {})
        assert not verdict.grounded
        assert verdict.rule == "citation not retrieved"

    def test_the_coverage_floor_is_injectable(self):
        claim = Claim(
            "Safe methods enable prefetching without side effects on origin "
            "resources, so browsers speculatively issue them.",
            "rfc9110#9.2.1",
        )
        assert not check_claim(claim, self.passages).grounded
        assert check_claim(claim, self.passages, coverage_floor=0.0).grounded

    def test_check_claims_preserves_order(self):
        verdicts = check_claims(
            [
                Claim(
                    "The safe methods are GET, HEAD, OPTIONS, and TRACE.",
                    "rfc9110#9.2.1",
                ),
                Claim("The server responds with 406.", "rfc9110#15.5.6"),
            ],
            self.passages,
        )
        assert [v.grounded for v in verdicts] == [True, False]
