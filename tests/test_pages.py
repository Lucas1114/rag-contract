"""The two HTML routes, and the one thing they are for.

There is nothing here to measure that `test_app.py` does not already measure
about the JSON surface — the pages call the same `Service.answer` and get the
same `Answer`. What these tests pin is that the page does not quietly become a
nicer story than the API tells: a withdrawn claim has to appear on it *with the
rule that withdrew it*, a refusal has to read as a refusal, and the status code
has to be the state's, or the surface says one thing to a person and another to
a client.
"""

import re

import pytest
from fastapi.testclient import TestClient

from rag_contract.answering import AnswerState
from rag_contract.app import REPO_URL, create_app
from rag_contract.evalset import Question
from rag_contract.grounding import Claim
from rag_contract.ratelimit import RateLimiter
from rag_contract.registry import IndexRegistry
from rag_contract.service import Service

from .synthetic import make_index
from .test_app import QUESTION_VECTORS, QUESTIONS, VECTORS, index
from .test_registry import StaticDrafter


class Drafter:
    """Returns exactly the claims a test wants checked."""

    def __init__(self, *claims: Claim):
        self.claims = list(claims)

    def draft(self, question, passages):
        return self.claims


def pages(registry, drafter=None, questions=None):
    service = Service(
        registry=registry,
        questions=questions or QUESTIONS,
        drafter=drafter or StaticDrafter(),
    )
    # Burst well past what any test here sends: the limiter is `test_app.py`'s
    # subject, and a page test failing on an allowance would be measuring the
    # wrong thing.
    return TestClient(
        create_app(service, limiter=RateLimiter(600, 100, clock=lambda: 0.0)),
        raise_server_exceptions=False,
    )


@pytest.fixture
def serving():
    return IndexRegistry(index(["rfc9110#1", "rfc9110#2"], "a" * 12))


def test_the_home_page_lists_the_fixed_set_and_nothing_else(serving):
    questions = {
        "qa": QUESTIONS["qa"],
        "ub": Question(
            id="ub",
            question="how does http/2 multiplex?",
            answerable=False,
            expected_state=AnswerState.UNSUPPORTED,
        ),
    }
    body = pages(serving, questions=questions).get("/").text
    assert "/q/qa" in body
    assert "/q/ub" in body
    assert "how does http/2 multiplex?" in body


def test_the_home_page_links_to_the_source(serving):
    """PROJECT_RULES: a live demo says where it came from, or it is a screenshot."""
    assert REPO_URL in pages(serving).get("/").text


def test_a_page_reports_the_index_it_was_built_from(serving):
    """Guarantee 4 does not stop applying because a human is reading."""
    assert "a" * 12 in pages(serving).get("/q/qa").text


def test_a_withdrawn_claim_appears_with_the_rule_that_withdrew_it(serving):
    """The reason these routes exist.

    A refusal a person cannot see is a refusal they have to take on trust, and
    the whole of guarantee 3 is that they should not have to.
    """
    drafter = Drafter(
        Claim(text="rfc9110#1 says nothing of the sort.", citation="rfc9110#404")
    )
    response = pages(serving, drafter).get("/q/qa")
    assert "citation not retrieved" in response.text
    assert "rfc9110#404" in response.text


def test_a_page_carries_the_states_status_code(serving):
    """A rendered refusal is still a refusal.

    `no_context` is the one state that is not a valid answer, so the page is a
    503 exactly as the JSON is. A 200 here would make the surface disagree with
    itself depending on who was reading it.
    """
    empty = IndexRegistry(
        make_index(
            [], VECTORS[:0], question_ids=["qa"], question_vectors=QUESTION_VECTORS
        )
    )
    assert pages(empty).get("/q/qa").status_code == 503
    assert pages(serving).get("/q/qa").status_code == 200


def test_a_question_outside_the_set_is_a_404_page(serving):
    response = pages(serving).get("/q/nope")
    assert response.status_code == 404
    assert "no free-text input" in response.text


def test_the_pages_are_rate_limited_like_every_other_route(serving):
    """The limiter is middleware, so a route added later is covered by default.

    That default is the reason it is middleware, and this is the test that
    notices if a page is ever mounted outside it.
    """
    service = Service(registry=serving, questions=QUESTIONS, drafter=StaticDrafter())
    api = TestClient(
        create_app(service, limiter=RateLimiter(60, 1, clock=lambda: 0.0)),
        raise_server_exceptions=False,
    )
    assert api.get("/").status_code == 200
    assert api.get("/q/qa").status_code == 429


def test_a_page_serves_no_external_asset(serving):
    """Every byte a visitor gets came out of the process the gate measured.

    A page pulling a font or a script from someone else's host has an
    availability this repository makes no claims about, on a service whose
    entire subject is claims it can hold.
    """
    for path in ("/", "/q/qa", "/q/nope"):
        body = pages(serving).get(path).text
        for marker in ("src=", "<script", "@import"):
            assert marker not in body, f"{path} reaches for something external"
        # The only absolute URL any page may name is where its source lives.
        assert set(re.findall(r"""https?://[^\s"'<>]+""", body)) <= {REPO_URL}
