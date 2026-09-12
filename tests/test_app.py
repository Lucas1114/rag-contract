"""The HTTP surface, and what it carries about the index.

Guarantee 4 says an index version is returned in response metadata. These are
the tests that make "response" mean a response rather than a dataclass: real
requests through the ASGI app, including one held open across a cutover, and
the 503 that an unavailable index produces.
"""

import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_contract.app import create_app
from rag_contract.evalset import Question
from rag_contract.registry import IndexRegistry
from rag_contract.service import Service

from .synthetic import make_index
from .test_registry import BlockingDrafter, StaticDrafter

VECTORS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
QUESTION_VECTORS = np.array([[1.0, 0.0]], dtype=np.float32)
QUESTIONS = {
    "qa": Question(id="qa", question="what does rfc9110#1 say?", answerable=True)
}


def index(sections, version):
    return make_index(
        sections,
        VECTORS,
        question_ids=["qa"],
        question_vectors=QUESTION_VECTORS,
        version=version,
    )


def client(registry, drafter=None):
    service = Service(
        registry=registry, questions=QUESTIONS, drafter=drafter or StaticDrafter()
    )
    return TestClient(create_app(service), raise_server_exceptions=False)


@pytest.fixture
def serving():
    return IndexRegistry(index(["rfc9110#1", "rfc9110#2"], "a" * 12))


class TestAnswer:
    def test_an_answer_reports_the_index_it_was_built_from(self, serving):
        response = client(serving).get("/answer/qa")
        assert response.status_code == 200
        assert response.json()["index_version"] == "a" * 12

    def test_the_version_is_in_a_header_too(self, serving):
        # The body is absent from exactly the case where attribution matters
        # most, so the header carries it independently of whether there is an
        # answer to put it next to.
        assert client(serving).get("/answer/qa").headers["X-Index-Version"] == "a" * 12

    def test_the_state_and_its_status_travel_together(self, serving):
        body = client(serving).get("/answer/qa").json()
        assert body["state"] == "grounded"
        assert body["http_status"] == 200

    def test_a_question_outside_the_fixed_set_is_a_404(self, serving):
        response = client(serving).get("/answer/nope")
        assert response.status_code == 404
        assert "takes no free-text input" in response.json()["detail"]

    def test_the_fixed_set_is_what_the_service_offers(self, serving):
        listed = client(serving).get("/questions").json()["questions"]
        assert [q["id"] for q in listed] == ["qa"]


class TestNoIndex:
    def test_an_unavailable_index_answers_503(self):
        response = client(IndexRegistry()).get("/answer/qa")
        assert response.status_code == 503
        assert response.json()["state"] == "no_context"

    def test_it_says_it_has_no_version_rather_than_inventing_one(self):
        response = client(IndexRegistry()).get("/answer/qa")
        assert response.json()["index_version"] is None
        assert response.headers["X-Index-Version"] == "none"

    def test_a_503_is_not_a_refusal(self):
        # `unsupported` is a decision about the corpus and answers 200. This
        # is the service failing to consult the corpus at all.
        body = client(IndexRegistry()).get("/answer/qa").json()
        assert body["consulted"] == []
        assert body["withdrawn"] == []

    def test_health_says_which_index_is_being_served(self, serving):
        health = client(serving).get("/health").json()
        assert health["status"] == "serving"
        assert health["index_version"] == "a" * 12

    def test_health_reports_the_absence_of_an_index(self):
        health = client(IndexRegistry()).get("/health").json()
        assert health["status"] == "no index"
        assert health["index_version"] is None
        assert health["index"] is None


class TestCutoverOverHttp:
    def test_a_request_in_flight_is_answered_from_the_index_it_started_on(
        self, serving
    ):
        # The guarantee stated as a request rather than as a function call: a
        # real HTTP request is open when the swap lands, and it completes with
        # the old version in both the body and the header.
        drafter = BlockingDrafter()
        http = client(serving, drafter)
        responses = {}

        thread = threading.Thread(
            target=lambda: responses.update(before=http.get("/answer/qa"))
        )
        thread.start()
        assert drafter.entered.wait(timeout=5), "the request never reached the app"

        serving.install(index(["rfc9111#7", "rfc9111#8"], "b" * 12))
        drafter.release.set()
        thread.join(timeout=10)
        assert not thread.is_alive(), "the request was dropped by the cutover"

        in_flight = responses["before"]
        assert in_flight.status_code == 200
        assert in_flight.json()["index_version"] == "a" * 12
        assert in_flight.headers["X-Index-Version"] == "a" * 12

        # And the next request sees the new one.
        after = http.get("/answer/qa")
        assert after.json()["index_version"] == "b" * 12

    def test_withdrawing_the_index_mid_request_still_answers_that_request(
        self, serving
    ):
        drafter = BlockingDrafter()
        http = client(serving, drafter)
        responses = {}

        thread = threading.Thread(
            target=lambda: responses.update(before=http.get("/answer/qa"))
        )
        thread.start()
        assert drafter.entered.wait(timeout=5)
        serving.unload()
        drafter.release.set()
        thread.join(timeout=10)

        assert responses["before"].status_code == 200
        assert responses["before"].json()["index_version"] == "a" * 12
        assert http.get("/answer/qa").status_code == 503
