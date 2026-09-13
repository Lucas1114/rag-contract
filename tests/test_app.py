"""The HTTP surface, and what it carries about the index.

Guarantee 4 says an index version is returned in response metadata. These are
the tests that make "response" mean a response rather than a dataclass: real
requests through the ASGI app, including one held open across a cutover, and
the 503 that an unavailable index produces.
"""

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_contract.app import create_app
from rag_contract.budget import Budget
from rag_contract.evalset import Question
from rag_contract.ratelimit import RateLimiter
from rag_contract.registry import IndexRegistry
from rag_contract.service import Service

from .synthetic import make_index
from .test_budget import Clock, TickingClock
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


# --- Guarantee 5: what a request spent, reported on the request -------------


def budgeted_client(registry, deadline_ms=150.0, clock=None, drafter=None):
    service = Service(
        registry=registry,
        questions=QUESTIONS,
        drafter=drafter or StaticDrafter(),
        budget=Budget(deadline_ms=deadline_ms, clock=clock or time.perf_counter),
    )
    return TestClient(create_app(service), raise_server_exceptions=False)


def test_an_answer_reports_what_the_request_spent(serving):
    response = budgeted_client(serving).get("/answer/qa")

    assert response.status_code == 200
    budget = response.json()["budget"]
    assert budget["deadline_ms"] == 150.0
    assert budget["exceeded"] is False
    assert set(budget["stages"]) >= {"retrieval", "drafting", "grounding"}
    assert budget["elapsed_ms"] < 150.0


def test_a_request_that_spends_its_budget_is_abandoned_with_a_503(serving):
    """Not a refusal. The service has no verdict on the corpus to report.

    A refusal is a 200 naming the passages it consulted and the claims it
    rejected. This one abandoned the grounding check part way through, so it
    has neither — only what it spent getting nowhere.
    """
    response = budgeted_client(
        serving, deadline_ms=0.001, clock=TickingClock(ms_per_read=5.0)
    ).get("/answer/qa")

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"] == "deadline exceeded"
    assert "state" not in payload
    assert "answer" not in payload
    assert payload["budget"]["exceeded"] is True


def test_an_abandoned_request_still_names_the_index_it_was_serving(serving):
    """The header carries attribution precisely where the body has no answer."""
    response = budgeted_client(
        serving, deadline_ms=0.001, clock=TickingClock(ms_per_read=5.0)
    ).get("/answer/qa")

    assert response.headers["x-index-version"] == "a" * 12
    assert response.json()["index_version"] == "a" * 12


def test_the_abandonment_says_why_a_partial_answer_was_not_given(serving):
    response = budgeted_client(
        serving, deadline_ms=0.001, clock=TickingClock(ms_per_read=5.0)
    ).get("/answer/qa")

    detail = response.json()["detail"]
    assert "abandoned" in detail
    assert "remaining claims" in detail


def test_an_unknown_question_is_still_a_404_under_a_deadline(serving):
    """The budget is acquired before the question is looked up, and neither
    decision is allowed to hide the other."""
    assert budgeted_client(serving).get("/answer/nope").status_code == 404


# --- The rate limit, through the surface that holds it ---------------------
#
# Guarantee 5's third ceiling. It is the one rule on this surface that is not
# decided further down, because the eval harness has no notion of a caller —
# see `app.py`. So it can only be pinned here.


def limited(registry, requests_per_minute=60, burst=3, clock=None):
    service = Service(registry=registry, questions=QUESTIONS, drafter=StaticDrafter())
    limiter = RateLimiter(requests_per_minute, burst, clock=clock or (lambda: 0.0))
    return TestClient(
        create_app(service, limiter=limiter), raise_server_exceptions=False
    )


def test_a_client_past_its_allowance_is_refused_with_a_429(serving):
    api = limited(serving, burst=3)
    assert [api.get("/answer/qa").status_code for _ in range(3)] == [200, 200, 200]
    assert api.get("/answer/qa").status_code == 429


def test_the_refusal_says_when_to_come_back(serving):
    api = limited(serving, requests_per_minute=60, burst=1)
    api.get("/answer/qa")
    refused = api.get("/answer/qa")
    # A header, not just prose in the body: Retry-After is what a client
    # library reads, and a 429 without one is an invitation to spin.
    assert refused.headers["Retry-After"] == "1"
    assert refused.json()["retry_after_s"] == 1


def test_the_refusal_names_the_allowance_it_is_enforcing(serving):
    api = limited(serving, requests_per_minute=60, burst=1)
    api.get("/answer/qa")
    body = api.get("/answer/qa").json()
    assert body["limit"] == {"requests_per_minute": 60, "burst": 1}


def test_a_429_attributes_itself_to_no_index(serving):
    """It is not an answer that failed; it is a request that was never made.

    Every other response on this surface carries `X-Index-Version`, including
    the ones with no answer in them, because attribution is the point of
    guarantee 4. This one carries none — the limiter runs before an index is
    acquired, and naming a version here would attribute a refusal to vectors
    that never saw it.
    """
    api = limited(serving, burst=1)
    api.get("/answer/qa")
    refused = api.get("/answer/qa")
    assert "X-Index-Version" not in refused.headers
    assert "index_version" not in refused.json()


def test_the_limit_covers_health_too(serving):
    """The most expensive route on the surface, and the one with no deadline.

    `/health` re-reads and re-hashes the whole corpus to say whether the index
    still describes it. Exempting it for the sake of liveness probes would
    carve the hole in exactly the shape of the cheapest way to take the
    process.
    """
    api = limited(serving, burst=1)
    assert api.get("/health").status_code == 200
    assert api.get("/health").status_code == 429


def test_the_limit_is_spent_before_a_question_is_looked_up(serving):
    """The limiter runs as middleware, so an unknown question costs a token.

    A 404 that did not count would leave a free route: routing, matching and
    building the response are work whoever asked did not pay for.
    """
    api = limited(serving, burst=1)
    assert api.get("/answer/nope").status_code == 404
    assert api.get("/answer/qa").status_code == 429


def test_two_clients_are_limited_independently_over_http(serving):
    """Per client, over the wire, keyed on the socket peer and nothing else."""
    service = Service(registry=serving, questions=QUESTIONS, drafter=StaticDrafter())
    app = create_app(service, limiter=RateLimiter(60, 1, clock=lambda: 0.0))
    first = TestClient(app, client=("10.0.0.1", 40000))
    second = TestClient(app, client=("10.0.0.2", 40000))

    assert first.get("/answer/qa").status_code == 200
    assert first.get("/answer/qa").status_code == 429
    assert second.get("/answer/qa").status_code == 200


def proxied(registry, hops, burst=1):
    """A surface configured for a deployment `hops` proxies deep."""
    service = Service(registry=registry, questions=QUESTIONS, drafter=StaticDrafter())
    app = create_app(
        service,
        limiter=RateLimiter(60, burst, clock=lambda: 0.0),
        trusted_proxy_hops=hops,
    )
    return TestClient(app, client=("10.0.0.1", 40000))


def test_a_forwarded_for_header_does_not_buy_a_fresh_allowance(serving):
    """With no trusted hop, the header is written by nobody worth believing.

    A limiter any client can step over by setting a header is not one, and this
    is the concrete form of that: the same peer asking twice under two claimed
    identities is still one client.
    """
    api = proxied(serving, hops=0)

    assert (
        api.get("/answer/qa", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    )
    assert (
        api.get("/answer/qa", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429
    )


def test_behind_one_proxy_two_visitors_are_two_clients(serving):
    """The deployment case, and the reason the hop count exists.

    Both requests arrive from the same socket — the proxy — so without the
    committed hop this is one client and the second visitor to a public
    deployment gets a 429.
    """
    api = proxied(serving, hops=1)

    assert (
        api.get("/answer/qa", headers={"X-Forwarded-For": "203.0.113.1"}).status_code
        == 200
    )
    assert (
        api.get("/answer/qa", headers={"X-Forwarded-For": "203.0.113.2"}).status_code
        == 200
    )
    assert (
        api.get("/answer/qa", headers={"X-Forwarded-For": "203.0.113.1"}).status_code
        == 429
    )


def test_a_visitor_cannot_prepend_its_way_out_of_the_limit(serving):
    """What counting from the right buys.

    The proxy appends the peer it saw, so whatever the caller wrote is to the
    left of it. Two requests dressed up as different clients are still one.
    """
    api = proxied(serving, hops=1)

    assert (
        api.get(
            "/answer/qa", headers={"X-Forwarded-For": "1.1.1.1, 203.0.113.1"}
        ).status_code
        == 200
    )
    assert (
        api.get(
            "/answer/qa", headers={"X-Forwarded-For": "2.2.2.2, 203.0.113.1"}
        ).status_code
        == 429
    )


def test_a_request_that_skipped_the_proxy_is_the_socket_peer(serving):
    """No header where one hop is committed: the request did not come through it.

    Falling back to the peer is the conservative direction — that address is
    shared by everything reaching the process directly, so it refuses too much
    rather than handing out an unlimited identity.
    """
    api = proxied(serving, hops=1)

    assert api.get("/answer/qa").status_code == 200
    assert api.get("/answer/qa", headers={"X-Forwarded-For": ""}).status_code == 429


def test_the_surface_takes_its_hop_count_from_the_committed_file(serving):
    """Not passed in by the caller of `create_app`, in the deployed case.

    The allowance and its subject are read from `eval/thresholds.yaml` in the
    same place, so the gate and the runtime cannot disagree about either.
    """
    from rag_contract.gate import load_thresholds

    service = Service(registry=serving, questions=QUESTIONS, drafter=StaticDrafter())
    app = create_app(service)
    assert app.state.trusted_proxy_hops == load_thresholds().budget.trusted_proxy_hops


def test_the_allowance_refills_for_a_client_that_waits(serving):
    clock = Clock()
    api = limited(serving, requests_per_minute=60, burst=1, clock=clock)
    assert api.get("/answer/qa").status_code == 200
    assert api.get("/answer/qa").status_code == 429
    clock.advance_ms(1000)
    assert api.get("/answer/qa").status_code == 200
