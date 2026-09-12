"""The cutover: guarantee 4's second half.

The claim under test is that an index can be replaced while requests are
running, and that a request which started before the swap finishes against the
index it started on and reports that version. That is a statement about
concurrency, so the tests that matter here run real threads and hold a real
request open across a real swap. A test that swapped the index between two
sequential calls would assert nothing — the interesting interleaving is the one
where a request is *in the middle* when the swap lands.
"""

import threading

import numpy as np
import pytest

from rag_contract.answering import AnswerState
from rag_contract.evalset import Question
from rag_contract.grounding import Claim
from rag_contract.index import IndexError_
from rag_contract.registry import IndexRegistry
from rag_contract.service import Service, UnknownQuestion

from .synthetic import make_index

VECTORS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
QUESTION_VECTORS = np.array([[1.0, 0.0]], dtype=np.float32)

QUESTIONS = {
    "qa": Question(id="qa", question="what does rfc9110#1 say?", answerable=True)
}


def index_a():
    return make_index(
        ["rfc9110#1", "rfc9110#2"],
        VECTORS,
        question_ids=["qa"],
        question_vectors=QUESTION_VECTORS,
        version="a" * 12,
    )


def index_b():
    """A different index entirely: different sections, different version.

    Nothing it holds overlaps with `index_a`, so a request that drifted onto it
    mid-flight could not produce `index_a`'s passages by coincidence.
    """
    return make_index(
        ["rfc9111#7", "rfc9111#8"],
        VECTORS,
        question_ids=["qa"],
        question_vectors=QUESTION_VECTORS,
        version="b" * 12,
    )


class BlockingIndex:
    """An index that blocks the first time a request touches it.

    The seam has to be here rather than in the drafter. Blocking at the
    drafting step only holds a request open *after* retrieval has finished, so
    a swap landing there cannot be seen by the retrieval half of the request —
    and a service that re-read the registry to fetch its chunks would pass such
    a test while mixing two indexes in one answer.
    """

    def __init__(self, index):
        self._index = index
        self.entered = threading.Event()
        self.release = threading.Event()

    def __getattr__(self, name):
        return getattr(self._index, name)

    def question_vector(self, question_id):
        self.entered.set()
        assert self.release.wait(timeout=5), "the request was never released"
        return self._index.question_vector(question_id)


class StaticDrafter:
    """Claims that ground against whichever section is retrieved first."""

    def draft(self, question, passages):
        del question
        if not passages:
            return []
        top = passages[0]
        return [Claim(text=top.text, citation=top.section_id)]


class BlockingDrafter(StaticDrafter):
    """Holds a request open at the drafting step until it is released.

    The block sits after the index has been acquired, which is exactly where a
    cutover has to be survivable: the request is committed to an index and has
    not finished using it.
    """

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def draft(self, question, passages):
        self.entered.set()
        assert self.release.wait(timeout=5), "the request was never released"
        return super().draft(question, passages)


def service(registry, drafter=None):
    return Service(
        registry=registry, questions=QUESTIONS, drafter=drafter or StaticDrafter()
    )


class TestTheReference:
    def test_an_empty_registry_hands_out_nothing(self):
        registry = IndexRegistry()
        assert registry.acquire() is None
        assert registry.version is None
        assert not registry.loaded

    def test_the_loaded_index_is_what_is_handed_out(self):
        index = index_a()
        registry = IndexRegistry(index)
        assert registry.acquire() is index
        assert registry.version == "a" * 12

    def test_installing_replaces_what_later_requests_get(self):
        registry = IndexRegistry(index_a())
        registry.install(index_b())
        assert registry.version == "b" * 12

    def test_unloading_makes_the_index_unavailable(self):
        registry = IndexRegistry(index_a())
        registry.unload()
        assert registry.acquire() is None
        assert not registry.loaded

    def test_every_generation_is_recorded_with_its_reason(self):
        registry = IndexRegistry(index_a())
        registry.install(index_b(), reason="rebuilt after corpus change")
        registry.unload(reason="withdrawn")
        versions = [(g.version, g.reason) for g in registry.history]
        assert versions == [
            ("a" * 12, "initial load"),
            ("b" * 12, "rebuilt after corpus change"),
            (None, "withdrawn"),
        ]


class TestReload:
    def test_reload_installs_what_is_on_disk(self):
        registry = IndexRegistry()
        registry.reload()
        assert registry.loaded
        assert registry.version is not None

    def test_a_failed_reload_keeps_serving_the_last_good_index(self, tmp_path):
        # The load is what fails, and it fails before the swap. A rebuild that
        # wrote a broken index must not be able to take the service down.
        registry = IndexRegistry(index_a())
        with pytest.raises(IndexError_):
            registry.reload(tmp_path)
        assert registry.version == "a" * 12
        assert len(registry.history) == 1


class TestCutoverDoesNotDropRequests:
    """Both tests below were checked against the bug they exist to catch.

    Reading the version from the registry instead of from the acquired index,
    and re-reading the registry for chunks after the query vector was taken
    from the acquired one, both turn the first test red. What stays green is a
    service that re-acquires *before* touching the index at all, and that is
    correct rather than missed: acquiring twice with nothing in between is the
    same request arriving a moment later.
    """

    def test_a_swap_between_acquiring_and_retrieving_is_not_seen(self):
        # The narrowest window there is: the request holds the index and has
        # not used it yet. A service that re-read the registry to fetch its
        # chunks would answer here with index B's passages under index A's
        # version — one answer attributable to neither.
        blocking = BlockingIndex(index_a())
        registry = IndexRegistry(blocking)
        answers = {}

        def request():
            answers["in_flight"] = service(registry).answer("qa")

        thread = threading.Thread(target=request)
        thread.start()
        assert blocking.entered.wait(timeout=5), "the request never started"
        registry.install(index_b())
        blocking.release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

        answer = answers["in_flight"]
        assert answer.index_version == "a" * 12
        assert [p.section_id for p in answer.consulted] == ["rfc9110#1", "rfc9110#2"]

    def test_a_request_in_flight_finishes_on_the_index_it_started_on(self):
        drafter = BlockingDrafter()
        registry = IndexRegistry(index_a())
        answers = {}

        def request():
            answers["in_flight"] = service(registry, drafter).answer("qa")

        thread = threading.Thread(target=request)
        thread.start()
        assert drafter.entered.wait(timeout=5), "the request never started"

        # The swap lands with the request holding index A.
        registry.install(index_b())
        assert registry.version == "b" * 12

        drafter.release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

        answer = answers["in_flight"]
        assert answer.index_version == "a" * 12
        assert [p.section_id for p in answer.consulted] == ["rfc9110#1", "rfc9110#2"]
        assert answer.state is AnswerState.GROUNDED

    def test_a_request_arriving_after_the_swap_gets_the_new_index(self):
        registry = IndexRegistry(index_a())
        registry.install(index_b())
        answer = service(registry).answer("qa")
        assert answer.index_version == "b" * 12
        assert [p.section_id for p in answer.consulted] == ["rfc9111#7", "rfc9111#8"]

    def test_unloading_mid_request_does_not_drop_the_request(self):
        # The stronger form of the same claim: the index is withdrawn entirely
        # rather than replaced, and the request still completes, because it
        # holds the object and not the registry.
        drafter = BlockingDrafter()
        registry = IndexRegistry(index_a())
        answers = {}

        def request():
            answers["in_flight"] = service(registry, drafter).answer("qa")

        thread = threading.Thread(target=request)
        thread.start()
        assert drafter.entered.wait(timeout=5)
        registry.unload()
        drafter.release.set()
        thread.join(timeout=5)

        assert answers["in_flight"].index_version == "a" * 12
        assert answers["in_flight"].state is AnswerState.GROUNDED

    def test_repeated_swaps_never_mix_two_indexes_in_one_answer(self):
        # Every answer must be attributable to exactly one index: the sections
        # it consulted and the version it reports have to come from the same
        # object, whatever the registry did while it was being served.
        registry = IndexRegistry(index_a())
        results = []
        stop = threading.Event()

        def serve():
            while not stop.is_set():
                results.append(service(registry).answer("qa"))

        workers = [threading.Thread(target=serve) for _ in range(4)]
        for worker in workers:
            worker.start()
        for _ in range(50):
            registry.install(index_b())
            registry.install(index_a())
        stop.set()
        for worker in workers:
            worker.join(timeout=5)

        expected = {
            "a" * 12: ["rfc9110#1", "rfc9110#2"],
            "b" * 12: ["rfc9111#7", "rfc9111#8"],
        }
        assert results
        for answer in results:
            assert [p.section_id for p in answer.consulted] == expected[
                answer.index_version
            ]


class TestNoContextIsReachable:
    """The state P04 defined and could not exercise.

    No question can reach `no_context` against this corpus — brute-force cosine
    over 868 chunks always returns ten. It was never a property of a question.
    An unavailable index is what reaches it, and that lives in the registry.
    """

    def test_an_unloaded_index_answers_no_context(self):
        answer = service(IndexRegistry()).answer("qa")
        assert answer.state is AnswerState.NO_CONTEXT
        assert answer.state.http_status == 503

    def test_it_reports_no_version_because_there_is_no_index(self):
        answer = service(IndexRegistry()).answer("qa")
        assert answer.index_version == ""
        assert answer.to_dict()["index_version"] is None

    def test_withdrawing_the_index_moves_later_requests_into_no_context(self):
        registry = IndexRegistry(index_a())
        assert service(registry).answer("qa").state is AnswerState.GROUNDED
        registry.unload()
        assert service(registry).answer("qa").state is AnswerState.NO_CONTEXT

    def test_no_context_is_not_a_refusal(self):
        # A refusal is a decision the service made about the corpus; this is
        # the service failing to consult it at all. Nothing was consulted and
        # nothing was withdrawn, which is what distinguishes the two.
        answer = service(IndexRegistry()).answer("qa")
        assert answer.consulted == []
        assert answer.withdrawn == []


class TestUnknownQuestion:
    def test_a_question_that_does_not_exist_is_not_a_failure_state(self):
        # There is no open-ended input: a request names a question in the
        # fixed set or it names nothing. That is a 404, not a refusal.
        with pytest.raises(UnknownQuestion):
            service(IndexRegistry(index_a())).answer("nope")

    def test_it_is_raised_before_the_index_is_consulted(self):
        with pytest.raises(UnknownQuestion):
            service(IndexRegistry()).answer("nope")
