"""Detecting that the committed index no longer describes the corpus.

The index version is a hash of the corpus, the embedding model and the chunk
parameters, so staleness is not a heuristic: recompute the address from what is
on disk and compare. These tests drive each input separately, because the
useful message is which one changed rather than that something did.
"""

import numpy as np

from rag_contract.chunking import ChunkParams
from rag_contract.corpus import corpus_fingerprint, load_documents
from rag_contract.embedding import MODEL
from rag_contract.evalset import questions_fingerprint
from rag_contract.gate import freshness_check
from rag_contract.index import compute_version, load_index
from rag_contract.lifecycle import index_status

from .synthetic import make_index

VECTORS = np.array([[1.0, 0.0]], dtype=np.float32)


def index_built_from(
    corpus=None, model=MODEL, chunk_words=220, overlap_words=40, questions=None
):
    """A synthetic index whose meta claims it was built from these inputs.

    The recorded version is the address those inputs actually produce, so the
    fixture is self-consistent the way a real `meta.json` is. A faked corpus
    fingerprint is the one exception — a hash cannot be inverted back into the
    documents it came from — and no assertion here depends on that case's
    version.
    """
    index = make_index(["rfc9110#1"], VECTORS)
    meta = index.meta
    documents = load_documents()
    params = ChunkParams(chunk_words=chunk_words, overlap_words=overlap_words)
    return index.__class__(
        meta=meta.__class__(
            **{
                **meta.__dict__,
                "version": compute_version(documents, model, params),
                "corpus_fingerprint": corpus or corpus_fingerprint(documents),
                "embedding_model": model,
                "chunk_words": chunk_words,
                "overlap_words": overlap_words,
                "questions_fingerprint": questions or questions_fingerprint(),
            }
        ),
        chunks=index.chunks,
        vectors=index.vectors,
        question_vectors=index.question_vectors,
        question_ids=index.question_ids,
    )


class TestFresh:
    def test_the_committed_index_describes_the_committed_corpus(self):
        # The claim the gate makes on every build. If this fails, every number
        # in the README is about a corpus that is no longer in the repository.
        status = index_status(load_index())
        assert status.fresh, status.message
        assert status.version == status.expected_version

    def test_an_index_built_from_what_is_on_disk_is_fresh(self):
        assert index_status(index_built_from()).fresh


class TestDivergence:
    def test_a_changed_corpus_is_named(self):
        status = index_status(index_built_from(corpus="0" * 64))
        assert not status.fresh
        assert [d.input for d in status.divergences] == ["corpus"]
        assert "build-index" in status.message

    def test_a_changed_embedding_model_is_named(self):
        status = index_status(index_built_from(model="text-embedding-2-tiny"))
        assert [d.input for d in status.divergences] == ["embedding model"]

    def test_changed_chunk_parameters_are_named(self):
        status = index_status(index_built_from(chunk_words=180))
        assert [d.input for d in status.divergences] == ["chunk parameters"]

    def test_a_changed_question_set_is_named(self):
        status = index_status(index_built_from(questions="0" * 64))
        assert [d.input for d in status.divergences] == ["question set"]

    def test_several_inputs_changing_are_all_reported(self):
        status = index_status(index_built_from(corpus="0" * 64, chunk_words=180))
        assert [d.input for d in status.divergences] == ["corpus", "chunk parameters"]

    def test_the_message_names_the_version_a_rebuild_will_produce(self):
        status = index_status(index_built_from(corpus="0" * 64))
        expected = compute_version(load_documents(), MODEL, ChunkParams())
        assert status.expected_version == expected
        assert expected in status.message


class TestWhichInputsMoveTheVersion:
    """The question set is checked but is not part of the content address.

    Chunk vectors do not depend on `questions.yaml`; the committed *question*
    vectors do. Two artefacts, two reasons to rebuild, one of which does not
    change what the index is called.
    """

    def test_corpus_model_and_chunk_parameters_move_it(self):
        for index in (
            index_built_from(corpus="0" * 64),
            index_built_from(model="other"),
            index_built_from(chunk_words=180),
        ):
            assert all(d.rebuilds_version for d in index_status(index).divergences)

    def test_the_question_set_does_not(self):
        status = index_status(index_built_from(questions="0" * 64))
        assert not any(d.rebuilds_version for d in status.divergences)
        assert status.version == status.expected_version


class TestTheGateCheck:
    def test_a_fresh_index_passes(self):
        assert freshness_check(index_status(load_index())).passed

    def test_a_stale_index_fails_and_says_what_changed(self):
        check = freshness_check(index_status(index_built_from(corpus="0" * 64)))
        assert not check.passed
        assert "corpus" in check.detail

    def test_the_check_holds_no_threshold(self):
        # Unlike every other check in the gate, this one has no bar in
        # eval/thresholds.yaml. There is nothing to set: the committed vectors
        # either were built from this corpus or were not.
        check = freshness_check(index_status(load_index()))
        assert check.observed == check.limit
